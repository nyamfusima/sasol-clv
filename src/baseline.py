"""Sasol CLV baseline: rolling-cutoff snapshots -> customer features -> LightGBM.
Run from the repo root (data files go in data/):
    python src/baseline.py
Validates on the 2025-09-01 snapshot, then retrains on everything and writes submission.csv.
"""
import argparse, json
import numpy as np, pandas as pd, lightgbm as lgb
from sklearn.metrics import f1_score
import label_rules as L

SEED = 42
TEST_CUTOFF = '2025-12-01'
VAL_CUTOFF = '2025-09-01'
EXCLUDED = ['Value added services', 'Print media', 'Carrier Bags']

def features(d, cutoff, cats):
    t = pd.Timestamp(cutoff)
    h = d[d._time < t]
    ids = L.eligible_ids(d, t, 3)
    f = pd.DataFrame(index=ids)
    f['cut_month'] = t.month
    f['hist_days'] = (t - d._time.min()).days            # how much history exists at all
    g = h.groupby('ID')._time
    f['recency'] = (t - g.max()).dt.days.reindex(ids)
    f['tenure'] = (t - g.min()).dt.days.reindex(ids)
    f['n_locations'] = h.groupby('ID').LocationId.nunique().reindex(ids)
    f['n_cats_ever'] = h[h.amt > 0].groupby('ID').ItemCategoryLevel2.nunique().reindex(ids)
    b = h.drop_duplicates(['ID', 'TransactionId']).sort_values('_time')
    gap = b.groupby('ID')._time.diff().dt.total_seconds() / 86400
    f['gap_mean'] = gap.groupby(b.ID).mean().reindex(ids)
    f['gap_std'] = gap.groupby(b.ID).std().reindex(ids)
    # quarter-by-quarter history (q1 = most recent 3 months), plus last month and all time
    wins = {f'q{k}': (t - pd.DateOffset(months=3 * k), t - pd.DateOffset(months=3 * (k - 1))) for k in range(1, 5)}
    wins['m1'] = (t - pd.DateOffset(months=1), t)
    wins['all'] = (d._time.min(), t)
    for name, (a, z) in wins.items():
        w = h[(h._time >= a) & (h._time < z)]
        fu, nf = w[w.is_fuel], w[w.is_nonfuel]
        f[f'{name}_baskets'] = w.groupby('ID').TransactionId.nunique().reindex(ids)
        f[f'{name}_fuel_l'] = fu.groupby('ID').qty.sum().reindex(ids)
        f[f'{name}_fuel_r'] = fu.groupby('ID').amt.sum().reindex(ids)
        f[f'{name}_fuel_n'] = fu.groupby('ID').TransactionId.nunique().reindex(ids)
        f[f'{name}_nf_r'] = nf.groupby('ID').amt.sum().reindex(ids)
        f[f'{name}_nf_n'] = nf.groupby('ID').TransactionId.nunique().reindex(ids)
    f = f.fillna({c: 0 for c in f.columns if c[:2] in ('q1', 'q2', 'q3', 'q4', 'm1', 'al')})
    months = np.maximum(f.tenure.fillna(1), 30) / 30.4
    f['fuel_r_per_month'] = f.all_fuel_r / months
    f['nf_r_per_month'] = f.all_nf_r / months
    f['fuel_q1_vs_rate'] = f.q1_fuel_r / (3 * f.fuel_r_per_month + 1)   # low = room to "grow" back
    f['fuel_q1_vs_q2'] = f.q1_fuel_r / (f.q2_fuel_r + 1)
    f['nf_q1_vs_q2'] = f.q1_nf_r / (f.q2_nf_r + 1)
    # per category: spend last quarter, and whether ever bought (drives adoption vs growth)
    q1 = h[h._time >= t - pd.DateOffset(months=3)]
    sp = q1.groupby(['ID', 'ItemCategoryLevel2']).amt.sum().unstack(fill_value=0).reindex(index=ids, columns=cats, fill_value=0)
    ev = h[(h.amt > 0) & (h.qty > 0)].groupby(['ID', 'ItemCategoryLevel2']).TransactionId.nunique().unstack(fill_value=0).reindex(index=ids, columns=cats, fill_value=0)
    f = f.join(sp.add_prefix('q1sp_')).join(ev.add_prefix('ever_'))
    f.columns = [c.replace(' ', '_').replace('&', 'and').replace('-', '_') for c in f.columns]
    return f

def build(d, cutoff, cfg, cats, with_labels=True):
    X = features(d, cutoff, cats)
    if not with_labels:
        return X, None
    y = L.transform(L.raw_labels(d, cutoff, cfg), cfg).set_index('ID').reindex(X.index)
    return X, y

def fit_predict(Xtr, ytr, Xte, labels):
    out = pd.DataFrame(index=Xte.index)
    for col in ['CLV_fuel', 'CLV_nonfuel']:
        m = lgb.LGBMRegressor(n_estimators=500, learning_rate=0.03, num_leaves=31, min_child_samples=30,
                              subsample=0.8, subsample_freq=1, colsample_bytree=0.7, random_state=SEED, verbose=-1)
        out[col] = np.clip(m.fit(Xtr, ytr[col]).predict(Xte), 0, None)
    code = {l: i for i, l in enumerate(labels)}
    m = lgb.LGBMClassifier(n_estimators=400, learning_rate=0.03, num_leaves=15, min_child_samples=30,
                           subsample=0.8, subsample_freq=1, colsample_bytree=0.7, random_state=SEED, verbose=-1)
    m.fit(Xtr, ytr.Opportunity.map(code))
    out['Opportunity'] = np.array(labels)[m.predict(Xte)]
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--train', default='data/train.csv'); ap.add_argument('--test', default='data/test.csv')
    ap.add_argument('--config', default='src/label_config.json'); ap.add_argument('--out', default='submissions/submission.csv')
    a = ap.parse_args()
    cfg = json.load(open(a.config)); labels = cfg['opportunity_labels']
    d = L.load_transactions(a.train)
    d = d[~d.ItemName.isin(cfg['adjustment_items'])].copy()
    d['qty'] = d._quantity / L.SCALE; d['amt'] = d._amount / L.SCALE
    d['is_fuel'] = d.ItemCategoryLevel2.eq('Fuel') & d.LineItemType.eq('Sale_Fuel')
    d['is_nonfuel'] = ~d.ItemCategoryLevel2.isin(['Fuel'] + EXCLUDED)
    cats = sorted(c for c, v in cfg['category_mapping'].items() if v['included'])
    # one training snapshot per month: any cutoff whose 3-month outcome ends inside the data
    cutoffs = [str(c.date()) for c in pd.date_range('2024-06-01', VAL_CUTOFF, freq='MS')]
    snaps = {c: build(d, c, cfg, cats) for c in cutoffs}
    # validation: train only on snapshots whose outcome window ends before the validation cutoff
    tr = [c for c in cutoffs if pd.Timestamp(c) + pd.DateOffset(months=3) <= pd.Timestamp(VAL_CUTOFF)]
    Xtr = pd.concat([snaps[c][0] for c in tr]); ytr = pd.concat([snaps[c][1] for c in tr])
    Xva, yva = snaps[VAL_CUTOFF]
    p = fit_predict(Xtr, ytr, Xva, labels)
    rf = np.sqrt(((p.CLV_fuel - yva.CLV_fuel) ** 2).mean()); rn = np.sqrt(((p.CLV_nonfuel - yva.CLV_nonfuel) ** 2).mean())
    f1 = f1_score(yva.Opportunity, p.Opportunity, average='weighted')
    print(f'VALIDATION ({len(tr)} train snapshots, {len(Xtr)} rows) -> RMSE fuel {rf:.4f} | RMSE nonfuel {rn:.4f} | weighted F1 {f1:.4f}')
    # final: retrain on every snapshot, predict the test cutoff
    Xall = pd.concat([snaps[c][0] for c in cutoffs]); yall = pd.concat([snaps[c][1] for c in cutoffs])
    Xte, _ = build(d, TEST_CUTOFF, cfg, cats, with_labels=False)
    test_ids = pd.read_csv(a.test, dtype=str).ID
    sub = fit_predict(Xall, yall, Xte, labels).reindex(test_ids)
    assert sub.notna().all().all(), 'some test customers have no features'
    sub.rename_axis('ID').reset_index().to_csv(a.out, index=False)
    print(f'Wrote {a.out}: {len(sub)} rows'); print(sub.Opportunity.value_counts(normalize=True).round(3).head(6).to_string())

if __name__ == '__main__':
    main()
