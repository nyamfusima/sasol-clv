"""Sasol CLV baseline: rolling-cutoff snapshots -> customer features -> LightGBM.
Run from the repo root (data files go in data/):
    python src/baseline.py
Validates on the 2025-09-01 snapshot, then retrains on everything and writes
submissions/submission_baseline_v1.csv (its own named, logged output).

Features live in features.py and snapshot building/caching in snapshots.py; this
file is the reference result every variant is measured against, so its numbers
must stay at RMSE fuel 0.6052 | RMSE nonfuel 0.7489 | weighted F1 0.5047.
"""
import argparse, json
import numpy as np, pandas as pd, lightgbm as lgb
from sklearn.metrics import f1_score
import snapshots as S

SEED = 42
TEST_CUTOFF = '2025-12-01'
VAL_CUTOFF = '2025-09-01'
BLOCKS = ('base',)


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
    ap.add_argument('--config', default='src/label_config.json'); ap.add_argument('--out', default='submissions/submission_baseline_v1.csv')
    ap.add_argument('--refresh', action='store_true', help='rebuild the snapshot cache')
    a = ap.parse_args()
    cfg = json.load(open(a.config)); labels = cfg['opportunity_labels']
    # one training snapshot per month: any cutoff whose 3-month outcome ends inside the data
    cutoffs = S.monthly('2024-06-01', VAL_CUTOFF)
    snaps = S.load(cutoffs + [TEST_CUTOFF], a.train, cfg, BLOCKS,
                   label_cutoffs=cutoffs, refresh=a.refresh)
    # validation: train only on snapshots whose outcome window ends before the validation cutoff
    tr = S.trainable_for(cutoffs, VAL_CUTOFF)
    Xtr = pd.concat([snaps[c][0] for c in tr]); ytr = pd.concat([snaps[c][1] for c in tr])
    Xva, yva = snaps[VAL_CUTOFF]
    p = fit_predict(Xtr, ytr, Xva, labels)
    rf = np.sqrt(((p.CLV_fuel - yva.CLV_fuel) ** 2).mean()); rn = np.sqrt(((p.CLV_nonfuel - yva.CLV_nonfuel) ** 2).mean())
    f1 = f1_score(yva.Opportunity, p.Opportunity, average='weighted')
    print(f'VALIDATION ({len(tr)} train snapshots, {len(Xtr)} rows) -> RMSE fuel {rf:.4f} | RMSE nonfuel {rn:.4f} | weighted F1 {f1:.4f}')
    # final: retrain on every snapshot, predict the test cutoff
    Xall = pd.concat([snaps[c][0] for c in cutoffs]); yall = pd.concat([snaps[c][1] for c in cutoffs])
    Xte, _ = snaps[TEST_CUTOFF]
    test_ids = pd.read_csv(a.test, dtype=str).ID
    sub = fit_predict(Xall, yall, Xte, labels).reindex(test_ids)
    assert sub.notna().all().all(), 'some test customers have no features'
    sub.rename_axis('ID').reset_index().to_csv(a.out, index=False)
    print(f'Wrote {a.out}: {len(sub)} rows'); print(sub.Opportunity.value_counts(normalize=True).round(3).head(6).to_string())


if __name__ == '__main__':
    main()
