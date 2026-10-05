"""Customer features at a cutoff.

`base` is the exact feature set baseline.py shipped with; the other blocks are
opt-in so every experiment can be scored against the same reference. Blocks are
selected by name, e.g. build(d, cutoff, cats, blocks=('base', 'growth')).
"""
import numpy as np, pandas as pd
import label_rules as L

# keys of category_mapping that drive their own (non-Other) growth label
GROWTH_CATS = ['Fuel', 'Beverages', 'Food & beverage']
BLOCKS = ('base', 'growth', 'recency', 'season')


def _clean(cols):
    return [c.replace(' ', '_').replace('&', 'and').replace('-', '_') for c in cols]


def base(f, d, h, t, ids, cats):
    """Verbatim from the original baseline.features (do not change: the baseline
    numbers are the reference every variant is measured against)."""
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
    return f.join(sp.add_prefix('q1sp_')).join(ev.add_prefix('ever_'))


def growth(f, d, h, t, ids, cats):
    """(a) How the previous quarter compares with the customer's own earlier
    quarters, and how far short of the growth rule next quarter would fall.

    Growth needs future > 1.25 * prev AND future - prev >= R20, so the binding
    threshold is max(1.25*prev, prev+20). Express it relative to what the
    customer normally spends: a small ratio means the rule is easy to clear."""
    # per-category net spend in each of the four quarters before the cutoff
    qsp = {}
    for k in range(1, 5):
        a, z = t - pd.DateOffset(months=3 * k), t - pd.DateOffset(months=3 * (k - 1))
        w = h[(h._time >= a) & (h._time < z)]
        qsp[k] = (w.groupby(['ID', 'ItemCategoryLevel2']).amt.sum().unstack(fill_value=0)
                  .reindex(index=ids, columns=cats, fill_value=0))
    for c in GROWTH_CATS:
        n = c.replace(' ', '_').replace('&', 'and').replace('-', '_')
        prev = qsp[1][c]
        older = pd.concat([qsp[k][c] for k in (2, 3, 4)], axis=1)
        avg = older.mean(axis=1)                      # own average of earlier quarters
        mx = older.max(axis=1)
        f[f'g_{n}_prev'] = prev
        f[f'g_{n}_avg_older'] = avg
        f[f'g_{n}_prev_vs_avg'] = prev / (avg + 1)    # <1 = last quarter was unusually low
        f[f'g_{n}_prev_vs_max'] = prev / (mx + 1)
        f[f'g_{n}_nonzero_q'] = (older > 0).sum(axis=1) + (prev > 0).astype(int)
        need = np.maximum(1.25 * prev.clip(lower=0), prev + 20)   # spend needed to qualify
        f[f'g_{n}_need'] = need
        f[f'g_{n}_need_gap'] = need - prev                        # rands of headroom required
        f[f'g_{n}_need_vs_avg'] = need / (avg + 1)                # <=1 = typical quarter clears it
        f[f'g_{n}_need_vs_max'] = need / (mx + 1)
        # at prev == 0 the rule reduces to "spend R20", which is why returners qualify
        f[f'g_{n}_prev_zero'] = (prev <= 0).astype(int)
    # same idea on the two regression aggregates
    for n, col in [('fuel_r', 'is_fuel'), ('nf_r', 'is_nonfuel')]:
        qs = []
        for k in range(1, 5):
            a, z = t - pd.DateOffset(months=3 * k), t - pd.DateOffset(months=3 * (k - 1))
            w = h[(h._time >= a) & (h._time < z)]
            qs.append(w[w[col]].groupby('ID').amt.sum().reindex(ids).fillna(0))
        older = pd.concat(qs[1:], axis=1)
        f[f'g_tot_{n}_prev_vs_avg'] = qs[0] / (older.mean(axis=1) + 1)
        f[f'g_tot_{n}_prev_vs_max'] = qs[0] / (older.max(axis=1) + 1)
    return f


def recency(f, d, h, t, ids, cats):
    """(b) Short-window activity, per-stream recency, worst gap and a 6-month slope."""
    b = h.drop_duplicates(['ID', 'TransactionId'])
    for days in (7, 14, 30, 60):
        w = h[h._time >= t - pd.Timedelta(days=days)]
        f[f'd{days}_baskets'] = w.groupby('ID').TransactionId.nunique().reindex(ids).fillna(0)
        f[f'd{days}_fuel_r'] = w[w.is_fuel].groupby('ID').amt.sum().reindex(ids).fillna(0)
        f[f'd{days}_fuel_l'] = w[w.is_fuel].groupby('ID').qty.sum().reindex(ids).fillna(0)
        f[f'd{days}_nf_r'] = w[w.is_nonfuel].groupby('ID').amt.sum().reindex(ids).fillna(0)
    # days since last purchase, split by stream (missing = never bought that stream)
    for n, col in [('fuel', 'is_fuel'), ('shop', 'is_nonfuel')]:
        w = h[h[col] & (h.amt > 0)]
        f[f'since_{n}'] = (t - w.groupby('ID')._time.max()).dt.days.reindex(ids)
    bs = b.sort_values('_time')
    gap = bs.groupby('ID')._time.diff().dt.total_seconds() / 86400
    f['gap_max'] = gap.groupby(bs.ID).max().reindex(ids)           # longest quiet stretch
    f['gap_last_vs_max'] = f.recency / (f.gap_max + 1)             # >1 = quieter than ever before
    f['gap_last_vs_mean'] = f.recency / (f.gap_mean + 1)
    # OLS slope of monthly totals over the last 6 months (x = 0..5, zero-filled)
    w = h[h._time >= t - pd.DateOffset(months=6)].copy()
    mo = ((w._time.dt.year - (t - pd.DateOffset(months=6)).year) * 12
          + w._time.dt.month - (t - pd.DateOffset(months=6)).month).clip(0, 5)
    w['_mo'] = mo
    x = np.arange(6)
    xc = x - x.mean()
    denom = (xc ** 2).sum()
    for n, sel, val in [('baskets', slice(None), None), ('fuel_r', 'is_fuel', 'amt'), ('nf_r', 'is_nonfuel', 'amt')]:
        ww = w if val is None else w[w[sel]]
        if val is None:
            m = ww.groupby(['ID', '_mo']).TransactionId.nunique()
        else:
            m = ww.groupby(['ID', '_mo'])[val].sum()
        m = m.unstack(fill_value=0).reindex(index=ids, columns=x, fill_value=0)
        f[f'slope6_{n}'] = (m.to_numpy() * xc).sum(axis=1) / denom
        f[f'last6_{n}'] = m.to_numpy().sum(axis=1)
    return f


def season(f, d, h, t, ids, cats):
    """(c) The same three calendar months one year earlier: what the customer
    spent then, and what the label rules would have called them. NaN where the
    history does not reach back that far (the model reads it as 'unknown')."""
    a, z = t - pd.DateOffset(months=12), t - pd.DateOffset(months=9)
    have = a >= d._time.min()                      # is that window inside the data at all?
    w = h[(h._time >= a) & (h._time < z)]
    nan = pd.Series(np.nan, index=ids)
    if not have or w.empty:
        for c in ['sy_baskets', 'sy_fuel_l', 'sy_fuel_r', 'sy_nf_r', 'sy_fuel_vs_q1', 'sy_nf_vs_q1']:
            f[c] = nan
        f['sy_label'] = np.nan
        f['sy_has'] = 0
        return f
    f['sy_baskets'] = w.groupby('ID').TransactionId.nunique().reindex(ids).fillna(0)
    f['sy_fuel_l'] = w[w.is_fuel].groupby('ID').qty.sum().reindex(ids).fillna(0)
    f['sy_fuel_r'] = w[w.is_fuel].groupby('ID').amt.sum().reindex(ids).fillna(0)
    f['sy_nf_r'] = w[w.is_nonfuel].groupby('ID').amt.sum().reindex(ids).fillna(0)
    f['sy_fuel_vs_q1'] = f.sy_fuel_r / (f.q1_fuel_r + 1)
    f['sy_nf_vs_q1'] = f.sy_nf_r / (f.q1_nf_r + 1)
    f['sy_has'] = 1
    return f


def season_label(d, cutoff, cfg, ids):
    """The label the rules give at the cutoff one year earlier, as a code.
    Separate from season() because it needs the full frame and the config."""
    t = pd.Timestamp(cutoff)
    prior = t - pd.DateOffset(months=12)
    if prior < d._time.min() + pd.Timedelta(days=1):
        return pd.Series(np.nan, index=ids)
    labels = cfg['opportunity_labels']
    code = {l: i for i, l in enumerate(labels)}
    r = L.raw_labels(d, prior, cfg).set_index('ID')
    return r.Opportunity.map(code).reindex(ids)


def build(d, cutoff, cats, blocks=('base',), cfg=None):
    t = pd.Timestamp(cutoff)
    h = d[d._time < t]
    ids = L.eligible_ids(d, t, 3)
    f = pd.DataFrame(index=ids)
    for name in BLOCKS:                      # fixed order => stable column order
        if name in blocks:
            f = globals()[name](f, d, h, t, ids, cats)
    if 'season' in blocks and cfg is not None:
        f['sy_label'] = season_label(d, cutoff, cfg, ids)
    f.columns = _clean(f.columns)
    return f
