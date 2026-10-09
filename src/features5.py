"""Renewal / fill-rhythm features (sweep 8, blocks R1-R3).

The premise: fuel is a refill good. A customer who fills every 11 days will make
a predictable number of fills in the next quarter, and whether that number is
one more or one fewer than last quarter decides a label that the official rule
defines in rands:

    Fuel growth  <=>  future_fuel_spend > 1.25 * previous_fuel_spend
                      AND  future - previous >= R20
                      AND  Fuel's increment is the largest across categories

(see `raw_labels` in src/label_rules.py). Everything here is built from rows
strictly before the cutoff. Nothing reads the outcome window except its calendar
LENGTH, which is known at prediction time -- the window at 2024-12-01 is 90 days
and at 2025-06-01 it is 92, and a fill-count projection has to know which.

Three blocks:
  r1  fuel fills only (distinct baskets with Sale_Fuel and positive quantity)
  r2  all qualifying baskets, plus a no-purchase estimate aimed at Inactivity
  r3  calendar phase: where in the month the customer fills, and how many
      month-end and month-start days the window contains

A note on which spend definition is used where. Fills are counted on
`is_fuel` (Sale_Fuel line items), matching the user's definition. The
previous-quarter fuel RANDS used in the ratio test come from the whole `Fuel`
category, because that is what the label rule's `previous_spend` matrix sums,
and the point of `r1_hits_line` is to approximate the rule as closely as a
history-only feature can.
"""
import numpy as np, pandas as pd
import features as F

MIN_FILLS = 3          # fewer than this and every rhythm statistic is NaN
LAST_N = 6             # "last 6 fills" means the 5 most recent gaps
EOM_DAYS = (24, 31)    # month-end payday window
BOM_DAYS = (1, 5)


def _windows(t):
    """Exact calendar lengths in days of the previous quarter and the outcome
    window at this cutoff. Both are known without seeing any outcome."""
    prev_start = t - pd.DateOffset(months=3)
    out_end = t + pd.DateOffset(months=3)
    return float((t - prev_start).days), float((out_end - t).days), out_end


def _events(h, mask=None):
    """One row per basket: ID, its timestamp, its rands and its litres."""
    w = h if mask is None else h[mask]
    g = w.groupby(['ID', 'TransactionId'], observed=True)
    e = g.agg(time=('_time', 'min'), amt=('amt', 'sum'), qty=('qty', 'sum'))
    return e.reset_index().sort_values(['ID', 'time'], kind='stable')


def _project(last_days_before, g, prev_days, out_days):
    """Number of k >= 1 with  cutoff <= last_fill + k*gap < window_end.

    `last_days_before` is how long before the cutoff the last fill was, so the
    fill sits at -last_days_before on a clock where the cutoff is 0. The
    condition becomes 0 <= k*g - last <= ... i.e. k >= last/g and
    k < (last + out_days)/g."""
    with np.errstate(divide='ignore', invalid='ignore'):
        kmin = np.maximum(1.0, np.ceil(last_days_before / g))
        kmax = np.ceil((last_days_before + out_days) / g) - 1.0
    return np.maximum(0.0, kmax - kmin + 1.0)


def _rhythm(e, ids, t, prev_days, out_days, out_end, p):
    """Gap statistics, phase and fill-count projections for one event stream.
    `p` prefixes every column so r1 and r2 can share this code."""
    f = pd.DataFrame(index=ids)
    e = e.copy()
    e['gap'] = e.groupby('ID', observed=True).time.diff().dt.total_seconds() / 86400.0
    gaps = e.dropna(subset=['gap'])

    n = e.groupby('ID', observed=True).size().reindex(ids).fillna(0.0)
    f[f'{p}_nfills'] = n
    ga = gaps.groupby('ID', observed=True).gap
    f[f'{p}_gap_med'] = ga.median().reindex(ids)
    f[f'{p}_gap_mean'] = ga.mean().reindex(ids)
    sd = ga.std().reindex(ids)
    f[f'{p}_gap_cv'] = sd / f[f'{p}_gap_mean']

    recent = gaps.groupby('ID', observed=True).tail(LAST_N - 1)
    ra = recent.groupby('ID', observed=True).gap
    f[f'{p}_gap_med6'] = ra.median().reindex(ids)
    f[f'{p}_gap_cv6'] = ra.std().reindex(ids) / ra.mean().reindex(ids)

    last = e.groupby('ID', observed=True).time.max().reindex(ids)
    since = (t - last).dt.total_seconds() / 86400.0
    f[f'{p}_since_last'] = since
    f[f'{p}_phase'] = since / f[f'{p}_gap_med']

    f[f'{p}_prev_days'] = prev_days
    f[f'{p}_out_days'] = out_days

    prev_fills = (e[e.time >= t - pd.DateOffset(months=3)]
                  .groupby('ID', observed=True).size().reindex(ids).fillna(0.0))
    f[f'{p}_prev_fills'] = prev_fills

    sv = since.to_numpy()
    for tag, gcol in (('med', f'{p}_gap_med'), ('m6', f'{p}_gap_med6')):
        gv = f[gcol].to_numpy()
        proj = _project(sv, gv, prev_days, out_days)
        f[f'{p}_proj_{tag}'] = proj
        f[f'{p}_projdiff_{tag}'] = proj - prev_fills.to_numpy()
        with np.errstate(divide='ignore', invalid='ignore'):
            f[f'{p}_projratio_{tag}'] = proj / prev_fills.to_numpy()
        # regularity weighting: an irregular customer's projection means less
        cv = f[f'{p}_gap_cv'].to_numpy()
        f[f'{p}_proj_{tag}_reg'] = proj / (1.0 + cv)

    # median rands and litres per fill over the last 6 fills
    tail = e.groupby('ID', observed=True).tail(LAST_N)
    f[f'{p}_rand_med6'] = tail.groupby('ID', observed=True).amt.median().reindex(ids)
    f[f'{p}_litre_med6'] = tail.groupby('ID', observed=True).qty.median().reindex(ids)

    enough = n >= MIN_FILLS
    return f, enough, out_end


def _growth_line(f, p, proj, rand_med, prev_spend, label):
    """How the projection lands against the official growth test, and how close
    to the line the customer is at one fill more and one fill fewer."""
    spend = proj * rand_med
    f[f'{p}_proj_spend{label}'] = spend
    with np.errstate(divide='ignore', invalid='ignore'):
        f[f'{p}_spend_ratio{label}'] = spend / prev_spend
    for d, nm in ((0, ''), (1, '_p1'), (-1, '_m1')):
        s = np.maximum(0.0, proj + d) * rand_med
        hit = (s > 1.25 * np.maximum(prev_spend, 0.0)) & ((s - prev_spend) >= 20.0)
        f[f'{p}_hits_line{label}{nm}'] = hit.astype(float)
    return f


def build_r1(d, cutoff, ids, cfg=None):
    """Fuel renewal: fills are distinct baskets with a Sale_Fuel line and
    positive quantity."""
    t = pd.Timestamp(cutoff)
    prev_days, out_days, out_end = _windows(t)
    h = d[d._time < t]
    e = _events(h, h.is_fuel & (h.qty > 0))
    f, enough, _ = _rhythm(e, ids, t, prev_days, out_days, out_end, 'r1')

    # previous-quarter fuel rands on the LABEL's definition (whole Fuel
    # category), so the ratio test below approximates the official rule
    fc = h[h.ItemCategoryLevel2.eq('Fuel') & (h._time >= t - pd.DateOffset(months=3))]
    prev_spend = fc.groupby('ID', observed=True).amt.sum().reindex(ids).fillna(0.0)
    f['r1_prev_fuel_r'] = prev_spend
    rm = f['r1_rand_med6'].to_numpy()
    for tag in ('med', 'm6'):
        f = _growth_line(f, 'r1', f[f'r1_proj_{tag}'].to_numpy(), rm,
                         prev_spend.to_numpy(), f'_{tag}')
    f['r1_proj_litres_med'] = f['r1_proj_med'] * f['r1_litre_med6']
    f['r1_proj_litres_m6'] = f['r1_proj_m6'] * f['r1_litre_med6']
    return _finish(f, enough, keep=['r1_nfills', 'r1_prev_fills', 'r1_prev_fuel_r',
                                    'r1_prev_days', 'r1_out_days'])


def build_r2(d, cutoff, ids, cfg=None):
    """The same rhythm over ALL qualifying baskets, plus a no-purchase estimate.

    `r2_noprob` is exp(-W/g): the chance of no arrival in a window of W days for
    a memoryless process with mean gap g. `r2_noprob_phase` charges the window
    only from the moment the next purchase was already due, i.e. it replaces W
    by W - max(0, g - since_last), so a customer who is already overdue gets a
    lower no-purchase estimate than one who filled yesterday. Inactivity is the
    label this is aimed at."""
    t = pd.Timestamp(cutoff)
    prev_days, out_days, out_end = _windows(t)
    h = d[d._time < t]
    e = _events(h, (h.qty > 0) & (h.amt > 0))
    f, enough, _ = _rhythm(e, ids, t, prev_days, out_days, out_end, 'r2')

    g = f['r2_gap_med'].to_numpy()
    since = f['r2_since_last'].to_numpy()
    with np.errstate(divide='ignore', invalid='ignore'):
        f['r2_noprob'] = np.exp(-out_days / g)
        residual = np.maximum(0.0, g - since)
        f['r2_noprob_phase'] = np.exp(-np.maximum(0.0, out_days - residual) / g)
    return _finish(f, enough, keep=['r2_nfills', 'r2_prev_fills', 'r2_prev_days',
                                    'r2_out_days'])


def build_r3(d, cutoff, ids, cfg=None):
    """Calendar phase. The share of a customer's fuel fills that land in the
    month-end and month-start paydays, and how many such days the outcome window
    contains against the previous quarter -- these last are the same for every
    customer at a cutoff, and exist so the model can tell a 90-day window from a
    92-day one."""
    t = pd.Timestamp(cutoff)
    prev_days, out_days, out_end = _windows(t)
    h = d[d._time < t]
    e = _events(h, h.is_fuel & (h.qty > 0))
    f = pd.DataFrame(index=ids)
    dom = e.time.dt.day
    n = e.groupby('ID', observed=True).size().reindex(ids)
    for nm, (lo, hi) in (('eom', EOM_DAYS), ('bom', BOM_DAYS)):
        k = (e[(dom >= lo) & (dom <= hi)].groupby('ID', observed=True).size()
             .reindex(ids).fillna(0.0))
        f[f'r3_share_{nm}'] = (k / n).where(n >= MIN_FILLS)
    prev_rng = pd.date_range(t - pd.DateOffset(months=3), t, inclusive='left')
    out_rng = pd.date_range(t, out_end, inclusive='left')
    for nm, (lo, hi) in (('eom', EOM_DAYS), ('bom', BOM_DAYS)):
        po = int(((prev_rng.day >= lo) & (prev_rng.day <= hi)).sum())
        oo = int(((out_rng.day >= lo) & (out_rng.day <= hi)).sum())
        f[f'r3_{nm}_prev'] = float(po)
        f[f'r3_{nm}_out'] = float(oo)
        f[f'r3_{nm}_diff'] = float(oo - po)
    f['r3_days_diff'] = out_days - prev_days
    f.columns = F._clean(f.columns)
    return f.replace([np.inf, -np.inf], np.nan)


def _finish(f, enough, keep):
    """Blank every rhythm statistic for customers with too few fills. Counts and
    cutoff-level window lengths stay, because 0 fills is a fact, not a gap."""
    cols = [c for c in f.columns if c not in keep]
    f.loc[~enough.reindex(f.index).fillna(False).to_numpy(), cols] = np.nan
    f.columns = F._clean(f.columns)
    return f.replace([np.inf, -np.inf], np.nan)


# R3 mixes two kinds of column and they carry very different risk, so they can
# be tested apart. The share features are per customer. The window counts are
# identical for every customer at a cutoff, take only 9 distinct values across
# the 17 cutoffs, and pair the test cutoff uniquely with 2024-12-01 -- the same
# calendar month -- so a tree can use them to identify the cutoff and specialise
# on one December window rather than learning anything about renewal.
R3_CUSTOMER = ['r3_share_eom', 'r3_share_bom']


def build_r3a(d, cutoff, ids, cfg=None):
    """R3's customer-level share features only."""
    return build_r3(d, cutoff, ids, cfg)[R3_CUSTOMER]


def build_r3b(d, cutoff, ids, cfg=None):
    """R3's cutoff-level window counts only. Three of these (the month-start
    counts) are constant across every cutoff and cannot be split on at all."""
    return build_r3(d, cutoff, ids, cfg).drop(columns=R3_CUSTOMER)


BUILDERS = {'r1': build_r1, 'r2': build_r2, 'r3': build_r3,
            'r3a': build_r3a, 'r3b': build_r3b}
