"""Block F features: columns the baseline never looked at.

Groups are independent so each can be added on top of the baseline one at a
time. Everything here reads only rows strictly before the cutoff, except
`site_rates`, which reads earlier snapshots' OUTCOMES and is therefore built
forward in time by `site_rate_table` (never from the validation window).

Unlike features.py this module needs the adjustment rows (F5) and a few columns
the baseline parses away, so it has its own loader.
"""
import numpy as np, pandas as pd
import label_rules as L
import features as F

GROUPS = ('f1', 'f2', 'f3', 'f4', 'f5')
FUEL_GRADES = ['ULP95', 'ULP93', 'Diesel 10ppm', 'Diesel 50ppm']
ID_BASE = 4001000000000
HOUR_BANDS = [('morning', 5, 11), ('midday', 11, 16), ('evening', 16, 21)]


def load_rich(path, cfg):
    """(d, dall): d is exactly what snapshots.load_data gives (adjustment rows
    dropped); dall keeps every row plus the extra parsed columns F3/F5 need."""
    dall = L.load_transactions(path)
    dall['qty'] = dall._quantity / L.SCALE
    dall['amt'] = dall._amount / L.SCALE
    dall['is_fuel'] = dall.ItemCategoryLevel2.eq('Fuel') & dall.LineItemType.eq('Sale_Fuel')
    dall['is_nonfuel'] = ~dall.ItemCategoryLevel2.isin(
        ['Fuel', 'Value added services', 'Print media', 'Carrier Bags'])
    dall['unit_price'] = pd.to_numeric(dall.UnitSellingExclPrice, errors='coerce')
    dall['is_voucher'] = dall.LineItemType.eq('Sale_Voucher')
    dall['is_adj'] = dall.ItemName.isin(cfg['adjustment_items'])
    d = dall[~dall.is_adj].copy()
    return d, dall


# --- F1 customer id ---------------------------------------------------------

def f1(f, d, dall, h, t, ids):
    n = pd.Series(ids.astype('int64') - ID_BASE, index=ids)
    f['id_num'] = n
    f['id_rank'] = n.rank(pct=True)               # position among the IDs present
    return f


# --- F2 sites ---------------------------------------------------------------

def home_site(h, ids):
    """Most-used site before the cutoff, by distinct baskets."""
    b = h.drop_duplicates(['ID', 'TransactionId'])
    c = b.groupby(['ID', 'LocationId']).size()
    if c.empty:
        return pd.Series(index=ids, dtype=object)
    # highest basket count, ties broken by LocationId for determinism
    c = c.reset_index(name='n').sort_values(['ID', 'n', 'LocationId'],
                                            ascending=[True, False, True])
    return c.groupby('ID').LocationId.first().reindex(ids)


def f2(f, d, dall, h, t, ids):
    b = h.drop_duplicates(['ID', 'TransactionId'])
    hs = home_site(h, ids)
    f['home_site'] = pd.Categorical(hs).codes.astype(float)
    nb = b.groupby('ID').size().reindex(ids)
    at_home = (b.LocationId.to_numpy() ==
               hs.reindex(b.ID).to_numpy()).astype(float)
    f['home_share'] = (pd.Series(at_home, index=b.ID.to_numpy())
                       .groupby(level=0).sum().reindex(ids) / nb)
    q1 = b[b._time >= t - pd.DateOffset(months=3)]
    f['sites_q1'] = q1.groupby('ID').LocationId.nunique().reindex(ids).fillna(0)
    f['sites_ever'] = b.groupby('ID').LocationId.nunique().reindex(ids)
    return f


def site_rate_table(home_by_cut, label_by_cut, earlier, prior=20.0):
    """Site-level outcome rates from EARLIER snapshots only, smoothed toward the
    global mean so sites with few observations are not trusted. `earlier` is the
    list of cutoffs whose outcome windows end on or before the target cutoff."""
    if not earlier:
        return None
    rows = []
    for c in earlier:
        hs, lab = home_by_cut[c], label_by_cut[c]
        rows.append(pd.DataFrame({'site': hs.reindex(lab.index),
                                  'inact': lab.Opportunity.eq('Inactivity').astype(float),
                                  'fgrow': lab.Opportunity.eq(
                                      'Existing-category growth: Fuel').astype(float),
                                  'fuel_l': lab.fuel_litres.astype(float)}))
    a = pd.concat(rows).dropna(subset=['site'])
    g = a.groupby('site').agg(n=('inact', 'size'), inact=('inact', 'mean'),
                              fgrow=('fgrow', 'mean'), fuel_l=('fuel_l', 'mean'))
    gm = a[['inact', 'fgrow', 'fuel_l']].mean()
    for col in ('inact', 'fgrow', 'fuel_l'):
        g[col] = (g[col] * g.n + gm[col] * prior) / (g.n + prior)
    return g


def f2_rates(f, hs, table):
    cols = ['site_inact_rate', 'site_fuelgrowth_rate', 'site_mean_fuel_l', 'site_n_obs']
    if table is None:
        for c in cols:
            f[c] = np.nan
        return f
    m = table.reindex(hs.to_numpy())
    f['site_inact_rate'] = m.inact.to_numpy()
    f['site_fuelgrowth_rate'] = m.fgrow.to_numpy()
    f['site_mean_fuel_l'] = m.fuel_l.to_numpy()
    f['site_n_obs'] = m.n.to_numpy()
    return f


# --- F3 fuel type and price -------------------------------------------------

def f3(f, d, dall, h, t, ids):
    fu = h[h.is_fuel]
    tot = fu.groupby('ID').qty.sum().reindex(ids)
    for g in FUEL_GRADES:
        n = g.replace(' ', '_')
        f[f'grade_share_{n}'] = (fu[fu.ItemName.eq(g)].groupby('ID').qty.sum()
                                 .reindex(ids).fillna(0) / tot)
    for l3 in ('Petrol', 'Diesel'):
        f[f'l3_share_{l3}'] = (fu[fu.ItemCategoryLevel3.eq(l3)].groupby('ID').qty.sum()
                               .reindex(ids).fillna(0) / tot)
    fills = fu.groupby('ID').TransactionId.nunique().reindex(ids)
    f['litres_per_fill'] = tot / fills
    f['fills_ever'] = fills
    # the price each customer actually paid, last quarter vs the one before
    for k, name in ((1, 'q1'), (2, 'q2')):
        w = fu[(fu._time >= t - pd.DateOffset(months=3 * k))
               & (fu._time < t - pd.DateOffset(months=3 * (k - 1)))]
        f[f'price_{name}'] = w.groupby('ID').unit_price.mean().reindex(ids)
    f['price_change'] = f.price_q1 - f.price_q2
    f['price_ratio'] = f.price_q1 / f.price_q2
    # market-wide fuel price in each of the last 3 months (same for everyone)
    for k in (1, 2, 3):
        w = h[h.is_fuel & (h._time >= t - pd.DateOffset(months=k))
              & (h._time < t - pd.DateOffset(months=k - 1))]
        f[f'mkt_price_m{k}'] = w.unit_price.mean() if len(w) else np.nan
    f['mkt_price_trend'] = f.mkt_price_m1 - f.mkt_price_m3
    return f


# --- F4 timing --------------------------------------------------------------

def f4(f, d, dall, h, t, ids):
    b = h.drop_duplicates(['ID', 'TransactionId']).copy()
    nb = b.groupby('ID').size().reindex(ids)
    b['dow'] = b._time.dt.dayofweek
    b['hr'] = b._time.dt.hour
    f['weekend_share'] = (b[b.dow >= 5].groupby('ID').size().reindex(ids).fillna(0) / nb)
    for name, lo, hi in HOUR_BANDS:
        f[f'hour_{name}'] = (b[(b.hr >= lo) & (b.hr < hi)].groupby('ID').size()
                             .reindex(ids).fillna(0) / nb)
    f['hour_night'] = 1 - f[[f'hour_{n}' for n, _, _ in HOUR_BANDS]].sum(axis=1)
    f['dom_mean'] = b.groupby('ID')._time.apply(lambda s: s.dt.day.mean()).reindex(ids)
    f['dom_std'] = b.groupby('ID')._time.apply(lambda s: s.dt.day.std()).reindex(ids)
    f['hour_mean'] = b.groupby('ID').hr.mean().reindex(ids)
    return f


# --- F5 vouchers and adjustments --------------------------------------------

def f5(f, d, dall, h, t, ids):
    hall = dall[dall._time < t]
    q1 = hall[hall._time >= t - pd.DateOffset(months=3)]
    for name, frame in (('all', hall), ('q1', q1)):
        v = frame[frame.is_voucher]
        f[f'vouch_n_{name}'] = v.groupby('ID').size().reindex(ids).fillna(0)
        f[f'vouch_amt_{name}'] = v.groupby('ID').amt.sum().reindex(ids).fillna(0)
        a = frame[frame.is_adj]
        f[f'adj_n_{name}'] = a.groupby('ID').size().reindex(ids).fillna(0)
        f[f'adj_amt_{name}'] = a.groupby('ID').amt.sum().reindex(ids).fillna(0)
    return f


# --- build ------------------------------------------------------------------

def build(d, dall, cutoff, ids, groups, site_table=None):
    """Extra columns for `groups` at `cutoff`, indexed by `ids`."""
    t = pd.Timestamp(cutoff)
    h = d[d._time < t]
    f = pd.DataFrame(index=ids)
    for g in GROUPS:
        if g in groups:
            f = globals()[g](f, d, dall, h, t, ids)
    if 'f2' in groups:
        f = f2_rates(f, home_site(h, ids), site_table)
    f.columns = F._clean(f.columns)
    return f.replace([np.inf, -np.inf], np.nan)
