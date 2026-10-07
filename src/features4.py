"""Sweep 6 feature blocks: new information only.

Every block here is meant to carry information the model does not already have.
That is the one thing that has ever moved the classifier: the lag series (L1)
added monthly resolution absent from the quarterly aggregates and gained
+0.00139, while restructuring the label space, swapping model families and
re-deriving computable features failed fourteen times between them.

  N1  own label history at 3/6/9 months back, plus per-category growth flags
  N2  per-category monthly lags, named-label categories only
  N3  per-category recency and age
  N4  basket structure
  N5  weekly non-fuel rands and basket counts
  N6  monthly lags extended from 12 to 18 months

Everything reads only rows strictly before the cutoff. `precompute` builds the
monthly and weekly aggregates once; the per-cutoff builders slice them.
"""
import numpy as np, pandas as pd
import features as F

NAMED = ['Beverages', 'Food & beverage', 'Fuel', 'Braai & Ice', 'Breads & rolls',
         'Confectionary', 'Groceries', 'Ice-cream', 'Lubricants', 'Snacking', 'Tobacco']
GROWTH_CATS = ['Fuel', 'Beverages', 'Food & beverage']
BIG4 = ['Stable', 'Inactivity', 'Existing-category growth: Fuel',
        'Existing-category growth: Other']
N2_LAGS = 6
N5_LAGS = 13
N6_LAGS = 18
RECENCY_WINDOWS = (30, 90, 180)


def precompute(d):
    """Monthly and weekly aggregates, built once over the whole history."""
    month = d._time.dt.to_period('M').dt.to_timestamp()
    week = d._time.dt.to_period('W').dt.start_time
    qual = (d.qty > 0) & (d.amt > 0)
    out = {}
    # per-category monthly: net spend and qualifying basket count
    sub = d[d.ItemCategoryLevel2.isin(NAMED)]
    sm = month[sub.index]
    out['mcat_spend'] = sub.groupby([sub.ID, sub.ItemCategoryLevel2, sm]).amt.sum()
    q = sub[qual.reindex(sub.index, fill_value=False)]
    out['mcat_n'] = q.groupby([q.ID, q.ItemCategoryLevel2,
                               month[q.index]]).TransactionId.nunique()
    # whole-customer monthly series (N6 extends the 12 months features3 builds)
    fu, nf = d[d.is_fuel], d[d.is_nonfuel]
    out['m_fuel_l'] = fu.groupby([fu.ID, month[fu.index]]).qty.sum()
    out['m_fuel_r'] = fu.groupby([fu.ID, month[fu.index]]).amt.sum()
    out['m_nf_r'] = nf.groupby([nf.ID, month[nf.index]]).amt.sum()
    out['m_baskets'] = d.groupby([d.ID, month]).TransactionId.nunique()
    # weekly non-fuel rands and basket counts
    out['w_nf_r'] = nf.groupby([nf.ID, week[nf.index]]).amt.sum()
    out['w_baskets'] = d.groupby([d.ID, week]).TransactionId.nunique()
    # qualifying purchases, for N3
    out['qual'] = d[qual][['ID', 'ItemCategoryLevel2', '_time']].copy()
    out['first_seen'] = d.groupby('ID')._time.min()
    return out


def _slice(series, ids, key, level_vals):
    """Pull one (ID, period) slice out of a precomputed MultiIndex series."""
    try:
        v = series.xs(key, level=level_vals)
    except KeyError:
        return pd.Series(0.0, index=ids)
    return v.reindex(ids).fillna(0.0)


# --- N1 own label history -----------------------------------------------------

def growth_flags(X, tgt, cfg, cats):
    """Which of the growth categories qualified as growth in this snapshot's
    outcome window. Same rule as label_rules, verified in sweep 1."""
    clean = {c: F._clean([c])[0] for c in cats}
    out = {}
    for c in GROWTH_CATS:
        n = clean[c]
        prev = X[f'q1sp_{n}'].to_numpy()
        ever = X[f'ever_{n}'].to_numpy()
        fsp = tgt[f'fsp_{n}'].to_numpy()
        fn = tgt[f'fn_{n}'].to_numpy()
        ok = ((ever > 0) & (fn > 0) & ((fsp - prev) >= cfg['growth_min_rand'])
              & (fsp > np.clip(prev, 0, None) * (1 + cfg['growth_min_relative'])))
        out[c] = ok.astype(float)
    return pd.DataFrame(out, index=X.index)


def n1(cutoff, ids, labels_by_cut, feats_by_cut, tgts_by_cut, cfg, labels):
    """The customer's own label 3, 6 and 9 months back. Those windows end at the
    current cutoff or earlier, so they are fully observed. NaN where the customer
    was not yet eligible at that earlier cutoff."""
    t = pd.Timestamp(cutoff)
    cats = sorted(c for c, v in cfg['category_mapping'].items() if v['included'])
    code = {l: i for i, l in enumerate(labels)}
    ccode = {c: i for i, c in enumerate(cats)}
    out = pd.DataFrame(index=ids)
    for k in (3, 6, 9):
        c = str((t - pd.DateOffset(months=k)).date())
        pre = f'n1_l{k}_'
        y = labels_by_cut.get(c)
        if y is None:
            for nm in (['opp', 'wincat', 'winincr'] + [f'is_{x}' for x in range(4)]
                       + [f'grew_{F._clean([g])[0]}' for g in GROWTH_CATS]):
                out[pre + nm] = np.nan
            continue
        yy = y.reindex(ids)
        known = yy.Opportunity.notna()
        out[pre + 'opp'] = yy.Opportunity.map(code)
        out[pre + 'wincat'] = yy.winning_source_category.replace('', np.nan).map(ccode)
        out[pre + 'winincr'] = yy.winning_increment_rand
        for i, big in enumerate(BIG4):
            out[pre + f'is_{i}'] = yy.Opportunity.eq(big).astype(float).where(known)
        g = growth_flags(feats_by_cut[c], tgts_by_cut[c], cfg, cats).reindex(ids)
        for gc in GROWTH_CATS:
            out[pre + f'grew_{F._clean([gc])[0]}'] = g[gc]
    out.columns = F._clean(out.columns)
    return out


# --- N2 per-category monthly lags --------------------------------------------

def n2(cutoff, ids, pre):
    t = pd.Timestamp(cutoff)
    first = pre['first_seen'].reindex(ids)
    cols = {}
    for k in range(1, N2_LAGS + 1):
        m = t - pd.DateOffset(months=k)
        before_first = first >= (t - pd.DateOffset(months=k - 1))
        for cat in NAMED:
            n = F._clean([cat])[0]
            for tag, src in (('sp', 'mcat_spend'), ('n', 'mcat_n')):
                try:
                    v = pre[src].xs((cat, m), level=(1, 2)).reindex(ids).fillna(0.0)
                except KeyError:
                    v = pd.Series(0.0, index=ids)
                cols[f'n2_{n}_{tag}_l{k}'] = v.where(~before_first, np.nan)
    out = pd.DataFrame(cols, index=ids)
    out.columns = F._clean(out.columns)
    return out


# --- N3 per-category recency and age -----------------------------------------

def n3(cutoff, ids, pre, cfg):
    t = pd.Timestamp(cutoff)
    cats = sorted(c for c, v in cfg['category_mapping'].items() if v['included'])
    q = pre['qual']
    q = q[q._time < t]
    g = q.groupby(['ID', 'ItemCategoryLevel2'])._time
    last = g.max().unstack()
    first = g.min().unstack()
    cols = {}
    for c in cats:
        n = F._clean([c])[0]
        lc = last[c].reindex(ids) if c in last.columns else pd.Series(pd.NaT, index=ids)
        fc = first[c].reindex(ids) if c in first.columns else pd.Series(pd.NaT, index=ids)
        cols[f'n3_since_last_{n}'] = (t - lc).dt.days
        cols[f'n3_since_first_{n}'] = (t - fc).dt.days
    for w in RECENCY_WINDOWS:
        recent = q[q._time >= t - pd.Timedelta(days=w)]
        cols[f'n3_ncats_{w}d'] = (recent.groupby('ID').ItemCategoryLevel2.nunique()
                                 .reindex(ids).fillna(0.0))
    out = pd.DataFrame(cols, index=ids)
    out.columns = F._clean(out.columns)
    return out


# --- N4 basket structure ------------------------------------------------------

def n4(d, cutoff, ids):
    t = pd.Timestamp(cutoff)
    h = d[d._time < t]
    cols = {}
    for tag, hh in (('all', h), ('q1', h[h._time >= t - pd.DateOffset(months=3)])):
        fu = hh[hh.is_fuel]
        per_fill = fu.groupby(['ID', 'TransactionId']).qty.sum()
        gp = per_fill.groupby(level=0)
        cols[f'n4_{tag}_litres_med'] = gp.median().reindex(ids)
        cols[f'n4_{tag}_litres_max'] = gp.max().reindex(ids)
        cols[f'n4_{tag}_litres_sd'] = gp.std().reindex(ids)
        nfills = gp.size().reindex(ids)
        months = 3.0 if tag == 'q1' else np.maximum(
            (t - hh.groupby('ID')._time.min()).dt.days.reindex(ids) / 30.4, 1.0)
        cols[f'n4_{tag}_fills_per_month'] = nfills / months
        # baskets containing both fuel and shop lines
        bask = hh.groupby('TransactionId').agg(has_f=('is_fuel', 'any'),
                                               has_s=('is_nonfuel', 'any'))
        both = set(bask.index[bask.has_f & bask.has_s])
        b = hh.drop_duplicates(['ID', 'TransactionId'])
        cols[f'n4_{tag}_mixed_share'] = (b.TransactionId.isin(both).groupby(b.ID).mean()
                                         .reindex(ids))
        nf = hh[hh.is_nonfuel]
        cols[f'n4_{tag}_nf_basket_med'] = (nf.groupby(['ID', 'TransactionId']).amt.sum()
                                           .groupby(level=0).median().reindex(ids))
        sp = nf.groupby(['ID', 'ItemCategoryLevel2']).amt.sum()
        tot = sp.groupby(level=0).sum()
        cols[f'n4_{tag}_top_cat_share'] = (sp.groupby(level=0).max() /
                                           tot.replace(0, np.nan)).reindex(ids)
        cols[f'n4_{tag}_n_products'] = hh.groupby('ID').StockItemId.nunique().reindex(ids)
    out = pd.DataFrame(cols, index=ids)
    out.columns = F._clean(out.columns)
    return out


# --- N5 weekly non-fuel and baskets ------------------------------------------

def n5(cutoff, ids, pre):
    t = pd.Timestamp(cutoff)
    first = pre['first_seen'].reindex(ids)
    cols = {}
    for k in range(1, N5_LAGS + 1):
        a = (t - pd.Timedelta(days=7 * k)).to_period('W').start_time
        before_first = first >= (t - pd.Timedelta(days=7 * (k - 1)))
        for tag, src in (('nf_r', 'w_nf_r'), ('baskets', 'w_baskets')):
            try:
                v = pre[src].xs(a, level=1).reindex(ids).fillna(0.0)
            except KeyError:
                v = pd.Series(0.0, index=ids)
            cols[f'n5_{tag}_w{k}'] = v.where(~before_first, np.nan)
    out = pd.DataFrame(cols, index=ids)
    out.columns = F._clean(out.columns)
    return out


# --- N6 longer monthly history (lags 13-18) ----------------------------------

def n6(cutoff, ids, pre):
    t = pd.Timestamp(cutoff)
    first = pre['first_seen'].reindex(ids)
    cols = {}
    for k in range(13, N6_LAGS + 1):
        m = t - pd.DateOffset(months=k)
        before_first = first >= (t - pd.DateOffset(months=k - 1))
        for tag, src in (('fuel_l', 'm_fuel_l'), ('fuel_r', 'm_fuel_r'),
                         ('nf_r', 'm_nf_r'), ('baskets', 'm_baskets')):
            try:
                v = pre[src].xs(m, level=1).reindex(ids).fillna(0.0)
            except KeyError:
                v = pd.Series(0.0, index=ids)
            cols[f'n6_{tag}_l{k}'] = v.where(~before_first, np.nan)
    out = pd.DataFrame(cols, index=ids)
    out.columns = F._clean(out.columns)
    return out


BLOCKS = ('n1', 'n2', 'n3', 'n4', 'n5', 'n6')
