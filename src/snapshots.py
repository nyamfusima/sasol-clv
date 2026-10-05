"""Snapshot building, cached so experiments do not rebuild features every run.

A snapshot is one cutoff: customer features for everyone eligible at that cutoff
plus the label-rule outcomes for the three months after it. Snapshots for every
cutoff live in one parquet keyed by `_cutoff`; each feature-block combination
gets its own file so a stale cache can never feed the wrong columns to a model.
"""
import hashlib, json
from pathlib import Path
import numpy as np, pandas as pd
import features as F
import label_rules as L

EXCLUDED = ['Value added services', 'Print media', 'Carrier Bags']
CACHE_DIR = Path('preds')
Y_COLS = ['CLV_fuel', 'CLV_nonfuel', 'Opportunity', 'fuel_litres', 'nonfuel_rand',
          'winning_source_category', 'winning_increment_rand']


def load_data(path, cfg):
    """Transactions with the adjustment items dropped and the derived columns
    baseline.py used (identical preprocessing)."""
    d = L.load_transactions(path)
    d = d[~d.ItemName.isin(cfg['adjustment_items'])].copy()
    d['qty'] = d._quantity / L.SCALE
    d['amt'] = d._amount / L.SCALE
    d['is_fuel'] = d.ItemCategoryLevel2.eq('Fuel') & d.LineItemType.eq('Sale_Fuel')
    d['is_nonfuel'] = ~d.ItemCategoryLevel2.isin(['Fuel'] + EXCLUDED)
    return d


def categories(cfg):
    return sorted(c for c, v in cfg['category_mapping'].items() if v['included'])


def build_one(d, cutoff, cfg, cats, blocks=('base',), with_labels=True):
    X = F.build(d, cutoff, cats, blocks=blocks, cfg=cfg)
    if not with_labels:
        return X, None
    raw = L.raw_labels(d, cutoff, cfg)
    y = L.transform(raw, cfg).set_index('ID')
    y = y.join(raw.set_index('ID')[['fuel_litres', 'nonfuel_rand',
                                    'winning_source_category', 'winning_increment_rand']])
    return X, y.reindex(X.index)


def _tag(blocks):
    return '_'.join(b for b in F.BLOCKS if b in blocks) or 'none'


def _path(blocks):
    return CACHE_DIR / f'snapshots_{_tag(blocks)}.parquet'


def load(cutoffs, train_path='data/train.csv', cfg=None, blocks=('base',),
         label_cutoffs=None, refresh=False, verbose=True, d=None):
    """Return {cutoff: (X, y)} for every cutoff, building and caching as needed.

    `label_cutoffs` lists the cutoffs that get labels (default: all of them; the
    test cutoff has no observable outcome and must be left out).
    """
    cfg = cfg or json.loads(Path('src/label_config.json').read_text())
    cats = categories(cfg)
    cutoffs = [str(pd.Timestamp(c).date()) for c in cutoffs]
    labelled = set(cutoffs if label_cutoffs is None else
                   [str(pd.Timestamp(c).date()) for c in label_cutoffs])
    p = _path(blocks)
    cache = {}
    if p.exists() and not refresh:
        t = pd.read_parquet(p)
        for c, g in t.groupby('_cutoff', observed=True):
            cache[str(c)] = g.drop(columns=['_cutoff'])
    missing = [c for c in cutoffs if c not in cache]
    if missing:
        if d is None:
            d = load_data(train_path, cfg)
        for c in missing:
            if verbose:
                print(f'  building snapshot {c} ...', flush=True)
            X, y = build_one(d, c, cfg, cats, blocks=blocks, with_labels=c in labelled)
            cache[c] = X if y is None else X.join(y)
        CACHE_DIR.mkdir(exist_ok=True)
        t = pd.concat([cache[c].assign(_cutoff=c) for c in sorted(cache)])
        t.rename_axis('ID').reset_index().to_parquet(p, index=False)
        if verbose:
            print(f'  cached {len(cache)} snapshots -> {p}', flush=True)
    out = {}
    for c in cutoffs:
        g = cache[c]
        if 'ID' in g.columns:
            g = g.set_index('ID')
        ycols = [k for k in Y_COLS if k in g.columns]
        X = g.drop(columns=ycols)
        y = g[ycols] if (ycols and c in labelled) else None
        out[c] = (X, y)
    return out


def monthly(first='2024-06-01', last='2025-09-01'):
    """One cutoff per month, each with a full 3-month outcome window in the data."""
    return [str(c.date()) for c in pd.date_range(first, last, freq='MS')]


def trainable_for(cutoffs, val_cutoff):
    """Snapshots whose outcome window ends on or before val_cutoff, so their
    outcomes cannot overlap the validation outcome window."""
    v = pd.Timestamp(val_cutoff)
    return [c for c in cutoffs if pd.Timestamp(c) + pd.DateOffset(months=3) <= v]


# --- per-category outcome targets, for the rule-derived label variant ---------

CT_CACHE = CACHE_DIR / 'cat_targets.parquet'


def category_targets(d, cutoff, cfg, cats, ids=None):
    """Future (next 3 months) net spend and qualifying-basket count per source
    category -- the quantities the label rules actually compare. Mirrors the
    `matrix()` helper inside label_rules.raw_labels."""
    t = pd.Timestamp(cutoff)
    fut = d[(d._time >= t) & (d._time < t + pd.DateOffset(months=3))]
    fut = fut[fut.ItemCategoryLevel2.isin(cats)]
    if ids is None:
        ids = L.eligible_ids(d, t, cfg['minimum_historical_transactions'])
    sp = (fut.groupby(['ID', 'ItemCategoryLevel2']).amt.sum()
          .unstack(fill_value=0).reindex(index=ids, columns=cats, fill_value=0))
    q = fut[(fut.qty > 0) & (fut.amt > 0)]
    n = (q.groupby(['ID', 'ItemCategoryLevel2']).TransactionId.nunique()
         .unstack(fill_value=0).reindex(index=ids, columns=cats, fill_value=0))
    out = sp.add_prefix('fsp_').join(n.add_prefix('fn_'))
    out.columns = F._clean(out.columns)
    return out


def load_category_targets(cutoffs, train_path='data/train.csv', cfg=None,
                          refresh=False, verbose=True, d=None):
    cfg = cfg or json.loads(Path('src/label_config.json').read_text())
    cats = categories(cfg)
    cutoffs = [str(pd.Timestamp(c).date()) for c in cutoffs]
    cache = {}
    if CT_CACHE.exists() and not refresh:
        t = pd.read_parquet(CT_CACHE)
        for c, g in t.groupby('_cutoff', observed=True):
            cache[str(c)] = g.drop(columns=['_cutoff']).set_index('ID')
    missing = [c for c in cutoffs if c not in cache]
    if missing:
        if d is None:
            d = load_data(train_path, cfg)
        for c in missing:
            if verbose:
                print(f'  building category targets {c} ...', flush=True)
            cache[c] = category_targets(d, c, cfg, cats)
        CACHE_DIR.mkdir(exist_ok=True)
        pd.concat([cache[c].assign(_cutoff=c) for c in sorted(cache)]) \
          .rename_axis('ID').reset_index().to_parquet(CT_CACHE, index=False)
        if verbose:
            print(f'  cached {len(cache)} category-target snapshots -> {CT_CACHE}', flush=True)
    return {c: cache[c] for c in cutoffs}
