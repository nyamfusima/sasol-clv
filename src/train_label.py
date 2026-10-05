"""Opportunity-label model v2: variants a-f, each scored on both validation folds.

The regressions are already level with 1st place; the whole gap is the label, so
this file only touches Opportunity. Every variant returns a (n x 17) probability
frame over cfg['opportunity_labels'] so variants can be blended and reweighted
the same way, and the predicted label is always the argmax.

    python src/train_label.py                 # run the full variant sweep
    python src/train_label.py --only base,growth
"""
import argparse, json
from pathlib import Path
import numpy as np, pandas as pd, lightgbm as lgb
import features as F
import snapshots as S
import validate as V

SEED = 42
TEST_CUTOFF = '2025-12-01'
ALL_BLOCKS = ('base', 'growth', 'recency', 'season')
PRED_DIR = Path('preds')


# --- feature-block column selection -----------------------------------------

def block_columns(cols):
    """Which block emitted each column. growth only emits g_*, season only sy_*;
    the rest of the non-base names come from the recency block."""
    base = [c for c in cols if not (c.startswith('g_') or c.startswith('sy_')
                                    or c.startswith(('d7_', 'd14_', 'd30_', 'd60_', 'since_',
                                                     'slope6_', 'last6_'))
                                    or c in ('gap_max', 'gap_last_vs_max', 'gap_last_vs_mean'))]
    return {'base': base,
            'growth': [c for c in cols if c.startswith('g_')],
            'season': [c for c in cols if c.startswith('sy_')],
            'recency': [c for c in cols if c not in base and not c.startswith(('g_', 'sy_'))]}


def select(X, blocks, bc):
    keep = [c for b in ALL_BLOCKS if b in blocks for c in bc[b]]
    return X[keep]


# --- models ------------------------------------------------------------------

def _clf(n=400, leaves=15, **kw):
    return lgb.LGBMClassifier(n_estimators=n, learning_rate=0.03, num_leaves=leaves,
                              min_child_samples=30, subsample=0.8, subsample_freq=1,
                              colsample_bytree=0.7, random_state=SEED, verbose=-1, **kw)


def _reg(n=300, leaves=31, **kw):
    return lgb.LGBMRegressor(n_estimators=n, learning_rate=0.05, num_leaves=leaves,
                             min_child_samples=30, subsample=0.8, subsample_freq=1,
                             colsample_bytree=0.7, random_state=SEED, verbose=-1, **kw)


def _wide(proba, classes, labels, index):
    """Spread a model's proba over all 17 labels (training folds can miss some)."""
    out = pd.DataFrame(0.0, index=index, columns=labels)
    out.loc[:, [labels[i] for i in classes]] = proba
    return out


def single(Xtr, ytr, Xva, labels, **kw):
    """The baseline shape: one 17-class classifier."""
    code = {l: i for i, l in enumerate(labels)}
    m = _clf(**kw).fit(Xtr, ytr.Opportunity.map(code))
    return _wide(m.predict_proba(Xva), m.classes_, labels, Xva.index)


def two_stage(Xtr, ytr, Xva, labels, **kw):
    """(d) Stage 1 active vs Inactivity, stage 2 the class among active customers.
    P(class) = P(active) * P(class | active); P(Inactivity) = P(inactive)."""
    code = {l: i for i, l in enumerate(labels)}
    inactive = labels.index('Inactivity')
    y = ytr.Opportunity.map(code)
    yv = y.to_numpy()
    s1 = _clf(**kw).fit(Xtr, (yv == inactive).astype(int))
    p_inact = s1.predict_proba(Xva)[:, list(s1.classes_).index(1)]
    act = yv != inactive
    s2 = _clf(**kw).fit(Xtr[act], yv[act])
    p_act = _wide(s2.predict_proba(Xva), s2.classes_, labels, Xva.index)
    out = p_act.mul(1 - p_inact, axis=0)
    out['Inactivity'] = p_inact
    return out


def rule_derived(Xtr, ytr, Xva, labels, cfg, tgt_tr, tgt_va, cats, temp=1.0):
    """(e) Predict next-quarter net spend per category, then run the published
    label rules over those predictions instead of classifying the label directly.

    Spend is modelled on a signed log scale (log1p of |spend| keeps the handful
    of return-dominated cells from dominating the fit), and 'at least one
    qualifying basket' is a separate per-category classifier."""
    clean = {c: F._clean([c])[0] for c in cats}
    prev = pd.DataFrame({c: Xva.get(f'q1sp_{clean[c]}', 0.0) for c in cats}, index=Xva.index)
    ever = pd.DataFrame({c: Xva.get(f'ever_{clean[c]}', 0.0) for c in cats}, index=Xva.index)
    fsp = pd.DataFrame(0.0, index=Xva.index, columns=cats)
    fn = pd.DataFrame(0.0, index=Xva.index, columns=cats)
    for c in cats:
        n = clean[c]
        y_sp, y_n = tgt_tr[f'fsp_{n}'], (tgt_tr[f'fn_{n}'] > 0).astype(int).to_numpy()
        s = np.sign(y_sp) * np.log1p(y_sp.abs())        # signed log keeps returns in scale
        h = _reg().fit(Xtr, s).predict(Xva)
        fsp[c] = np.sign(h) * np.expm1(np.abs(h))       # invert the transform
        if len(np.unique(y_n)) < 2:
            fn[c] = float(y_n[0])
        else:
            m = _clf(n=250).fit(Xtr, y_n)
            fn[c] = m.predict_proba(Xva)[:, list(m.classes_).index(1)]
    return rules_to_proba(prev, ever, fsp, fn, cats, cfg, labels, temp)


def rules_to_proba(prev, ever, fsp, fn, cats, cfg, labels, temp=1.0):
    """Apply rules 5-8 to predicted per-category spend. Returns a one-hot-ish
    frame so the result can be averaged with a classifier's probabilities."""
    mapping = cfg['category_mapping']
    buys = fn >= 0.5
    delta = fsp - prev
    grow = (ever > 0) & buys & (delta >= cfg['growth_min_rand'])
    grow &= fsp > prev.clip(lower=0) * (1 + cfg['growth_min_relative'])
    adopt = (ever == 0) & buys & (fsp > 0)
    av = fsp.where(adopt, 0).to_numpy()
    gv = delta.where(grow, 0).to_numpy()
    am, gm = av.max(axis=1), gv.max(axis=1)
    is_adopt = (am > 0) & (am >= gm)
    is_growth = (gm > 0) & ~is_adopt
    out = np.full(len(prev), 'Stable', dtype=object)
    for mask, values, kind in [(is_adopt, av, 'New category adoption'),
                               (is_growth, gv, 'Existing-category growth')]:
        cat = np.array(cats)[values.argmax(axis=1)]
        out[mask] = [kind + ': ' + mapping[c][kind] for c in cat[mask]]
    out[~buys.any(axis=1).to_numpy()] = 'Inactivity'
    p = pd.DataFrame(0.0, index=prev.index, columns=labels)
    for i, l in enumerate(out):
        p.iat[i, labels.index(l)] = 1.0
    return p


# --- (f) class-prior adjustment ----------------------------------------------

def fit_weights(proba, y_true, labels, rounds=4, grid=(0.5, 0.7, 0.85, 1.0, 1.2, 1.5, 2.0, 3.0)):
    """Coordinate ascent on per-class multipliers, maximising weighted F1."""
    w = np.ones(len(labels))
    best = V.f1(y_true, apply_weights(proba, w, labels))
    for _ in range(rounds):
        improved = False
        for j in range(len(labels)):
            cur = w[j]
            for g in grid:
                if g == cur:
                    continue
                w[j] = g
                s = V.f1(y_true, apply_weights(proba, w, labels))
                if s > best + 1e-6:
                    best, cur, improved = s, g, True
            w[j] = cur
        if not improved:
            break
    return w, best


def apply_weights(proba, w, labels):
    return np.array(labels)[(proba.to_numpy() * w).argmax(axis=1)]


def argmax_labels(proba, labels):
    return np.array(labels)[proba.to_numpy().argmax(axis=1)]


# --- sweep -------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--train', default='data/train.csv')
    ap.add_argument('--config', default='src/label_config.json')
    ap.add_argument('--only', default='', help='comma-separated variant names')
    ap.add_argument('--detail', action='store_true', help='per-class F1 + confusion matrix')
    a = ap.parse_args()
    cfg = json.load(open(a.config))
    labels = cfg['opportunity_labels']
    cats = S.categories(cfg)
    want = set(x.strip() for x in a.only.split(',') if x.strip())

    cutoffs = S.monthly(V.FIRST_CUTOFF, V.LAST_CUTOFF)
    snaps = S.load(cutoffs, a.train, cfg, ALL_BLOCKS)
    tgts = S.load_category_targets(cutoffs, a.train, cfg)
    bc = block_columns(list(snaps[cutoffs[0]][0].columns))
    print('feature-block sizes: ' + ', '.join(f'{k} {len(v)}' for k, v in bc.items()))

    data = {}
    for v in V.FOLDS:
        tr = S.trainable_for(cutoffs, v)
        data[v] = dict(
            tr=tr,
            Xtr=pd.concat([snaps[c][0] for c in tr]),
            ytr=pd.concat([snaps[c][1] for c in tr]),
            Ttr=pd.concat([tgts[c] for c in tr]),
            Xva=snaps[v][0], yva=snaps[v][1], Tva=tgts[v])

    rows, probas = [], {}

    def run(name, fn, blocks, note=''):
        if want and name not in want:
            return None
        res, pr = {}, {}
        for v in V.FOLDS:
            D = data[v]
            Xtr, Xva = select(D['Xtr'], blocks, bc), select(D['Xva'], blocks, bc)
            p = fn(Xtr, D['ytr'], Xva, labels, D)
            pr[v] = p
            res[v] = (D['yva'].Opportunity.to_numpy(), argmax_labels(p, labels))
        mean, sc = V.report(name, res, labels, show_detail=a.detail)
        rows.append([name, sc[V.FOLDS[0]], sc[V.FOLDS[1]], mean, note])
        probas[name] = (pr, res)
        return mean

    # -- a) growth-rule features, b) recency, c) same season last year ---------
    base_mean = run('base', lambda Xtr, ytr, Xva, L_, D: single(Xtr, ytr, Xva, L_), ('base',))
    run('a_growth', lambda Xtr, ytr, Xva, L_, D: single(Xtr, ytr, Xva, L_), ('base', 'growth'))
    run('b_recency', lambda Xtr, ytr, Xva, L_, D: single(Xtr, ytr, Xva, L_), ('base', 'growth', 'recency'))
    run('c_season', lambda Xtr, ytr, Xva, L_, D: single(Xtr, ytr, Xva, L_), ALL_BLOCKS)
    # -- d) two-stage ----------------------------------------------------------
    run('d_two_stage', lambda Xtr, ytr, Xva, L_, D: two_stage(Xtr, ytr, Xva, L_),
        ('base', 'growth', 'recency'))
    # -- e) rule-derived -------------------------------------------------------
    run('e_rule', lambda Xtr, ytr, Xva, L_, D: rule_derived(
        Xtr, ytr, Xva, L_, cfg, D['Ttr'], D['Tva'], cats), ('base', 'growth', 'recency'))

    print('\n' + '=' * 78)
    print(f'{"variant":<16}{"fold 2025-06":>14}{"fold 2025-09":>14}{"mean":>10}  note')
    for n, f1a, f1b, m, note in rows:
        print(f'{n:<16}{f1a:>14.4f}{f1b:>14.4f}{m:>10.4f}  {note}')
    print(f'\nbaseline mean for reference: {base_mean if base_mean else float("nan"):.4f}')


if __name__ == '__main__':
    main()
