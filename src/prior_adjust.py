"""Conservative second pick: bag20 with a 3-parameter class-prior adjustment.

Sweep 1 fitted all 17 class weights and the result barely generalised -- in-fold
+0.0093 against held-out +0.0006/+0.0020, i.e. the weights largely memorised
their own fold. Here the search is restricted to the three classes with enough
support to estimate a weight from (Stable, Existing-category growth: Fuel,
Existing-category growth: Other), which between them carry about 80% of the
weighted-F1 mass.

Every number reported is cross-fold: a weight vector fitted on one fold is only
ever scored on the other. The applied vector is the mean of the two, which is
contaminated by construction, so its in-fold score is NOT used for the decision.

    python src/prior_adjust.py
"""
import argparse, json
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.metrics import f1_score
import make_submission as M
import snapshots as S
import validate as V

FREE = ['Stable', 'Existing-category growth: Fuel', 'Existing-category growth: Other']
GRID = (0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.25, 1.4, 1.6, 2.0)


def f1w(y, pred):
    return f1_score(y, pred, average='weighted', zero_division=0)


def apply_w(proba, w, labels):
    return np.array(labels)[(proba * w).argmax(axis=1)]


def fit_weights(proba, y, labels, free_idx, rounds=6):
    """Coordinate ascent over the free classes only; everything else stays 1.0."""
    w = np.ones(len(labels))
    best = f1w(y, apply_w(proba, w, labels))
    for _ in range(rounds):
        improved = False
        for j in free_idx:
            cur = w[j]
            for g in GRID:
                w[j] = g
                s = f1w(y, apply_w(proba, w, labels))
                if s > best + 1e-9:
                    best, cur, improved = s, g, True
            w[j] = cur
        if not improved:
            break
    return w, best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--train', default='data/train.csv')
    ap.add_argument('--config', default='src/label_config.json')
    ap.add_argument('--bag20', default='submissions/submission_v2_bag20.csv')
    ap.add_argument('--out', default='submissions/submission_v2_prior.csv')
    a = ap.parse_args()
    cfg = json.loads(Path(a.config).read_text())
    labels = cfg['opportunity_labels']
    free_idx = [labels.index(c) for c in FREE]
    cutoffs = M.monthly()
    snaps = S.load(cutoffs, a.train, cfg, ('base',), verbose=False)

    proba, truth = {}, {}
    for v in V.FOLDS:
        p = pd.read_parquet(f'preds/oof_bag20_{v}.parquet').set_index('ID')
        y = snaps[v][1]
        p = p.reindex(y.index)
        proba[v] = p[labels].to_numpy()
        truth[v] = y.Opportunity.to_numpy()

    base = {v: f1w(truth[v], np.array(labels)[proba[v].argmax(axis=1)]) for v in V.FOLDS}
    print(f'bag20 baseline F1: ' + ' | '.join(f'{v} {base[v]:.4f}' for v in V.FOLDS)
          + f' | mean {np.mean(list(base.values())):.4f}')
    print(f'free parameters ({len(FREE)}): {FREE}')
    support = pd.Series(np.concatenate([truth[v] for v in V.FOLDS])).value_counts()
    print('  support across both folds: '
          + ', '.join(f'{c} {support.get(c, 0)}' for c in FREE))

    print('\n--- cross-fold: fit on one fold, score ONLY on the other ---')
    weights, held = {}, {}
    for fit_on in V.FOLDS:
        other = [v for v in V.FOLDS if v != fit_on][0]
        w, in_f1 = fit_weights(proba[fit_on], truth[fit_on], labels, free_idx)
        out_f1 = f1w(truth[other], apply_w(proba[other], w, labels))
        weights[fit_on], held[other] = w, out_f1
        shown = {c: round(w[labels.index(c)], 3) for c in FREE}
        print(f'  fit {fit_on} {shown}')
        print(f'    in-fold {base[fit_on]:.4f} -> {in_f1:.4f} (ignored) | '
              f'held-out {other}: {base[other]:.4f} -> {out_f1:.4f} '
              f'({out_f1 - base[other]:+.4f})')

    worse = [v for v in V.FOLDS if held[v] < base[v] - 1e-12]
    mean_held = float(np.mean([held[v] for v in V.FOLDS]))
    print(f'\n  cross-fold mean F1 {mean_held:.4f} vs bag20 '
          f'{np.mean(list(base.values())):.4f} ({mean_held - np.mean(list(base.values())):+.4f})')
    if worse:
        print(f'  F1 gets WORSE on {worse} -> not writing {a.out}')
        return
    print('  no fold gets worse -> writing the file')

    wmean = np.mean([weights[v] for v in V.FOLDS], axis=0)
    print('  applied weights (mean of the two fold fits): '
          + ', '.join(f'{c} {wmean[labels.index(c)]:.3f}' for c in FREE))
    pte = pd.read_parquet('preds/oof_bag20_test.parquet').set_index('ID')
    bag = pd.read_csv(a.bag20, dtype={'ID': str}).set_index('ID')
    test_ids = pd.read_csv('data/test.csv', dtype=str).ID
    pte = pte.reindex(test_ids)[labels].to_numpy()
    sub = bag[['CLV_fuel', 'CLV_nonfuel']].reindex(test_ids).copy()
    sub['Opportunity'] = apply_w(pte, wmean, labels)
    assert len(sub) == 5488, len(sub)
    assert sub.index.equals(pd.Index(test_ids)), 'IDs do not match data/test.csv'
    assert sub.notna().all().all(), 'missing predictions'
    assert not set(sub.Opportunity) - set(labels), 'label outside the config'
    assert (sub[['CLV_fuel', 'CLV_nonfuel']] >= 0).all().all(), 'negative CLV'
    assert sub[['CLV_fuel', 'CLV_nonfuel']].equals(
        bag[['CLV_fuel', 'CLV_nonfuel']].reindex(test_ids)), 'CLV drifted from bag20'
    assert list(sub.columns) == ['CLV_fuel', 'CLV_nonfuel', 'Opportunity'], sub.columns
    Path(a.out).parent.mkdir(exist_ok=True)
    sub.rename_axis('ID').reset_index().to_csv(a.out, index=False)
    agree = (bag.Opportunity.reindex(test_ids).to_numpy() == sub.Opportunity.to_numpy()).mean()
    print(f'  wrote {a.out}: 5488 rows, IDs match, labels in config, CLV >= 0 '
          f'and identical to bag20; Opportunity agrees with bag20 on {agree:.2%}')
    Path('preds/prior_weights.json').write_text(json.dumps(dict(
        free=FREE, weights={c: float(wmean[labels.index(c)]) for c in FREE},
        cross_fold={v: dict(base=base[v], held_out=held[v]) for v in V.FOLDS},
        cross_fold_mean=mean_held), indent=2))


if __name__ == '__main__':
    main()
