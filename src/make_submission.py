"""One command from the raw CSVs to a submission. No cached artifacts.

    python src/make_submission.py                      # reproduces submission_v2.csv
    python src/make_submission.py --seeds 42-61 --out submissions/submission_v2_bag20.csv

Method, in order:
  1. Monthly snapshots at cutoffs 2024-06-01 .. 2025-09-01. A snapshot holds, for
     every customer with 3+ baskets before the cutoff, features built only from
     rows strictly before it, and the label-rule outcomes for the 3 months after.
  2. Features: the `base` block of features.py -- recency, tenure, basket gaps,
     per-quarter fuel/non-fuel spend and counts, per-category previous-quarter
     spend and ever-bought flags (97 columns).
  3. Opportunity: one 17-class LightGBM, predictions averaged over the seed set.
  4. CLV_fuel / CLV_nonfuel: a hurdle model -- P(y>0) from a classifier times
     E[y|y>0] from a regressor fitted on positive rows only -- also seed-averaged.
  5. Validation: two time-based folds (2025-06-01, 2025-09-01), each training
     only on snapshots whose 3-month outcome window ends on or before the cutoff,
     so no training outcome overlaps the outcome being scored.

Everything is seeded; the only source of variation is the seed set.
"""
import argparse, json, time
from pathlib import Path
import numpy as np, pandas as pd, lightgbm as lgb
from sklearn.metrics import f1_score
import snapshots as S   # load_data / build_one / categories; no cache is touched

SEEDS = (42, 43, 44, 45, 46)
TEST_CUTOFF = '2025-12-01'
FOLDS = ('2025-06-01', '2025-09-01')
FIRST_CUTOFF, LAST_CUTOFF = '2024-06-01', '2025-09-01'
TARGETS = ('CLV_fuel', 'CLV_nonfuel')
BLOCKS = ('base',)
CLF = dict(n_estimators=400, learning_rate=0.03, num_leaves=15, min_child_samples=30,
           subsample=0.8, subsample_freq=1, colsample_bytree=0.7)
REG = dict(n_estimators=500, learning_rate=0.03, num_leaves=31, min_child_samples=30,
           subsample=0.8, subsample_freq=1, colsample_bytree=0.7)
# leaderboard: Score = 0.4*F1 + 0.3*(1 - RMSE_fuel/0.74) + 0.3*(1 - RMSE_nonfuel/0.816)
FUEL_NORM, NONFUEL_NORM = 0.74, 0.816


def parse_seeds(text):
    if '-' in text:
        a, b = text.split('-')
        return tuple(range(int(a), int(b) + 1))
    return tuple(int(x) for x in text.split(','))


def monthly(first=FIRST_CUTOFF, last=LAST_CUTOFF):
    return [str(c.date()) for c in pd.date_range(first, last, freq='MS')]


def trainable_for(cutoffs, cutoff):
    """Snapshots whose 3-month outcome window ends on or before `cutoff`."""
    t = pd.Timestamp(cutoff)
    return [c for c in cutoffs if pd.Timestamp(c) + pd.DateOffset(months=3) <= t]


def combined(f1, rmse_fuel, rmse_nonfuel):
    return (0.4 * f1 + 0.3 * (1 - rmse_fuel / FUEL_NORM)
            + 0.3 * (1 - rmse_nonfuel / NONFUEL_NORM))


# --- models ------------------------------------------------------------------

def wide(proba, classes, labels, index):
    out = pd.DataFrame(0.0, index=index, columns=list(labels))
    out.loc[:, [labels[i] for i in classes]] = proba
    return out


def bag_clf(Xtr, ycode, Xva, labels, seeds, w=None):
    p = np.zeros((len(Xva), len(labels)))
    for s in seeds:
        m = lgb.LGBMClassifier(random_state=s, verbose=-1, **CLF).fit(Xtr, ycode, sample_weight=w)
        p += wide(m.predict_proba(Xva), list(m.classes_), labels, Xva.index).to_numpy()
    return pd.DataFrame(p / len(seeds), index=Xva.index, columns=list(labels))


def bag_reg(Xtr, y, Xva, seeds, w=None):
    out = np.zeros(len(Xva))
    for s in seeds:
        out += lgb.LGBMRegressor(random_state=s, verbose=-1, **REG).fit(
            Xtr, y, sample_weight=w).predict(Xva)
    return np.clip(out / len(seeds), 0, None)


def bag_binary(Xtr, yb, Xva, seeds, w=None):
    out = np.zeros(len(Xva))
    for s in seeds:
        m = lgb.LGBMClassifier(random_state=s, verbose=-1, **CLF).fit(Xtr, yb, sample_weight=w)
        out += m.predict_proba(Xva)[:, list(m.classes_).index(1)]
    return out / len(seeds)


def hurdle(Xtr, y, Xva, seeds, w=None):
    """P(y>0) * E[y|y>0]; the second stage sees positive rows only."""
    pos = (y.to_numpy() > 0)
    if pos.all() or not pos.any():
        return bag_reg(Xtr, y, Xva, seeds, w)
    p = bag_binary(Xtr, pos.astype(int), Xva, seeds, w)
    mag = bag_reg(Xtr[pos], y[pos], Xva, seeds, None if w is None else w[pos])
    return np.clip(p * mag, 0, None)


# --- pipeline ----------------------------------------------------------------

def build_snapshots(d, cfg, cats, cutoffs, verbose=True):
    snaps = {}
    for c in cutoffs:
        if verbose:
            print(f'  snapshot {c}', flush=True)
        snaps[c] = S.build_one(d, c, cfg, cats, blocks=BLOCKS,
                               with_labels=c != TEST_CUTOFF)
    return snaps


def stack(snaps, cutoffs):
    X = pd.concat([snaps[c][0] for c in cutoffs])
    y = pd.concat([snaps[c][1] for c in cutoffs])
    assert len(X) == len(y)
    return X, y


def predict(Xtr, ytr, Xte, labels, code, seeds):
    opp = np.array(labels)[bag_clf(Xtr, ytr.Opportunity.map(code).to_numpy(),
                                   Xte, labels, seeds).to_numpy().argmax(axis=1)]
    reg = {t: hurdle(Xtr, ytr[t], Xte, seeds) for t in TARGETS}
    return opp, reg


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--train', default='data/train.csv')
    ap.add_argument('--test', default='data/test.csv')
    ap.add_argument('--config', default='src/label_config.json')
    ap.add_argument('--out', default='submissions/submission_v2.csv')
    ap.add_argument('--seeds', default='42-46', help='"42-46" or "42,43,44"')
    ap.add_argument('--skip-validation', action='store_true')
    a = ap.parse_args()
    t0 = time.time()
    seeds = parse_seeds(a.seeds)
    cfg = json.loads(Path(a.config).read_text())
    labels = cfg['opportunity_labels']
    code = {l: i for i, l in enumerate(labels)}
    cats = S.categories(cfg)
    print(f'seeds {seeds[0]}..{seeds[-1]} ({len(seeds)}) | out {a.out}')

    d = S.load_data(a.train, cfg)
    cutoffs = monthly()
    print(f'building {len(cutoffs) + 1} snapshots from {a.train}')
    snaps = build_snapshots(d, cfg, cats, cutoffs + [TEST_CUTOFF])

    if not a.skip_validation:
        scores = []
        for v in FOLDS:
            tr = trainable_for(cutoffs, v)
            Xtr, ytr = stack(snaps, tr)
            Xva, yva = snaps[v]
            opp, reg = predict(Xtr, ytr, Xva, labels, code, seeds)
            f1 = f1_score(yva.Opportunity, opp, average='weighted', zero_division=0)
            rf = float(np.sqrt(((reg['CLV_fuel'] - yva.CLV_fuel) ** 2).mean()))
            rn = float(np.sqrt(((reg['CLV_nonfuel'] - yva.CLV_nonfuel) ** 2).mean()))
            sc = combined(f1, rf, rn)
            scores.append((f1, rf, rn, sc))
            print(f'  fold {v}: {len(tr)} snapshots, {len(Xtr)} rows -> '
                  f'F1 {f1:.4f} | rmse_fuel {rf:.4f} | rmse_nonfuel {rn:.4f} | score {sc:.5f}')
        m = np.mean(scores, axis=0)
        print(f'VALIDATION mean: F1 {m[0]:.4f} | rmse_fuel {m[1]:.4f} | '
              f'rmse_nonfuel {m[2]:.4f} | score {m[3]:.5f}')

    Xtr, ytr = stack(snaps, cutoffs)
    Xte = snaps[TEST_CUTOFF][0]
    print(f'final fit: {len(cutoffs)} snapshots, {len(Xtr)} rows')
    opp, reg = predict(Xtr, ytr, Xte, labels, code, seeds)
    test_ids = pd.read_csv(a.test, dtype=str).ID
    sub = pd.DataFrame({'CLV_fuel': reg['CLV_fuel'], 'CLV_nonfuel': reg['CLV_nonfuel'],
                        'Opportunity': opp}, index=Xte.index).reindex(test_ids)
    # validate before writing, never after
    assert len(sub) == 5488, f'expected 5488 rows, got {len(sub)}'
    assert sub.index.equals(pd.Index(test_ids)), 'IDs do not match the test file'
    assert sub.notna().all().all(), 'missing predictions'
    bad = set(sub.Opportunity) - set(labels)
    assert not bad, f'labels outside the config: {bad}'
    assert (sub[['CLV_fuel', 'CLV_nonfuel']] >= 0).all().all(), 'negative CLV'
    assert list(sub.columns) == ['CLV_fuel', 'CLV_nonfuel', 'Opportunity'], sub.columns
    Path(a.out).parent.mkdir(exist_ok=True)
    sub.rename_axis('ID').reset_index().to_csv(a.out, index=False, float_format='%.17g')
    print(f'wrote {a.out}: {len(sub)} rows, IDs match, all labels in config, '
          f'CLV min {sub[list(TARGETS)].min().min():.4f}')
    print(f'total runtime {time.time() - t0:.0f}s')


if __name__ == '__main__':
    main()
