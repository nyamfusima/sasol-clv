"""One command from the raw CSVs to a submission. No cached artifacts.

    python src/make_submission.py --recipe v4_hybrid      # selected pick, alpha 0.5
    python src/make_submission.py --recipe v4_alpha075   # selected pick, alpha 0.75
    python src/make_submission.py --recipe v4_lags
    python src/make_submission.py --recipe v4_simple
    python src/make_submission.py --recipe v3
    python src/make_submission.py --recipe v2
    python src/make_submission.py --recipe v5_reg       # pre-registered diagnostic

Each recipe builds every snapshot in memory, prints the per-fold and mean
validation score, validates the output, and reports its runtime. Needs only
pandas, numpy, scikit-learn and lightgbm.

Method, in order:
  1. Monthly snapshots at cutoffs 2024-06-01 .. 2025-09-01. A snapshot holds,
     for every customer with 3+ baskets before the cutoff, features built only
     from rows strictly before it, and the label-rule outcomes for the 3 months
     after.
  2. Features: the `base` block of features.py (97 columns) and, for the v4
     recipes, the lag series of features3.py (62 more) -- twelve monthly lags of
     fuel litres, fuel rands, non-fuel rands and basket counts plus thirteen
     weekly fuel totals. Lags reaching before a customer's first transaction are
     NaN, not 0. For `v5_reg` the hurdle regressions instead add the r1
     renewal block of features5.py (34 columns: fill-gap statistics, phase, and
     projected fill counts and spend for the outcome window).
  3. Opportunity: one 17-class LightGBM, probabilities averaged over the seed
     set, then a decision rule (see `apply_rule`).
  4. CLV_fuel / CLV_nonfuel: a hurdle model, P(y>0) from a classifier times
     E[y|y>0] from a regressor fitted on positive rows only, each stage
     seed-averaged and the product clipped at 0.
  5. Final models are retrained on all 16 snapshots and predict the test cutoff
     2025-12-01, with features from all of train.csv (which ends 2025-11-30).
  6. Validation: two time-based folds (2025-06-01, 2025-09-01), each training
     only on snapshots whose 3-month outcome window ends on or before the
     cutoff, so no training outcome overlaps the outcome being scored.
"""
import argparse, json, time
from pathlib import Path
import numpy as np, pandas as pd, lightgbm as lgb
from sklearn.metrics import f1_score
import snapshots as S   # load_data / build_one / categories; no cache is touched

TEST_CUTOFF = '2025-12-01'
FOLDS = ('2025-06-01', '2025-09-01')
FIRST_CUTOFF, LAST_CUTOFF = '2024-06-01', '2025-09-01'
TARGETS = ('CLV_fuel', 'CLV_nonfuel')
CLF = dict(n_estimators=400, learning_rate=0.03, num_leaves=15, min_child_samples=30,
           subsample=0.8, subsample_freq=1, colsample_bytree=0.7)
REG = dict(n_estimators=500, learning_rate=0.03, num_leaves=31, min_child_samples=30,
           subsample=0.8, subsample_freq=1, colsample_bytree=0.7)
# leaderboard: Score = 0.4*F1 + 0.3*(1 - RMSE_fuel/0.74) + 0.3*(1 - RMSE_nonfuel/0.816)
# The 0.3/0.3/0.4 weights are official; the normalisers are inferred (see README).
FUEL_NORM, NONFUEL_NORM = 0.74, 0.816
FUEL_LABEL = 'Existing-category growth: Fuel'
BIG4 = ['Stable', 'Inactivity', FUEL_LABEL, 'Existing-category growth: Other']
FUEL_GRID = tuple(np.round(np.arange(1.0, 1.501, 0.125), 3))

RECIPES = {
    # name:        seeds        clf lags  reg lags  rule
    'v2':         dict(seeds=range(42, 47), clf_lags=False, reg_lags=False, rule='argmax'),
    'v3':         dict(seeds=range(42, 62), clf_lags=False, reg_lags=False, rule='prior',
                       alpha=0.5),
    'v4_hybrid':  dict(seeds=range(42, 62), clf_lags=True, reg_lags=False, rule='prior',
                       alpha=0.5),
    'v4_lags':    dict(seeds=range(42, 62), clf_lags=True, reg_lags=True, rule='prior',
                       alpha=0.5),
    'v4_simple':  dict(seeds=range(42, 62), clf_lags=True, reg_lags=False, rule='fuel',
                       weight=1.25),
    # alpha 0.75 rather than 0.5. The two validation folds disagreed on alpha
    # (fold 1 preferred 0.75, fold 2 preferred 0.5) so 0.5 was kept as the
    # default; the public board was then taken as a pre-registered third reading
    # and favoured 0.75. Both are carried as candidates.
    'v4_alpha075': dict(seeds=range(42, 62), clf_lags=True, reg_lags=False, rule='prior',
                        alpha=0.75),
    # Pre-registered diagnostic. Identical to v4_alpha075 on the label side --
    # same lag classifier, same 20 seeds, same prior matching at alpha 0.75, so
    # the Opportunity column is identical row for row -- and differs only in the
    # hurdle regressions, which add the r1 renewal block of features5.py to the
    # base features. Validation gain +0.00093 at 20 seeds, improving all four
    # RMSE/fold combinations but short of the +0.0015 keep bar, which is why it
    # ships as a diagnostic rather than a candidate.
    'v5_reg':     dict(seeds=range(42, 62), clf_lags=True, reg_lags=False,
                       reg_renew=True, rule='prior', alpha=0.75),
}
DEFAULT_OUT = {k: f'submissions/submission_{k}.csv' for k in RECIPES}
DEFAULT_OUT['v5_reg'] = 'submissions/submission_diag_v5_reg.csv'


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


def bag_clf(Xtr, ycode, Xva, labels, seeds):
    p = np.zeros((len(Xva), len(labels)))
    for s in seeds:
        m = lgb.LGBMClassifier(random_state=s, verbose=-1, **CLF).fit(Xtr, ycode)
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


# --- decision rules ----------------------------------------------------------

def ipf(proba, target, labels, match_idx, iters=400, damp=0.5):
    """Iterative proportional fitting on the argmax counts: nudge each matched
    class's weight by (target share / predicted share). argmax is piecewise
    constant so exact matching is not always reachable; the best iterate wins."""
    n = len(proba)
    w = np.ones(len(labels))
    best, best_w = np.inf, w.copy()
    for _ in range(iters):
        cnt = np.bincount((proba * w).argmax(axis=1), minlength=len(labels)) / n
        err = float(np.abs(cnt[match_idx] - target[match_idx]).sum())
        if err < best:
            best, best_w = err, w.copy()
        upd = np.ones(len(labels))
        for j in match_idx:
            t, c = target[j], cnt[j]
            upd[j] = 0.7 if (t <= 0 and c > 0) else (1.5 if c <= 0 else (t / c) ** damp)
        w = np.clip(w * upd, 1e-8, 1e8)
        w = w / np.exp(np.log(w).mean())      # argmax is scale-invariant
    return best_w, best


def target_mix(snaps, cutoffs, for_cutoff, labels):
    """True class mix of the most recent snapshot fully observed before
    `for_cutoff`. Uses only labels this run computed."""
    c = trainable_for(cutoffs, for_cutoff)[-1]
    s = snaps[c][1].Opportunity.value_counts(normalize=True)
    return np.array([s.get(l, 0.0) for l in labels]), c


def apply_rule(proba, labels, recipe, snaps, cutoffs, for_cutoff):
    """Turn probabilities into labels under the recipe's decision rule."""
    P = proba.to_numpy()
    if recipe['rule'] == 'argmax':
        return np.array(labels)[P.argmax(axis=1)], ''
    if recipe['rule'] == 'fuel':
        w = np.ones(len(labels))
        w[labels.index(FUEL_LABEL)] = recipe['weight']
        return np.array(labels)[(P * w).argmax(axis=1)], f'Fuel x{recipe["weight"]}'
    if recipe['rule'] == 'prior':
        tgt, src = target_mix(snaps, cutoffs, for_cutoff, labels)
        w, err = ipf(P, tgt, labels, [labels.index(l) for l in BIG4])
        return (np.array(labels)[(P * w ** recipe['alpha']).argmax(axis=1)],
                f'prior match to {src}, alpha {recipe["alpha"]}, mix-err {err:.3f}')
    raise ValueError(recipe['rule'])


# --- pipeline ----------------------------------------------------------------

def build_snapshots(d, cfg, cats, cutoffs, with_lags, with_renew=False, verbose=True):
    import features3 as F3
    snaps = {}
    for c in cutoffs:
        if verbose:
            print(f'  snapshot {c}', flush=True)
        X, y = S.build_one(d, c, cfg, cats, blocks=('base',),
                           with_labels=c != TEST_CUTOFF)
        snaps[c] = (X, y)
    lags = None
    if with_lags:
        lags = {}
        for c in cutoffs:
            if verbose:
                print(f'  lag features {c}', flush=True)
            lags[c] = F3.build(d, c, snaps[c][0].index)
    renew = None
    if with_renew:
        import features5 as F5     # numpy/pandas only, like features3
        renew = {}
        for c in cutoffs:
            if verbose:
                print(f'  renewal features {c}', flush=True)
            renew[c] = F5.build_r1(d, c, snaps[c][0].index, cfg)
    return snaps, lags, renew


def stack(snaps, cutoffs, extra=None):
    """Training matrix over `cutoffs`, optionally joined to an extra-feature
    dict (the lag series, or the renewal block) cutoff by cutoff. Joining per
    cutoff matters: IDs repeat across snapshots, so a join after the concat
    would be a many-to-many join on a non-unique index."""
    parts = []
    for c in cutoffs:
        X = snaps[c][0]
        parts.append(X if extra is None else X.join(extra[c]))
    X = pd.concat(parts)
    y = pd.concat([snaps[c][1] for c in cutoffs])
    assert len(X) == len(y)
    return X, y


def feats(snaps, extra, c):
    """One cutoff's prediction matrix, matching `stack`'s column order."""
    X = snaps[c][0]
    return X if extra is None else X.join(extra[c])


def clf_extra(rec, lags):
    return lags if rec['clf_lags'] else None


def reg_extra(rec, lags, renew):
    """Which extra features the hurdle regressions see. `reg_renew` and
    `reg_lags` are mutually exclusive across the recipe table."""
    if rec.get('reg_renew'):
        return renew
    return lags if rec['reg_lags'] else None


def fuel_weight_scan(snaps, cutoffs, lags, labels, code, seeds):
    """Re-validate the single Fuel-growth weight on these probabilities: each
    fold's own optimum, and each weight's held-out F1 from the other fold."""
    print('\nFuel-growth weight scan on the lag classifier probabilities')
    pr, truth = {}, {}
    for v in FOLDS:
        tr = trainable_for(cutoffs, v)
        Xtr, ytr = stack(snaps, tr, lags)
        Xva = feats(snaps, lags, v)
        pr[v] = bag_clf(Xtr, ytr.Opportunity.map(code).to_numpy(), Xva, labels,
                        seeds).to_numpy()
        truth[v] = snaps[v][1].Opportunity.to_numpy()
    iF = labels.index(FUEL_LABEL)
    print(f'  {"weight":>8}' + ''.join(f'{v:>14}' for v in FOLDS) + f'{"mean":>9}')
    table = {}
    for m in FUEL_GRID:
        w = np.ones(len(labels)); w[iF] = m
        fs = [f1_score(truth[v], np.array(labels)[(pr[v] * w).argmax(axis=1)],
                       average='weighted', zero_division=0) for v in FOLDS]
        table[float(m)] = fs
        print(f'  {m:>8.3f}' + ''.join(f'{f:>14.4f}' for f in fs) + f'{np.mean(fs):>9.4f}')
    own = {v: max(table, key=lambda m: table[m][i]) for i, v in enumerate(FOLDS)}
    print('  own-fold optimum: ' + ' | '.join(f'{v} {own[v]}' for v in FOLDS))
    for i, v in enumerate(FOLDS):
        other = FOLDS[1 - i]
        print(f'  weight {own[other]} fitted on {other} -> held-out F1 on {v} '
              f'{table[own[other]][i]:.4f}')
    agree = len(set(own.values())) == 1
    chosen = list(own.values())[0] if agree else 1.25
    print(f'  folds {"AGREE" if agree else "DISAGREE"} -> using Fuel x{chosen}'
          + ('' if agree else ' (the default; both folds must agree to change it)'))
    return chosen, pr, truth


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--recipe', default='v4_hybrid', choices=sorted(RECIPES))
    ap.add_argument('--train', default='data/train.csv')
    ap.add_argument('--test', default='data/test.csv')
    ap.add_argument('--config', default='src/label_config.json')
    ap.add_argument('--out', default=None)
    ap.add_argument('--cache-test-proba', default=None,
                    help='optional parquet path for the test-cutoff probabilities')
    ap.add_argument('--skip-validation', action='store_true')
    a = ap.parse_args()
    t0 = time.monotonic()   # monotonic: a wall-clock adjustment mid-run
                            # corrupted an earlier timing report
    rec = dict(RECIPES[a.recipe])
    seeds = tuple(rec['seeds'])
    out = a.out or DEFAULT_OUT[a.recipe]
    cfg = json.loads(Path(a.config).read_text())
    labels = cfg['opportunity_labels']
    code = {l: i for i, l in enumerate(labels)}
    cats = S.categories(cfg)
    print(f'recipe {a.recipe}: seeds {seeds[0]}-{seeds[-1]} ({len(seeds)}), '
          f'classifier lags {rec["clf_lags"]}, regression lags {rec["reg_lags"]}, '
          f'rule {rec["rule"]}')
    print(f'out {out}')

    d = S.load_data(a.train, cfg)
    cutoffs = monthly()
    need_lags = rec['clf_lags'] or rec['reg_lags']
    print(f'building {len(cutoffs) + 1} snapshots from {a.train}')
    snaps, lags, renew = build_snapshots(d, cfg, cats, cutoffs + [TEST_CUTOFF],
                                         need_lags, rec.get('reg_renew', False))
    XC = clf_extra(rec, lags)
    XR = reg_extra(rec, lags, renew)

    if rec['rule'] == 'fuel' and not a.skip_validation:
        rec['weight'], _, _ = fuel_weight_scan(snaps, cutoffs, XC, labels, code, seeds)

    if not a.skip_validation:
        scores = []
        for v in FOLDS:
            tr = trainable_for(cutoffs, v)
            Xc, yc = stack(snaps, tr, XC)
            Xvc = feats(snaps, XC, v)
            proba = bag_clf(Xc, yc.Opportunity.map(code).to_numpy(), Xvc, labels, seeds)
            opp, note = apply_rule(proba, labels, rec, snaps, cutoffs, v)
            Xr, yr = stack(snaps, tr, XR)
            Xvr = feats(snaps, XR, v)
            reg = {t: hurdle(Xr, yr[t], Xvr, seeds) for t in TARGETS}
            yva = snaps[v][1]
            f1 = f1_score(yva.Opportunity, opp, average='weighted', zero_division=0)
            rf = float(np.sqrt(((reg['CLV_fuel'] - yva.CLV_fuel) ** 2).mean()))
            rn = float(np.sqrt(((reg['CLV_nonfuel'] - yva.CLV_nonfuel) ** 2).mean()))
            sc = combined(f1, rf, rn)
            scores.append((f1, rf, rn, sc))
            print(f'  fold {v}: {len(tr)} snapshots, {len(Xc)} rows -> F1 {f1:.4f} | '
                  f'rmse_fuel {rf:.4f} | rmse_nonfuel {rn:.4f} | score {sc:.5f}'
                  + (f'  [{note}]' if note else ''))
        m = np.mean(scores, axis=0)
        print(f'VALIDATION mean: F1 {m[0]:.4f} | rmse_fuel {m[1]:.4f} | '
              f'rmse_nonfuel {m[2]:.4f} | score {m[3]:.5f}')

    Xc, yc = stack(snaps, cutoffs, XC)
    Xtc = feats(snaps, XC, TEST_CUTOFF)
    print(f'final fit: {len(cutoffs)} snapshots, {len(Xc)} rows, '
          f'{Xc.shape[1]} classifier features')
    proba = bag_clf(Xc, yc.Opportunity.map(code).to_numpy(), Xtc, labels, seeds)
    if a.cache_test_proba:
        Path(a.cache_test_proba).parent.mkdir(exist_ok=True)
        proba.rename_axis('ID').reset_index().to_parquet(a.cache_test_proba, index=False)
        print(f'  cached test probabilities -> {a.cache_test_proba}')
    opp, note = apply_rule(proba, labels, rec, snaps, cutoffs, TEST_CUTOFF)
    if note:
        print(f'  decision rule: {note}')
    Xr, yr = stack(snaps, cutoffs, XR)
    Xtr_ = feats(snaps, XR, TEST_CUTOFF)
    print(f'           {Xr.shape[1]} regression features')
    reg = {t: hurdle(Xr, yr[t], Xtr_, seeds) for t in TARGETS}

    test_ids = pd.read_csv(a.test, dtype=str).ID
    sub = pd.DataFrame({'CLV_fuel': reg['CLV_fuel'], 'CLV_nonfuel': reg['CLV_nonfuel'],
                        'Opportunity': opp}, index=Xtc.index).reindex(test_ids)
    # validate before writing, never after
    assert len(sub) == 5488, f'expected 5488 rows, got {len(sub)}'
    assert sub.index.equals(pd.Index(test_ids)), 'IDs do not match the test file'
    assert sub.notna().all().all(), 'missing predictions'
    bad = set(sub.Opportunity) - set(labels)
    assert not bad, f'labels outside the config: {bad}'
    assert (sub[['CLV_fuel', 'CLV_nonfuel']] >= 0).all().all(), 'negative CLV'
    assert list(sub.columns) == ['CLV_fuel', 'CLV_nonfuel', 'Opportunity'], sub.columns
    Path(out).parent.mkdir(exist_ok=True)
    sub.rename_axis('ID').reset_index().to_csv(out, index=False, float_format='%.17g')
    print(f'wrote {out}: {len(sub)} rows, IDs match, all labels in config, '
          f'CLV min {sub[list(TARGETS)].min().min():.4f}')
    print(f'total runtime {time.monotonic() - t0:.0f}s')


if __name__ == '__main__':
    main()
