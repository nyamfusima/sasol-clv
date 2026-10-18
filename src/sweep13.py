"""Sweep 13: the boosting horizon of the REGULARISED classifier, plus a
per-class audit of the shipped model. Classifier only; the hurdles never move.

WHY THIS AND NOT SOMETHING ELSE
-------------------------------
Sweep 11 was motivated by a measured over-training curve: validation
multi-logloss got WORSE from 200 to 400 trees on both folds (notes.md, sweep
11). It fixed that by shrinking capacity -- num_leaves 15 -> 7,
min_child_samples 30 -> 100, reg_lambda 0 -> 5 -- and pre-registered that
"tree counts and learning rates do not move". So the knob whose curve
diagnosed the overfitting was deliberately held at the value that demonstrated
it, and has never been re-measured against the regularised model.

Everything else on the classifier is closed by the record:
  sweep 2 A3    capacity grid, early stopping 103..224 trees -- on the
                UNREGULARISED config, 5 seeds, base features, argmax F1
  sweep 5 L2    6 model families + 5 blends -- decorrelation/accuracy monotone
  sweep 5 L3/L4 48 fine classes, 5 groups, one-vs-rest -- 17-class is optimum
  sweep 2 A1/A2 recency weighting (3 half-lives), snapshot spacing incl.
                quarterly non-overlapping -- flat
  sweep 7 E1    expected-F1 decoding; block H/I prior matching, alpha grid
  block I3      per-class isotonic + global temperature (== alpha, algebraic)
  sweep 10      Fuel/Stable boundary, closed by its own oracle
  sweep 12      colsample_bytree 0.4 -- failed
  external      true-mix oracle ~0, alpha curve, rare-class override, station
                context, forward stacking, customer ID, regularisation grid
The record's own conclusion is that the only remaining axis is "the quality of
the 17-class probabilities themselves". A boosting-horizon error is the one
untested way to be wrong about that, and it is the cheapest to measure.

HOW IT IS MEASURED
------------------
LightGBM boosting is sequential and its subsampling RNG at iteration t depends
only on iterations 1..t, so trees 1..k of a 1000-tree fit ARE the k-tree fit.
One fit per seed per fold therefore yields the whole curve via
`predict_proba(num_iteration=k)`. This identity is ASSERTED, not assumed:
--check refits at a grid point and compares to the staged prediction.

    arm A      k = 400, the shipped v6_clf (aUswGpyi) classifier
    arm B      k = k*, chosen on fold 2025-06 only (see protocol below)
Hurdles are the incumbent v5_fuel2 ones, read from preds/sweep11.json, so both
arms' CLV columns are byte-identical to submission_v5_fuel2.csv and
score_delta == 0.4 * F1_delta exactly. That identity is asserted at runtime.

PRE-REGISTERED, written before any sweep-13 number existed. Alpha is NOT
retuned: 0.75 and 0.5 exactly as they ship.

  Protocol. Sweep 2 A3 recorded that fold 2025-06 is the only legitimate
  selection set for a tree-count choice, because early stopping spends it;
  anything chosen on fold 2025-09 must be re-measured. So k* is chosen on
  fold 2025-06 by multi-logloss at the screen seeds, and fold 2025-09 is the
  clean check. k* is frozen before the confirmation run.

  Keep rule, all four required:
    1. multi-logloss better on BOTH folds at k*
    2. mean multi-logloss gain >= 0.005
    3. mean composite score gain >= +0.0015 at alpha 0.75
    4. better on BOTH folds at alpha 0.75

  Condition 2's bar: sweep 11's logloss gain was -0.0269/-0.0161 and it still
  did not transfer; sweep 12's was +0.00002/-0.0031 and was called noise.
  0.005 sits above sweep 12's larger fold movement and well below sweep 11's,
  so it admits only a genuine probability-quality gain.

  Conditions 3 and 4 are the project's standing bar, unchanged. If 1 and 2
  pass while 3 or 4 fails, that is reported as a probability gain that does
  not convert into decisions -- informative, NOT adoptable.

    python src/sweep13.py --audit                 # Phase 1, per-class, 5 seeds
    python src/sweep13.py --curve --seeds 5       # the screen
    python src/sweep13.py --curve --confirm       # 20 seeds at frozen k*
    python src/sweep13.py --check                 # staged-prediction identity
"""
import argparse, json, time
from pathlib import Path
import numpy as np, pandas as pd
import lightgbm as lgb
from sklearn.metrics import f1_score, log_loss, confusion_matrix
from sklearn.metrics import precision_recall_fscore_support
import make_submission as M
import sweep2 as W
import sweep5 as S5
import validate as V

RESULTS = Path('preds/sweep13.json')
SWEEP11 = Path('preds/sweep11.json')
NL = chr(10)
ALPHAS = (0.75, 0.5)
KEEP_BAR = 0.0015
LOGLOSS_BAR = 0.005
A_RECIPE = 'v6_clf'
SHIPPED_K = 400
MAX_TREES = 1000
K_GRID = (50, 100, 150, 200, 250, 300, 350, 400, 500, 600, 700, 800, 1000)
SELECT_FOLD = '2025-06-01'          # sweep 2 A3's rule
CHECK_K = 200                       # grid point used by --check


def clf_params():
    """The shipped v6_clf classifier params, read from the recipe not restated."""
    return dict(M.params_for(M.RECIPES[A_RECIPE])['clf'])


def cached_reg():
    """Incumbent v5_fuel2 hurdle RMSEs at 20 seeds, from sweep 11."""
    rows = json.loads(SWEEP11.read_text())['confirm']['rows']
    out = {}
    for v in V.FOLDS:
        a, b = rows['A (incumbent)'][v], rows['B_clf'][v]
        assert a['rmse_fuel'] == b['rmse_fuel'], (v, 'fuel')
        assert a['rmse_nonfuel'] == b['rmse_nonfuel'], (v, 'nonfuel')
        out[v] = (a['rmse_fuel'], a['rmse_nonfuel'])
    return out


def shipped_baseline():
    """sweep 11's B_clf arm IS v6_clf as shipped, 20 seeds."""
    rows = json.loads(SWEEP11.read_text())['confirm']['rows']
    return {v: rows['B_clf'][v] for v in V.FOLDS}


def fold_data(ctx, v, lags):
    tr = M.trainable_for(ctx.monthly, v)
    Xtr, ytr = M.stack(ctx.snaps, tr, lags)
    Xva = M.feats(ctx.snaps, lags, v)
    code = {l: i for i, l in enumerate(ctx.labels)}
    ycode = ytr.Opportunity.map(code).to_numpy()
    yva = ctx.snaps[v][1].Opportunity
    return Xtr, ycode, Xva, yva, code, len(tr)


def staged_proba(Xtr, ycode, Xva, labels, seeds, params, ks, n_est=MAX_TREES):
    """One fit per seed at n_est trees; bagged probabilities at every k in ks.

    Trees 1..k of this fit are exactly the k-tree fit (asserted by --check),
    so this returns the whole horizon curve for the cost of one long fit.
    """
    acc = {k: np.zeros((len(Xva), len(labels))) for k in ks}
    for s in seeds:
        m = lgb.LGBMClassifier(random_state=s, verbose=-1,
                               **{**params, 'n_estimators': n_est}).fit(Xtr, ycode)
        cls = list(m.classes_)
        for k in ks:
            p = m.predict_proba(Xva, num_iteration=k)
            acc[k] += M.wide(p, cls, labels, Xva.index).to_numpy()
    return {k: pd.DataFrame(a / len(seeds), index=Xva.index, columns=list(labels))
            for k, a in acc.items()}


def decide(ctx, proba, v, alpha):
    """The shipped rule: IPF prior matching to the stale mix, damped by alpha."""
    P = proba.to_numpy()
    tgt, _ = M.target_mix(ctx.snaps, ctx.monthly, v, ctx.labels)
    w, _ = M.ipf(P, tgt, ctx.labels, [ctx.labels.index(l) for l in M.BIG4])
    return np.asarray(ctx.labels)[(P * w ** alpha).argmax(axis=1)]


def evaluate(ctx, proba, v, yva, code, reg):
    rf, rn = reg[v]
    y = yva.to_numpy()
    e = dict(logloss=float(log_loss(yva.map(code).to_numpy(), proba.to_numpy(),
                                    labels=list(range(len(ctx.labels))))),
             rmse_fuel=rf, rmse_nonfuel=rn)
    for al in ALPHAS:
        pred = decide(ctx, proba, v, al)
        e[f'f1_a{al}'] = float(f1_score(y, pred, average='weighted', zero_division=0))
        e[f'score_a{al}'] = M.combined(e[f'f1_a{al}'], rf, rn)
    return e


def save(key, obj):
    RESULTS.parent.mkdir(exist_ok=True)
    allr = json.loads(RESULTS.read_text()) if RESULTS.exists() else {}
    allr[key] = obj
    RESULTS.write_text(json.dumps(allr, indent=2, default=float))


def seeds_for(args):
    if args.seeds:
        return tuple(range(42, 42 + args.seeds))
    return tuple(S5.CONFIRM if args.confirm else S5.SCREEN)


def setup(ctx):
    cuts = sorted(set(ctx.monthly + list(V.FOLDS) + [W.TEST_CUTOFF]))
    return S5.lag_extra(ctx, cuts)


# --- Phase 1: audit ----------------------------------------------------------

def track_audit(ctx, args):
    """Per-class precision/recall/F1 and confusion matrices, both folds, for
    the SHIPPED classifier read through the SHIPPED rule. Diagnosis only."""
    seeds = seeds_for(args)
    params = clf_params()
    reg = cached_reg()
    lags = setup(ctx)
    print(NL + f'--- Phase 1 audit: shipped v6_clf, k={SHIPPED_K}, '
          f'{len(seeds)} seeds {min(seeds)}-{max(seeds)} ---')
    print(f'    clf params: {params}')
    out = {}
    for v in V.FOLDS:
        t0 = time.monotonic()
        Xtr, ycode, Xva, yva, code, nsnap = fold_data(ctx, v, lags)
        pr = staged_proba(Xtr, ycode, Xva, ctx.labels, seeds, params, (SHIPPED_K,),
                          n_est=SHIPPED_K)[SHIPPED_K]
        e = evaluate(ctx, pr, v, yva, code, reg)
        print(NL + f'  == fold {v}: {len(Xtr)} train rows / {nsnap} snapshots, '
              f'{len(Xva)} val rows  [{time.monotonic() - t0:.0f}s]')
        print(f'     logloss {e["logloss"]:.4f} | F1 a0.75 {e["f1_a0.75"]:.4f} | '
              f'F1 a0.5 {e["f1_a0.5"]:.4f} | score a0.75 {e["score_a0.75"]:.5f}')
        y = yva.to_numpy()
        rows = {}
        for al in ALPHAS:
            pred = decide(ctx, pr, v, al)
            p, r, f, s = precision_recall_fscore_support(
                y, pred, labels=ctx.labels, zero_division=0)
            npred = pd.Series(pred).value_counts().reindex(ctx.labels, fill_value=0)
            t = pd.DataFrame({'support': s, 'predicted': npred.to_numpy(),
                              'precision': p, 'recall': r, 'F1': f},
                             index=ctx.labels).sort_values('support', ascending=False)
            print(NL + f'     -- alpha {al}: per-class --')
            print(t.to_string(float_format=lambda x: f'{x:.4f}'))
            cm, g = V.grouped_report(y, pred)
            print(NL + f'     -- alpha {al}: 4-group confusion (true rows) --')
            print(cm.to_string())
            print('     grouped F1: ' + ' | '.join(f'{k} {x:.4f}' for k, x in g.items()))
            rows[f'alpha {al}'] = {'per_class': t.to_dict(), 'confusion': cm.to_dict(),
                                   'grouped_f1': g.to_dict()}
        out[v] = {'metrics': e, 'rules': rows}
    save(f'audit_seeds{len(seeds)}', {'seeds': list(seeds), 'k': SHIPPED_K,
                                      'params': params, 'folds': out})
    return out


# --- validity check ----------------------------------------------------------

def track_check(ctx, args):
    """Assert trees 1..k of a long fit equal a k-tree fit, same seed."""
    params = clf_params()
    lags = setup(ctx)
    v = SELECT_FOLD
    Xtr, ycode, Xva, yva, code, _ = fold_data(ctx, v, lags)
    s = 42
    print(NL + f'--- staged-prediction identity check, fold {v}, seed {s}, '
          f'k={CHECK_K} ---')
    t0 = time.monotonic()
    long_m = lgb.LGBMClassifier(random_state=s, verbose=-1,
                                **{**params, 'n_estimators': MAX_TREES}).fit(Xtr, ycode)
    staged = long_m.predict_proba(Xva, num_iteration=CHECK_K)
    print(f'    long fit ({MAX_TREES} trees) [{time.monotonic() - t0:.0f}s]')
    t0 = time.monotonic()
    short_m = lgb.LGBMClassifier(random_state=s, verbose=-1,
                                 **{**params, 'n_estimators': CHECK_K}).fit(Xtr, ycode)
    direct = short_m.predict_proba(Xva)
    print(f'    short fit ({CHECK_K} trees) [{time.monotonic() - t0:.0f}s]')
    d = float(np.abs(staged - direct).max())
    print(f'    max |staged - direct| = {d:.3e}')
    ok = d < 1e-9
    print(f'    -> identity holds: {ok}')
    save('identity_check', {'fold': v, 'seed': s, 'k': CHECK_K,
                            'max_abs_diff': d, 'holds': bool(ok)})
    assert ok, f'staged prediction does not match a direct fit (max diff {d:.3e})'
    return ok


# --- Phase 2/3: the horizon curve -------------------------------------------

def track_curve(ctx, args):
    seeds = seeds_for(args)
    tag = {20: 'confirm', 5: 'screen'}.get(len(seeds), f'seeds{len(seeds)}')
    params = clf_params()
    reg = cached_reg()
    lags = setup(ctx)
    ks = tuple(sorted(set(K_GRID) | {SHIPPED_K}))
    if args.k_star:                       # confirmation: only the two arms
        ks = tuple(sorted({SHIPPED_K, args.k_star}))
    print(NL + f'--- sweep 13 horizon curve, {len(seeds)} seeds '
          f'{min(seeds)}-{max(seeds)}, k in {ks} ---')
    print(f'    clf params (n_estimators overridden): {params}')
    print(f'    hurdles: incumbent v5_fuel2, 20 seeds, from sweep 11')
    n_est = max(ks)
    rows = {}
    for v in V.FOLDS:
        t0 = time.monotonic()
        Xtr, ycode, Xva, yva, code, nsnap = fold_data(ctx, v, lags)
        probas = staged_proba(Xtr, ycode, Xva, ctx.labels, seeds, params, ks,
                              n_est=n_est)
        rows[v] = {k: evaluate(ctx, probas[k], v, yva, code, reg) for k in ks}
        print(f'  fold {v}: {len(Xtr)} rows, {len(seeds)} fits at {n_est} trees '
              f'[{time.monotonic() - t0:.0f}s]', flush=True)

    f0, f1_ = V.FOLDS
    print(NL + f'  {"k":>6}{"logl 06":>10}{"logl 09":>10}{"logl mean":>11}'
          f'{"F1.75 06":>10}{"F1.75 09":>10}{"score.75 mean":>14}'
          f'{"F1.5 06":>10}{"F1.5 09":>10}')
    for k in ks:
        a, b = rows[f0][k], rows[f1_][k]
        lm = np.mean([a['logloss'], b['logloss']])
        sm = np.mean([a['score_a0.75'], b['score_a0.75']])
        mark = '  <- shipped' if k == SHIPPED_K else ''
        print(f'  {k:>6}{a["logloss"]:>10.4f}{b["logloss"]:>10.4f}{lm:>11.4f}'
              f'{a["f1_a0.75"]:>10.4f}{b["f1_a0.75"]:>10.4f}{sm:>14.5f}'
              f'{a["f1_a0.5"]:>10.4f}{b["f1_a0.5"]:>10.4f}{mark}')

    # --- selection, on fold 2025-06 only -----------------------------------
    sel = {k: rows[SELECT_FOLD][k]['logloss'] for k in ks}
    k_star = args.k_star or min(sel, key=sel.get)
    other = [v for v in V.FOLDS if v != SELECT_FOLD][0]
    print(NL + f'  selection fold {SELECT_FOLD} (sweep 2 A3 protocol): '
          f'best logloss at k={k_star} ({sel[k_star]:.4f})')
    print(f'  shipped k={SHIPPED_K} on that fold: {sel[SHIPPED_K]:.4f}')
    print(f'  clean check fold {other}: k={k_star} {rows[other][k_star]["logloss"]:.4f} '
          f'vs k={SHIPPED_K} {rows[other][SHIPPED_K]["logloss"]:.4f}')

    verdict = report_rule(rows, k_star)
    out = {'seeds': list(seeds), 'ks': list(ks), 'k_star': int(k_star),
           'k_star_frozen': bool(args.k_star), 'select_fold': SELECT_FOLD,
           'params': params, 'n_estimators_fit': int(n_est), 'rows': rows,
           'verdict': verdict}
    save(tag, out)
    return out


def report_rule(rows, k_star):
    """The four pre-registered conditions, evaluated and printed."""
    ldl = [rows[v][k_star]['logloss'] - rows[v][SHIPPED_K]['logloss'] for v in V.FOLDS]
    v75 = {}
    for al in ALPHAS:
        dl = [rows[v][k_star][f'score_a{al}'] - rows[v][SHIPPED_K][f'score_a{al}']
              for v in V.FOLDS]
        fd = [rows[v][k_star][f'f1_a{al}'] - rows[v][SHIPPED_K][f'f1_a{al}']
              for v in V.FOLDS]
        for x, y in zip(dl, fd):      # RMSE half fixed -> score is affine in F1
            assert abs(x - 0.4 * y) < 1e-12, (x, y)
        v75[f'alpha {al}'] = dict(score_deltas=dl, f1_deltas=fd,
                                  mean_gain=float(np.mean(dl)),
                                  mean_f1_gain=float(np.mean(fd)),
                                  better_both_folds=all(x > 0 for x in dl))
        print(f'  alpha {al}: score {dl[0]:+.5f} / {dl[1]:+.5f}  '
              f'mean {np.mean(dl):+.5f}   (F1 {fd[0]:+.4f} / {fd[1]:+.4f})')
    print(f'  logloss delta: {ldl[0]:+.5f} / {ldl[1]:+.5f}  '
          f'mean {np.mean(ldl):+.5f}')

    c1 = all(x < 0 for x in ldl)
    c2 = -float(np.mean(ldl)) >= LOGLOSS_BAR
    c3 = v75['alpha 0.75']['mean_gain'] >= KEEP_BAR
    c4 = v75['alpha 0.75']['better_both_folds']
    print(NL + f'  --- keep rule at k={k_star} (all four required) ---')
    print(f'  1. logloss better on BOTH folds              : '
          f'-> {"PASS" if c1 else "FAIL"}')
    print(f'  2. mean logloss gain >= {LOGLOSS_BAR}                : '
          f'{-float(np.mean(ldl)):+.5f} -> {"PASS" if c2 else "FAIL"}')
    print(f'  3. mean score gain >= +{KEEP_BAR} at alpha 0.75 : '
          f'{v75["alpha 0.75"]["mean_gain"]:+.5f} -> {"PASS" if c3 else "FAIL"}')
    print(f'  4. better on BOTH folds at alpha 0.75        : '
          f'-> {"PASS" if c4 else "FAIL"}')
    passes = bool(c1 and c2 and c3 and c4)
    print(NL + f'  ==> SWEEP 13 {"PASSES" if passes else "FAILS"} at k={k_star}')
    return {'k_star': int(k_star), 'logloss_deltas': ldl, 'alphas': v75,
            'conditions': {'logloss_both_folds': c1, 'logloss_bar': c2,
                           'score_bar': c3, 'score_both_folds': c4},
            'passes': passes}


TRACKS = {'audit': track_audit, 'curve': track_curve, 'check': track_check}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--train', default='data/train.csv')
    ap.add_argument('--audit', action='store_true')
    ap.add_argument('--curve', action='store_true')
    ap.add_argument('--check', action='store_true')
    ap.add_argument('--confirm', action='store_true')
    ap.add_argument('--seeds', type=int, default=0,
                    help='explicit seed count; 42..42+n-1')
    ap.add_argument('--k-star', type=int, default=0,
                    help='frozen k for the confirmation run')
    a = ap.parse_args()
    blocks = [b for b in ('check', 'audit', 'curve') if getattr(a, b)]
    if not blocks:
        ap.error('pick at least one of --check / --audit / --curve')
    ctx = W.Ctx(a.train)
    for b in blocks:
        print(NL + f'########## block {b} ##########')
        t0 = time.monotonic()
        TRACKS[b](ctx, a)
        print(f'[block {b} took {time.monotonic() - t0:.0f}s]')


if __name__ == '__main__':
    main()
