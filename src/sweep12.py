"""Sweep 12: classifier feature subsampling, the last untouched knob.

Sweep 11 regularised the 17-class model through num_leaves, min_child_samples
and reg_lambda and left `colsample_bytree` at the 0.7 that was set by hand on
day one. A screen outside the repo (base 97 features, 3 seeds, everything else
as v6_clf) says 0.7 -> 0.4 is worth something:

    logloss      -0.0005 / -0.0006
    F1 alpha .75 +0.0056 / +0.0026
    F1 alpha .5  +0.0024 / -0.0031

That is a lead and not a result. The logloss barely moves, which is the whole
problem with it: the F1 movement is nine times the logloss movement in scale,
and every time this project has seen that shape the gain has been the decision
rule reacting to sharpness rather than a better model. Sweep 11 was exactly
this and lost 0.00106 on the public board. So the F1 gain is treated as
decision noise until 20 seeds say otherwise, and the keep rule below adds a
logloss condition the earlier sweeps did not have.

    arm A      v6_clf as shipped (aUswGpyi): lag classifier, num_leaves 7,
               min_child_samples 100, reg_lambda 5, 400 trees, lr 0.03,
               colsample_bytree 0.7; incumbent v5_fuel2 hurdles
    arm B      identical except classifier colsample_bytree 0.4

The gate and magnitude models are NOT touched. Regression changes have twice
failed to transfer (sweep 11's hurdle half, the r1 bundle), so the hurdles stay
at the incumbent and both arms' CLV columns remain byte-identical to
submission_v5_fuel2.csv. That also makes the comparison clean: the score is
affine in F1 with the RMSE half held fixed, so score_delta == 0.4 * F1_delta
exactly, and the test is a pure paired F1 comparison on one set of hurdles.

PRE-REGISTERED, written before any 20-seed result was seen. One setting only,
no grid. Alpha is NOT retuned -- 0.75 and 0.5 exactly as they ship.

  Keep rule, all three required:
    1. mean score gain >= +0.0015 at alpha 0.75
    2. better on BOTH folds at alpha 0.75
    3. multi-logloss not worse on EITHER fold

  Condition 3 is new. It is there because conditions 1 and 2 passed for sweep
  11 and the change still lost on the board; the one thing sweep 11 could not
  show was a probability-quality gain big enough to explain its F1 movement.

Arm A needs no refit: sweep 11's `B_clf` arm IS v6_clf as shipped -- the
regularised classifier on the incumbent hurdles -- and it was measured at 20
seeds on both folds. It is read from preds/sweep11.json and re-derived here as
a cross-check rather than trusted blindly; --fresh-a refits it instead.

    python src/sweep12.py --seeds 1 --fresh-reg      # harness smoke test
    python src/sweep12.py --confirm                  # the 20-seed test
"""
import argparse, json, time
from pathlib import Path
import numpy as np
from sklearn.metrics import f1_score, log_loss
import make_submission as M
import sweep2 as W
import sweep5 as S5
import sweep8 as S8
import validate as V

RESULTS = Path('preds/sweep12.json')
SWEEP11 = Path('preds/sweep11.json')
NL = chr(10)
TARGETS = ('CLV_fuel', 'CLV_nonfuel')
RENEW_BLOCKS = ('r1', 'r2', 'r3')
ALPHAS = (0.75, 0.5)
KEEP_BAR = 0.0015

# The one setting under test, fixed in advance.
B_COLSAMPLE = 0.4
# Arm A is read from the shipping recipe rather than restated, so the incumbent
# cannot drift away from what aUswGpyi actually is.
A_RECIPE = 'v6_clf'


def arm_params(which):
    """Full param dict per arm. Only clf.colsample_bytree differs."""
    p = M.params_for(M.RECIPES[A_RECIPE])
    if which == 'B':
        p['clf'] = {**p['clf'], 'colsample_bytree': B_COLSAMPLE}
    return p


def rmse(pred, truth):
    return float(np.sqrt(((np.asarray(pred) - np.asarray(truth)) ** 2).mean()))


def clf_half(ctx, v, lags, seeds, params):
    """Probabilities and multi-logloss for one fold, one classifier config."""
    tr = M.trainable_for(ctx.monthly, v)
    Xtr, ytr = M.stack(ctx.snaps, tr, lags)
    Xva = M.feats(ctx.snaps, lags, v)
    code = {l: i for i, l in enumerate(ctx.labels)}
    proba = M.bag_clf(Xtr, ytr.Opportunity.map(code).to_numpy(), Xva,
                      ctx.labels, list(seeds), params['clf'])
    yva = ctx.snaps[v][1].Opportunity
    ll = float(log_loss(yva.map(code).to_numpy(), proba.to_numpy(),
                        labels=list(range(len(ctx.labels)))))
    return proba, ll, yva.to_numpy()


def reg_half(ctx, v, renew, seeds, params):
    """The incumbent v5_fuel2 hurdles. Shared by both arms, so fit once."""
    tr = M.trainable_for(ctx.monthly, v)
    out = {}
    for t in TARGETS:
        extra = renew if t == 'CLV_fuel' else None
        Xtr, ytr = M.stack(ctx.snaps, tr, extra)
        Xva = M.feats(ctx.snaps, extra, v)
        pred = M.hurdle(Xtr, ytr[t], Xva, list(seeds), params=params)
        out[t] = rmse(pred, ctx.snaps[v][1][t])
    return out


def cached_reg(tag):
    """The incumbent hurdle RMSEs from sweep 11, which fit the same config.

    sweep 11's `A (incumbent)` and `B_clf` arms share the reg half, which is
    asserted here rather than assumed.
    """
    rows = json.loads(SWEEP11.read_text())[tag]['rows']
    out = {}
    for v in V.FOLDS:
        a, b = rows['A (incumbent)'][v], rows['B_clf'][v]
        assert a['rmse_fuel'] == b['rmse_fuel'], (v, 'fuel')
        assert a['rmse_nonfuel'] == b['rmse_nonfuel'], (v, 'nonfuel')
        out[v] = {'CLV_fuel': a['rmse_fuel'], 'CLV_nonfuel': a['rmse_nonfuel']}
    return out


def cached_arm_a(tag):
    """sweep 11's B_clf == v6_clf as shipped, at the same seeds."""
    rows = json.loads(SWEEP11.read_text())[tag]['rows']
    return {v: rows['B_clf'][v] for v in V.FOLDS}


def decide(ctx, proba, v, alpha):
    """The shipped rule: prior matching, damped by alpha. Not retuned."""
    P = proba.to_numpy()
    tgt, _ = M.target_mix(ctx.snaps, ctx.monthly, v, ctx.labels)
    w, _ = M.ipf(P, tgt, ctx.labels, [ctx.labels.index(l) for l in M.BIG4])
    return np.asarray(ctx.labels)[(P * w ** alpha).argmax(axis=1)]


def save(key, obj):
    RESULTS.parent.mkdir(exist_ok=True)
    allr = json.loads(RESULTS.read_text()) if RESULTS.exists() else {}
    allr[key] = obj
    RESULTS.write_text(json.dumps(allr, indent=2, default=float))


def track_clf(ctx, args):
    if args.seeds:
        seeds = tuple(range(42, 42 + args.seeds))
    else:
        seeds = tuple(S5.CONFIRM if args.confirm else S5.SCREEN)
    tag = {20: 'confirm', 5: 'screen'}.get(len(seeds), f'seeds{len(seeds)}')
    cuts = sorted(set(ctx.monthly + list(V.FOLDS) + [W.TEST_CUTOFF]))
    print(NL + f'--- sweep 12, {len(seeds)} seeds, colsample_bytree '
          f'0.7 -> {B_COLSAMPLE} on the classifier only ---')
    print(f'    arm A clf: {arm_params("A")["clf"]}')
    print(f'    arm B clf: {arm_params("B")["clf"]}')
    lags = S5.lag_extra(ctx, cuts)
    renew = S8.renewal_extra(ctx, cuts, list(RENEW_BLOCKS))

    # --- the regression half: one config, shared by both arms ---------------
    reg = {}
    if args.fresh_reg:
        for v in V.FOLDS:
            t0 = time.monotonic()
            reg[v] = reg_half(ctx, v, renew, seeds, arm_params('A'))
            print(f'  reg {v}: fuel {reg[v]["CLV_fuel"]:.4f} '
                  f'nonfuel {reg[v]["CLV_nonfuel"]:.4f}  '
                  f'[{time.monotonic() - t0:.0f}s]', flush=True)
    else:
        reg = cached_reg(tag)
        print(f'  reg: reused from sweep 11 ({tag}), identical config and seeds')

    # --- the classifier half: one fit per arm per fold ----------------------
    clf, truth = {}, {}
    arms = ('B',) if not args.fresh_a else ('A', 'B')
    for v in V.FOLDS:
        for which in arms:
            t0 = time.monotonic()
            proba, ll, y = clf_half(ctx, v, lags, seeds, arm_params(which))
            clf[(v, which)] = dict(logloss=ll, proba=proba)
            truth[v] = y
            print(f'  clf {which} {v}: logloss {ll:.4f}  '
                  f'[{time.monotonic() - t0:.0f}s]', flush=True)

    # --- compose both arms ---------------------------------------------------
    rows, cross = {}, {}
    cache_a = cached_arm_a(tag) if tag in ('screen', 'confirm') else None
    for which in ('A', 'B'):
        d = {}
        for v in V.FOLDS:
            if which == 'A' and not args.fresh_a:
                d[v] = dict(cache_a[v])          # measured in sweep 11
                continue
            cc, rr = clf[(v, which)], reg[v]
            e = dict(logloss=cc['logloss'], rmse_fuel=rr['CLV_fuel'],
                     rmse_nonfuel=rr['CLV_nonfuel'])
            for al in ALPHAS:
                pred = decide(ctx, cc['proba'], v, al)
                e[f'f1_a{al}'] = float(f1_score(truth[v], pred, average='weighted',
                                                zero_division=0))
                e[f'score_a{al}'] = M.combined(e[f'f1_a{al}'], e['rmse_fuel'],
                                               e['rmse_nonfuel'])
            d[v] = e
        rows[which] = d

    # arm A refitted: check it against what sweep 11 measured
    if args.fresh_a and cache_a is not None:
        for v in V.FOLDS:
            cross[v] = {k: rows['A'][v][k] - cache_a[v][k]
                        for k in ('logloss', 'f1_a0.75', 'f1_a0.5', 'score_a0.75')}
            print(f'  cross-check A vs sweep11 B_clf {v}: '
                  + '  '.join(f'{k} {d:+.2e}' for k, d in cross[v].items()))

    # --- the table -----------------------------------------------------------
    f0, f1_ = V.FOLDS
    for al in ALPHAS:
        print(NL + f'  --- alpha {al} ---')
        print(f'  {"arm":<22}{"F1 06":>9}{"F1 09":>9}{"logl 06":>10}{"logl 09":>10}'
              f'{"score 06":>10}{"score 09":>10}{"mean":>10}')
        for which, label in (('A', 'A (v6_clf, 0.7)'), ('B', f'B (colsample {B_COLSAMPLE})')):
            a, b = rows[which][f0], rows[which][f1_]
            mean = np.mean([a[f'score_a{al}'], b[f'score_a{al}']])
            print(f'  {label:<22}{a[f"f1_a{al}"]:>9.4f}{b[f"f1_a{al}"]:>9.4f}'
                  f'{a["logloss"]:>10.4f}{b["logloss"]:>10.4f}'
                  f'{a[f"score_a{al}"]:>10.5f}{b[f"score_a{al}"]:>10.5f}{mean:>10.5f}')
        print(f'  {"rmse (shared)":<22}fuel {rows["A"][f0]["rmse_fuel"]:.4f} / '
              f'{rows["A"][f1_]["rmse_fuel"]:.4f}   nonfuel '
              f'{rows["A"][f0]["rmse_nonfuel"]:.4f} / {rows["A"][f1_]["rmse_nonfuel"]:.4f}')

    # --- the pre-registered rule --------------------------------------------
    print(NL + '  --- keep rule (all three required) ---')
    verdict = {}
    for al in ALPHAS:
        dl = [rows['B'][v][f'score_a{al}'] - rows['A'][v][f'score_a{al}'] for v in V.FOLDS]
        fd = [rows['B'][v][f'f1_a{al}'] - rows['A'][v][f'f1_a{al}'] for v in V.FOLDS]
        # with the RMSE half held fixed the score is affine in F1; assert it
        for x, y in zip(dl, fd):
            assert abs(x - 0.4 * y) < 1e-12, (x, y)
        mg, mf = float(np.mean(dl)), float(np.mean(fd))
        both = all(x > 0 for x in dl)
        verdict[f'alpha {al}'] = dict(score_deltas=dl, f1_deltas=fd, mean_gain=mg,
                                      mean_f1_gain=mf, better_both_folds=both)
        print(f'  alpha {al}: score {dl[0]:+.5f} / {dl[1]:+.5f}  mean {mg:+.5f}'
              f'   (F1 {fd[0]:+.4f} / {fd[1]:+.4f}, mean {mf:+.4f})')

    ldl = [rows['B'][v]['logloss'] - rows['A'][v]['logloss'] for v in V.FOLDS]
    ll_ok = all(x <= 0 for x in ldl)
    print(f'  logloss delta: {ldl[0]:+.5f} / {ldl[1]:+.5f}  '
          f'-> not worse on either fold: {ll_ok}')

    c1 = verdict['alpha 0.75']['mean_gain'] >= KEEP_BAR
    c2 = verdict['alpha 0.75']['better_both_folds']
    c3 = ll_ok
    passes = bool(c1 and c2 and c3)
    print(NL + f'  1. mean score gain >= +{KEEP_BAR} at alpha 0.75 : '
          f'{verdict["alpha 0.75"]["mean_gain"]:+.5f}  -> {"PASS" if c1 else "FAIL"}')
    print(f'  2. better on both folds at alpha 0.75         : -> {"PASS" if c2 else "FAIL"}')
    print(f'  3. multi-logloss not worse on either fold     : -> {"PASS" if c3 else "FAIL"}')
    print(NL + f'  ==> SWEEP 12 {"PASSES" if passes else "FAILS"} at {len(seeds)} seeds')
    if verdict['alpha 0.5']['mean_gain'] >= 0:
        print('      (alpha 0.5 mean gain >= 0, so an a050 partner would be built '
              'if this passed)')
    else:
        print('      (alpha 0.5 mean gain < 0, so no a050 partner even on a pass)')

    out = {'seeds': list(seeds), 'colsample_a': arm_params('A')['clf']['colsample_bytree'],
           'colsample_b': B_COLSAMPLE, 'rows': rows, 'verdict': verdict,
           'logloss_deltas': ldl, 'conditions': {'mean_gain': c1, 'both_folds': c2,
                                                 'logloss': c3}, 'passes': passes,
           'arm_a_refitted': bool(args.fresh_a), 'cross_check': cross,
           'reg_fresh': bool(args.fresh_reg)}
    save(tag, out)
    return out


TRACKS = {'clf': track_clf}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--block', nargs='+', default=['clf'])
    ap.add_argument('--train', default='data/train.csv')
    ap.add_argument('--confirm', action='store_true')
    ap.add_argument('--seeds', type=int, default=0,
                    help='explicit seed count for a smoke test; 42..42+n-1')
    ap.add_argument('--fresh-a', action='store_true',
                    help='refit arm A instead of reading sweep 11')
    ap.add_argument('--fresh-reg', action='store_true',
                    help='refit the shared hurdles instead of reading sweep 11')
    a = ap.parse_args()
    ctx = W.Ctx(a.train)
    for b in a.block:
        print(NL + f'########## block {b} ##########')
        t0 = time.monotonic()
        TRACKS[b](ctx, a)
        print(f'[block {b} took {time.monotonic() - t0:.0f}s]')


if __name__ == '__main__':
    main()
