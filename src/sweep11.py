"""Sweep 11: LightGBM regularisation, the one thing never tuned since day one.

Every hyperparameter in this project was set by hand on day one and never
revisited; nine sweeps moved features, labels and decision rules around them.
A screen outside the repo (base features, seed 42, same tree counts) says the
settings are overfitting: validation multi-logloss gets WORSE from 200 to 400
trees on both folds, and a regularised configuration recovers it. That is a
lead, not a result -- it was one seed, on base features, outside this harness.
Sweep 11 measures it inside the harness, at the incumbent pipeline, on both
folds.

The configuration is fixed in advance and is the only one tried. No grid
search: this sweep either confirms the external lead at these exact settings or
closes it.

    arm A      the incumbent v5_fuel2 pipeline
    arm B      clf  num_leaves 7,  min_child_samples 100, reg_lambda 5 (400, lr .03)
               gate num_leaves 7,  min_child_samples 100, reg_lambda 5 (400)
               mag  num_leaves 15, min_child_samples 100, reg_lambda 5 (500)
    arm B_clf  classifier changed only
    arm B_reg  hurdles changed only

B_clf and B_reg exist so a gain can be attributed. The four arms need only
TWO classifier fits and TWO regression fits per fold, because the classifier
and the hurdles are independent: A = clfA+regA, B_clf = clfB+regA,
B_reg = clfA+regB, B = clfB+regB. Each distinct fit is computed once.

PRE-REGISTERED, written before any result was seen:

  Keep rule, unchanged from the project standard: mean score gain >= +0.0015
  AND better on both folds. Anything passing at 5 seeds is confirmed at 20
  (42-61) under the same rule. Alpha and the Fuel x1.25 factor are NOT
  retuned -- alpha 0.75 and 0.5 exactly as they ship.

    python src/sweep11.py --block reg
    python src/sweep11.py --block reg --confirm
"""
import argparse, json, time
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, log_loss
import make_submission as M
import sweep2 as W
import sweep5 as S5
import sweep8 as S8
import validate as V

RESULTS = Path('preds/sweep11.json')
NL = chr(10)
TARGETS = ('CLV_fuel', 'CLV_nonfuel')
RENEW_BLOCKS = ('r1', 'r2', 'r3')
ALPHAS = (0.75, 0.5)
KEEP_BAR = 0.0015

# Arm B, exactly as pre-registered. Arm A is whatever make_submission defaults
# to, so the incumbent is read from the shipping code rather than restated.
B_CLF = dict(num_leaves=7, min_child_samples=100, reg_lambda=5)
B_GATE = dict(num_leaves=7, min_child_samples=100, reg_lambda=5)
B_MAG = dict(num_leaves=15, min_child_samples=100, reg_lambda=5)


def arm_params(which):
    """which in {'A','B'} for each of the two independent halves."""
    p = M.params_for()
    if which == 'B':
        p['clf'] = {**p['clf'], **B_CLF}
        p['gate'] = {**p['gate'], **B_GATE}
        p['mag'] = {**p['mag'], **B_MAG}
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
    """rmse_fuel and rmse_nonfuel for one fold, one hurdle config.

    Feature sets are the incumbent's: CLV_fuel gets base + r1+r2+r3 renewal,
    CLV_nonfuel stays on base. Nothing about the features changes between
    arms -- only the LightGBM settings.
    """
    tr = M.trainable_for(ctx.monthly, v)
    out = {}
    for t in TARGETS:
        extra = renew if t == 'CLV_fuel' else None
        Xtr, ytr = M.stack(ctx.snaps, tr, extra)
        Xva = M.feats(ctx.snaps, extra, v)
        pred = M.hurdle(Xtr, ytr[t], Xva, list(seeds), params=params)
        out[t] = rmse(pred, ctx.snaps[v][1][t])
    return out


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


def track_reg(ctx, args):
    seeds = tuple(S5.CONFIRM if args.confirm else S5.SCREEN)
    tag = 'confirm' if args.confirm else 'screen'
    cuts = sorted(set(ctx.monthly + list(V.FOLDS) + [W.TEST_CUTOFF]))
    print(NL + f'--- sweep 11, {len(seeds)} seeds, incumbent v5_fuel2 pipeline ---')
    lags = S5.lag_extra(ctx, cuts)
    # renewal_extra already joins the blocks and returns {cutoff: frame}
    renew = S8.renewal_extra(ctx, cuts, list(RENEW_BLOCKS))

    # --- the four distinct fits per fold -------------------------------------
    clf, reg, truth = {}, {}, {}
    for v in V.FOLDS:
        for which in ('A', 'B'):
            pr = arm_params(which)
            t0 = time.monotonic()
            proba, ll, y = clf_half(ctx, v, lags, seeds, pr)
            clf[(v, which)] = dict(proba=proba, logloss=ll)
            truth[v] = y
            print(f'  clf {which} {v}: logloss {ll:.4f}  '
                  f'[{time.monotonic() - t0:.0f}s]', flush=True)
            t0 = time.monotonic()
            reg[(v, which)] = reg_half(ctx, v, renew, seeds, pr)
            print(f'  reg {which} {v}: fuel {reg[(v, which)]["CLV_fuel"]:.4f} '
                  f'nonfuel {reg[(v, which)]["CLV_nonfuel"]:.4f}  '
                  f'[{time.monotonic() - t0:.0f}s]', flush=True)

    # --- compose the four arms ----------------------------------------------
    ARMS = {'A (incumbent)': ('A', 'A'), 'B_clf': ('B', 'A'),
            'B_reg': ('A', 'B'), 'B (both)': ('B', 'B')}
    rows = {}
    for name, (c_half, r_half) in ARMS.items():
        d = {'clf_half': c_half, 'reg_half': r_half, 'folds': {}}
        for v in V.FOLDS:
            cc, rr = clf[(v, c_half)], reg[(v, r_half)]
            e = dict(logloss=cc['logloss'], rmse_fuel=rr['CLV_fuel'],
                     rmse_nonfuel=rr['CLV_nonfuel'])
            for al in ALPHAS:
                pred = decide(ctx, cc['proba'], v, al)
                e[f'f1_a{al}'] = float(f1_score(truth[v], pred, average='weighted',
                                                zero_division=0))
                e[f'score_a{al}'] = M.combined(e[f'f1_a{al}'], e['rmse_fuel'],
                                               e['rmse_nonfuel'])
            d['folds'][v] = e
        rows[name] = d

    f0, f1_ = V.FOLDS
    for al in ALPHAS:
        print(NL + f'  --- alpha {al} ---')
        print(f'  {"arm":<16}{"F1 06":>9}{"F1 09":>9}{"rmseF 06":>10}{"rmseF 09":>10}'
              f'{"rmseN 06":>10}{"rmseN 09":>10}{"logl 06":>9}{"logl 09":>9}'
              f'{"score 06":>10}{"score 09":>10}{"mean":>9}')
        for name, d in rows.items():
            a, b = d['folds'][f0], d['folds'][f1_]
            mean = np.mean([a[f'score_a{al}'], b[f'score_a{al}']])
            print(f'  {name:<16}{a[f"f1_a{al}"]:>9.4f}{b[f"f1_a{al}"]:>9.4f}'
                  f'{a["rmse_fuel"]:>10.4f}{b["rmse_fuel"]:>10.4f}'
                  f'{a["rmse_nonfuel"]:>10.4f}{b["rmse_nonfuel"]:>10.4f}'
                  f'{a["logloss"]:>9.4f}{b["logloss"]:>9.4f}'
                  f'{a[f"score_a{al}"]:>10.5f}{b[f"score_a{al}"]:>10.5f}{mean:>9.5f}')

    # --- the pre-registered rule --------------------------------------------
    print(NL + f'  --- keep rule: mean score gain >= +{KEEP_BAR} AND better on both folds ---')
    verdict = {}
    for al in ALPHAS:
        ref = rows['A (incumbent)']['folds']
        for name, d in rows.items():
            if name.startswith('A ('):
                continue
            dl = [d['folds'][v][f'score_a{al}'] - ref[v][f'score_a{al}'] for v in V.FOLDS]
            mg = float(np.mean(dl))
            ok = bool(mg >= KEEP_BAR and all(x > 0 for x in dl))
            verdict[f'{name} @ alpha {al}'] = dict(deltas=dl, mean_gain=mg, passes=ok)
            print(f'  {name + " @ alpha " + str(al):<28}'
                  f'{dl[0]:>+10.5f}{dl[1]:>+10.5f}   mean {mg:>+9.5f}   '
                  f'-> {"PASS" if ok else "fail"}')
    passing = [k for k, x in verdict.items() if x['passes']]
    print(NL + (f'  -> PASSES at {len(seeds)} seeds: {passing}' if passing else
                f'  -> nothing passes at {len(seeds)} seeds.'))
    if passing and not args.confirm:
        print('     (5-seed screen; a 20-seed confirmation is required before adopting)')

    out = {'seeds': list(seeds), 'rows': {k: v['folds'] for k, v in rows.items()},
           'verdict': verdict, 'passing': passing}
    save(tag, out)
    return out


TRACKS = {'reg': track_reg}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--block', nargs='+', default=['reg'])
    ap.add_argument('--train', default='data/train.csv')
    ap.add_argument('--confirm', action='store_true')
    a = ap.parse_args()
    ctx = W.Ctx(a.train)
    for b in a.block:
        print(NL + f'########## block {b} ##########')
        t0 = time.monotonic()
        TRACKS[b](ctx, a)
        print(f'[block {b} took {time.monotonic() - t0:.0f}s]')


if __name__ == '__main__':
    main()
