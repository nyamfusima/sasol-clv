"""Sweep 14: a binary Fuel-growth-vs-Stable specialist, blended into the
17-class probabilities. Classifier only; the hurdles never move.

WHY THIS IS NOT SWEEP 10
------------------------
Sweep 10 tested decision-side GATES on this boundary: multiply Fuel's score by
a constant m and re-decide. It found the off-diagonal large but NEARLY
SYMMETRIC (340/327, 387/283, 307/344, 354/291), so moving the boundary trades
one error for the other at close to 1:1 and every oracle came in at a fifth of
the bar. That closes RE-THRESHOLDING. It does not close better
DISCRIMINATION: a specialist that genuinely ranks the pair better shrinks both
off-diagonal cells instead of trading them, and a threshold oracle cannot
bound a ranking improvement. So the ranking ceiling on this pair is a number
the record does not contain, and Gate 0 below computes it.

The honest tension, recorded before running: sweep 5 L4 found one-vs-rest
binaries estimate the group marginals WORSE than the single 17-class softmax
(-0.00289), and blocks C/E concluded the two classes "genuinely overlap in
these features". A specialist is an OvR-shaped model, so L4 is evidence
against. Set against that, a binary gets the full 400-tree budget on one
decision instead of sharing gradient 17 ways with 11 classes starved, which is
a real mechanism and is why this is worth one cheap test.

The shipped v6_clf validation probabilities were never cached -- sweep 11 and
12 saved metrics only, and sweep 10 ran on the older lags20/r320 classifiers.
This script regenerates them against the real shipped baseline and caches them
to preds/proba_v6clf{n}_{fold}.parquet.

PRE-REGISTERED, written before any sweep-14 number existed. Alpha is NOT
retuned: 0.75 and 0.5 exactly as they ship. Specialist params are the shipped
V6_CLF, NOT tuned, so the blend weight w is the only free parameter.

  Gate 0, the ranking oracle, and sweep 10's protocol applied here: fix the
  pair conditional to the TRUTH for every row whose true label is Fuel or
  Stable, leave the other 15 classes untouched, run the shipped rule. If the
  mean gain is under +0.005 weighted F1 the direction is CLOSED whatever the
  blend says -- the test that closed block C and sweep 10.

  Gate 1, does the specialist rank better at all: on rows whose true label is
  in the pair, AUC of the specialist's P(Fuel) against the 17-class implied
  conditional p_fuel/(p_fuel+p_stable). Must exceed it on BOTH folds. A blend
  cannot add ranking information that is not there.

  Gate 2, the blend. Mass-preserving, one parameter, baseline at w=0:
      s = p_fuel + p_stable                      (held fixed)
      q = (1-w)*p_fuel/s + w*q_specialist
      p_fuel <- s*q ; p_stable <- s*(1-q)        (other 15 untouched)
  Predeclared grid w in {0.15, 0.30, 0.50}.

  Keep rule, all required:
    1. Gate 0 mean oracle gain >= +0.005 weighted F1
    2. Gate 1 specialist AUC > 17-class AUC on BOTH folds
    3. both folds independently choose the same w
    4. cross-fold (w chosen on 2025-06, scored on 2025-09) clears +0.0015
       composite on BOTH folds at alpha 0.75

  Conditions 1 and 3 are sweep 10's protocol verbatim; condition 4 is the
  project's standing bar. Criteria are not revised after the numbers are seen.

    python src/sweep14.py --screen --seeds 5
"""
import argparse, json, time
from pathlib import Path
import numpy as np, pandas as pd
import lightgbm as lgb
from sklearn.metrics import f1_score, log_loss, roc_auc_score
import make_submission as M
import sweep2 as W
import sweep5 as S5
import validate as V

RESULTS = Path('preds/sweep14.json')
SWEEP11 = Path('preds/sweep11.json')
NL = chr(10)
ALPHAS = (0.75, 0.5)
KEEP_BAR = 0.0015
ORACLE_BAR = 0.005
A_RECIPE = 'v6_clf'
FUEL = M.FUEL_LABEL                      # 'Existing-category growth: Fuel'
STABLE = 'Stable'
W_GRID = (0.15, 0.30, 0.50)


def clf_params():
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


def fold_data(ctx, v, lags):
    """Training rows are only snapshots whose outcome window ends on or before
    the fold cutoff -- the same leakage rule the 17-class model uses."""
    tr = M.trainable_for(ctx.monthly, v)
    Xtr, ytr = M.stack(ctx.snaps, tr, lags)
    Xva = M.feats(ctx.snaps, lags, v)
    yva = ctx.snaps[v][1].Opportunity
    return Xtr, ytr.Opportunity, Xva, yva, tr


def seventeen(Xtr, ylab, Xva, labels, seeds, params, cache=None):
    """The shipped 17-class bag. Cached to parquet, since nothing else has it."""
    if cache is not None and cache.exists():
        p = pd.read_parquet(cache)
        if list(p.columns) == list(labels) and len(p) == len(Xva):
            print(f'    17-class: reused {cache}')
            return p
    code = {l: i for i, l in enumerate(labels)}
    pr = M.bag_clf(Xtr, ylab.map(code).to_numpy(), Xva, labels, list(seeds), params)
    if cache is not None:
        pr.to_parquet(cache)
        print(f'    17-class: cached -> {cache}')
    return pr


def specialist(Xtr, ylab, Xva, seeds, params):
    """P(Fuel | label in {Fuel, Stable}), trained on pair rows only.

    Never sees validation labels; trains on the same leakage-safe snapshots.
    """
    m = ylab.isin([FUEL, STABLE]).to_numpy()
    yb = (ylab[m] == FUEL).astype(int).to_numpy()
    out = np.zeros(len(Xva))
    for s in seeds:
        g = lgb.LGBMClassifier(random_state=s, verbose=-1, **params).fit(Xtr[m], yb)
        out += g.predict_proba(Xva)[:, list(g.classes_).index(1)]
    return out / len(seeds), int(m.sum()), float(yb.mean())


def blend(proba, labels, qsp, w):
    """Mass-preserving swap of the pair conditional. w=0 is the baseline."""
    P = proba.to_numpy().copy()
    fi, si = labels.index(FUEL), labels.index(STABLE)
    s = P[:, fi] + P[:, si]
    q17 = np.divide(P[:, fi], s, out=np.full(len(P), 0.5), where=s > 0)
    q = (1 - w) * q17 + w * qsp
    P[:, fi], P[:, si] = s * q, s * (1 - q)
    return pd.DataFrame(P, index=proba.index, columns=list(labels))


def oracle_q(proba, labels, yva):
    """Gate 0: the pair conditional set to the truth on pair rows only."""
    y = yva.to_numpy()
    P = proba.to_numpy()
    fi, si = labels.index(FUEL), labels.index(STABLE)
    s = P[:, fi] + P[:, si]
    q = np.divide(P[:, fi], s, out=np.full(len(P), 0.5), where=s > 0)
    q = np.where(y == FUEL, 1.0, np.where(y == STABLE, 0.0, q))
    return q


def decide(ctx, proba, v, alpha):
    """The shipped rule, unchanged. Alpha is not retuned."""
    P = proba.to_numpy()
    tgt, _ = M.target_mix(ctx.snaps, ctx.monthly, v, ctx.labels)
    w, _ = M.ipf(P, tgt, ctx.labels, [ctx.labels.index(l) for l in M.BIG4])
    return np.asarray(ctx.labels)[(P * w ** alpha).argmax(axis=1)]


def score_proba(ctx, proba, v, yva, reg):
    rf, rn = reg[v]
    y = yva.to_numpy()
    e = {}
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


def track(ctx, args):
    seeds = tuple(range(42, 42 + args.seeds)) if args.seeds else tuple(S5.SCREEN)
    params = clf_params()
    reg = cached_reg()
    cuts = sorted(set(ctx.monthly + list(V.FOLDS) + [W.TEST_CUTOFF]))
    lags = S5.lag_extra(ctx, cuts)
    labels = ctx.labels
    print(NL + f'--- sweep 14 screen, {len(seeds)} seeds {min(seeds)}-{max(seeds)} ---')
    print(f'    params (shared by both models, NOT tuned): {params}')
    print(f'    blend grid w: {W_GRID}')

    rows, gate1, gate0 = {}, {}, {}
    for v in V.FOLDS:
        t0 = time.monotonic()
        Xtr, ylab, Xva, yva, tr = fold_data(ctx, v, lags)
        cache = Path(f'preds/proba_v6clf{len(seeds)}_{v}.parquet')
        p17 = seventeen(Xtr, ylab, Xva, labels, seeds, params, cache)
        t17 = time.monotonic() - t0
        t0 = time.monotonic()
        qsp, npair, posrate = specialist(Xtr, ylab, Xva, seeds, params)
        tsp = time.monotonic() - t0
        print(f'  fold {v}: {len(Xtr)} train rows / {len(tr)} snapshots, '
              f'{len(Xva)} val rows')
        print(f'    17-class [{t17:.0f}s] | specialist on {npair} pair rows '
              f'({100 * npair / len(Xtr):.1f}%, {100 * posrate:.1f}% Fuel) [{tsp:.0f}s]')

        # --- Gate 1: ranking, on rows whose TRUE label is in the pair -------
        y = yva.to_numpy()
        m = np.isin(y, [FUEL, STABLE])
        yb = (y[m] == FUEL).astype(int)
        P = p17.to_numpy()
        fi, si = labels.index(FUEL), labels.index(STABLE)
        s = P[:, fi] + P[:, si]
        q17 = np.divide(P[:, fi], s, out=np.full(len(P), 0.5), where=s > 0)
        a17 = float(roc_auc_score(yb, q17[m]))
        asp = float(roc_auc_score(yb, qsp[m]))
        corr = float(np.corrcoef(q17[m], qsp[m])[0, 1])
        gate1[v] = dict(n_pair=int(m.sum()), auc_17class=a17, auc_specialist=asp,
                        delta=asp - a17, corr=corr)
        print(f'    Gate 1 AUC on {m.sum()} true-pair rows: 17-class {a17:.4f} | '
              f'specialist {asp:.4f} | delta {asp - a17:+.4f} | corr {corr:.4f}')

        # --- Gate 0: the ranking oracle -------------------------------------
        base = score_proba(ctx, p17, v, yva, reg)
        orc = score_proba(ctx, blend(p17, labels, oracle_q(p17, labels, yva), 1.0),
                          v, yva, reg)
        gate0[v] = {'baseline': base, 'oracle': orc,
                    'f1_gain_a0.75': orc['f1_a0.75'] - base['f1_a0.75'],
                    'f1_gain_a0.5': orc['f1_a0.5'] - base['f1_a0.5']}
        print(f'    Gate 0 oracle: F1 a0.75 {base["f1_a0.75"]:.4f} -> '
              f'{orc["f1_a0.75"]:.4f} ({orc["f1_a0.75"] - base["f1_a0.75"]:+.4f})  '
              f'a0.5 {base["f1_a0.5"]:.4f} -> {orc["f1_a0.5"]:.4f} '
              f'({orc["f1_a0.5"] - base["f1_a0.5"]:+.4f})')

        # --- Gate 2: the predeclared blend grid ------------------------------
        d = {'0.0': base}
        for wt in W_GRID:
            d[str(wt)] = score_proba(ctx, blend(p17, labels, qsp, wt), v, yva, reg)
        rows[v] = d

    # --- tables -------------------------------------------------------------
    f0, f1_ = V.FOLDS
    for al in ALPHAS:
        print(NL + f'  --- alpha {al} ---')
        print(f'  {"w":>6}{"F1 06":>9}{"F1 09":>9}{"score 06":>11}{"score 09":>11}'
              f'{"mean":>11}{"d mean":>10}')
        b = np.mean([rows[f0]['0.0'][f'score_a{al}'], rows[f1_]['0.0'][f'score_a{al}']])
        for k in ['0.0'] + [str(x) for x in W_GRID]:
            a, c = rows[f0][k], rows[f1_][k]
            mn = np.mean([a[f'score_a{al}'], c[f'score_a{al}']])
            tag = '  <- baseline' if k == '0.0' else ''
            print(f'  {k:>6}{a[f"f1_a{al}"]:>9.4f}{c[f"f1_a{al}"]:>9.4f}'
                  f'{a[f"score_a{al}"]:>11.5f}{c[f"score_a{al}"]:>11.5f}'
                  f'{mn:>11.5f}{mn - b:>+10.5f}{tag}')

    # --- the pre-registered rule --------------------------------------------
    print(NL + '  --- keep rule ---')
    om = float(np.mean([gate0[v]['f1_gain_a0.75'] for v in V.FOLDS]))
    c1 = om >= ORACLE_BAR
    c2 = all(gate1[v]['delta'] > 0 for v in V.FOLDS)
    best = {v: max(W_GRID, key=lambda x: rows[v][str(x)]['score_a0.75']) for v in V.FOLDS}
    c3 = best[f0] == best[f1_]
    wsel = best[f0]
    xf = [rows[v][str(wsel)]['score_a0.75'] - rows[v]['0.0']['score_a0.75']
          for v in V.FOLDS]
    c4 = all(x >= KEEP_BAR for x in xf)
    print(f'  1. Gate 0 mean oracle F1 gain >= +{ORACLE_BAR} : '
          f'{om:+.5f} -> {"PASS" if c1 else "FAIL"}')
    print(f'  2. Gate 1 specialist AUC > 17-class, both folds : '
          f'{gate1[f0]["delta"]:+.4f} / {gate1[f1_]["delta"]:+.4f} '
          f'-> {"PASS" if c2 else "FAIL"}')
    print(f'  3. both folds choose the same w                 : '
          f'{best[f0]} / {best[f1_]} -> {"PASS" if c3 else "FAIL"}')
    print(f'  4. cross-fold w={wsel} clears +{KEEP_BAR} both folds : '
          f'{xf[0]:+.5f} / {xf[1]:+.5f} -> {"PASS" if c4 else "FAIL"}')
    passes = bool(c1 and c2 and c3 and c4)
    print(NL + f'  ==> SWEEP 14 {"PASSES" if passes else "FAILS"}')
    out = dict(seeds=list(seeds), params=params, w_grid=list(W_GRID), rows=rows,
               gate0=gate0, gate1=gate1,
               conditions=dict(oracle=c1, auc=c2, same_w=c3, cross_fold=c4),
               oracle_mean=om, w_selected=float(wsel), cross_fold=xf, passes=passes)
    save(f'screen_seeds{len(seeds)}', out)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--train', default='data/train.csv')
    ap.add_argument('--screen', action='store_true')
    ap.add_argument('--seeds', type=int, default=5)
    a = ap.parse_args()
    if not a.screen:
        ap.error('pass --screen')
    ctx = W.Ctx(a.train)
    t0 = time.monotonic()
    track(ctx, a)
    print(f'[sweep 14 took {time.monotonic() - t0:.0f}s]')


if __name__ == '__main__':
    main()
