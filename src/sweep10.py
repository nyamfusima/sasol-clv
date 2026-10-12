"""Sweep 10: the Fuel-growth / Stable decision boundary, as a targeted gate.

Context. Blocks C and E closed the "change what gets predicted" route at the
feature level and left one sentence standing: the race is Fuel growth vs Stable
discrimination, which is also exactly where the confusion matrix has always been
worst (432 of 1435 true Fuel growth predicted Stable, 362 of 1404 true Stable
predicted Fuel growth). Sweep 10 asks the one question left in that direction
that costs nothing: is the Fuel/Stable boundary itself mis-placed, separately
from the global class mix that prior matching already moves?

Two gates, both one parameter, both applied AFTER the usual alpha-damped IPF
weights, so alpha stays exactly what it is in the selected picks:

  g1 targeted   rows whose decision is already Fuel or Stable are re-decided
                between those two alone, with Fuel's score multiplied by m.
                No other class can lose a row. m = 1 is the baseline.
  g2 global     Fuel's score multiplied by m on every row, then argmax. This is
                v4_simple's Fuel x1.25 family stacked on IPF, and it is here
                only to isolate what "targeted" buys over "global".

PRE-REGISTERED, written before any result was seen:

  0. ORACLE FIRST. Per fold, with full sight of that fold's labels, take the
     best m on the grid. That is the ceiling no realisable version of this gate
     can beat. If the mean oracle gain over the shipped baseline is under
     +0.005 weighted F1, the direction is CLOSED and nothing here is adopted
     whatever the cross-fold numbers look like -- the same test that closed
     block C.
  1. Both folds' independent searches must choose the same m, as alpha's did.
  2. The honest reading is cross-fold: m chosen on one fold, scored on the
     other. It must improve the shipped baseline by >= 0.005 weighted F1 on
     BOTH held-out folds.

Any one of the three failing drops the gate. No refits: this runs on the cached
20-seed validation probabilities, so it is the cheapest open question left.

    python src/sweep10.py --block gate
    python src/sweep10.py --block gate --model r320
"""
import argparse, json, time
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
import make_submission as M
import sweep2 as W
import sweep5 as S5
import validate as V

RESULTS = Path('preds/sweep10.json')
NL = chr(10)
FUEL = 'Existing-category growth: Fuel'
STABLE = 'Stable'
ALPHAS = (0.0, 0.5, 0.75, 1.0)
# multiplier grid on Fuel's score; 1.0 is the baseline and must be on the grid
GRID = tuple(sorted(set([round(float(x), 3) for x in
                         np.exp(np.linspace(np.log(0.60), np.log(1.80), 25))] + [1.0])))
KEEP_BAR = 0.005


def f1w(y, p):
    return float(f1_score(y, p, average='weighted', zero_division=0))


def load_proba(ctx, model):
    """Cached 20-seed validation probabilities, aligned to each fold's labels."""
    P, truth = {}, {}
    for v in V.FOLDS:
        y = ctx.snaps[v][1]
        f = Path(f'preds/proba_{model}_{v}.parquet')
        if not f.exists():
            f = Path(f'preds/oof_{model}_{v}.parquet')
        P[v] = pd.read_parquet(f).set_index('ID').reindex(y.index)[ctx.labels].to_numpy()
        truth[v] = y.Opportunity.to_numpy()
    return P, truth


def weighted(ctx, P, v, alpha):
    """P * w**alpha, the scores the selected picks take an argmax over."""
    if alpha == 0:
        return P[v].copy()
    tgt, _ = M.target_mix(ctx.snaps, ctx.monthly, v, ctx.labels)
    w, _ = M.ipf(P[v], tgt, ctx.labels, [ctx.labels.index(l) for l in S5.BIG4])
    return P[v] * w ** alpha


def gate(S, fi, si, m, kind):
    """Decision indices under the gate. S is the alpha-weighted score matrix."""
    j = S.argmax(axis=1)
    if kind == 'global':
        T = S.copy()
        T[:, fi] = T[:, fi] * m
        return T.argmax(axis=1)
    contested = (j == fi) | (j == si)
    out = j.copy()
    pick = np.where(S[:, fi] * m > S[:, si], fi, si)
    out[contested] = pick[contested]
    return out


def pair_counts(truth, pred, labels):
    """The four cells of the Fuel/Stable confusion, plus predicted shares."""
    p = np.asarray(labels)[pred]
    t = np.asarray(truth)
    return dict(fuel_fuel=int(((t == FUEL) & (p == FUEL)).sum()),
                fuel_stable=int(((t == FUEL) & (p == STABLE)).sum()),
                stable_fuel=int(((t == STABLE) & (p == FUEL)).sum()),
                stable_stable=int(((t == STABLE) & (p == STABLE)).sum()),
                pred_fuel_share=float((p == FUEL).mean()),
                pred_stable_share=float((p == STABLE).mean()),
                true_fuel_share=float((t == FUEL).mean()),
                true_stable_share=float((t == STABLE).mean()))


def save(key, obj):
    RESULTS.parent.mkdir(exist_ok=True)
    allr = json.loads(RESULTS.read_text()) if RESULTS.exists() else {}
    allr[key] = obj
    RESULTS.write_text(json.dumps(allr, indent=2, default=float))


def track_gate(ctx, args):
    labels = ctx.labels
    fi, si = labels.index(FUEL), labels.index(STABLE)
    f0, f1_ = V.FOLDS
    P, truth = load_proba(ctx, args.model)
    print(NL + f'--- references: {args.model}, cached 20-seed probabilities, no refits ---')
    print(f'  {"rule":<26}{"fold 2025-06":>14}{"fold 2025-09":>14}{"mean":>9}')
    Sw, refs = {}, {}
    for al in ALPHAS:
        name = 'argmax' if al == 0 else f'alpha {al}'
        Sw[al] = {v: weighted(ctx, P, v, al) for v in V.FOLDS}
        fs = [f1w(truth[v], np.asarray(labels)[Sw[al][v].argmax(axis=1)]) for v in V.FOLDS]
        refs[name] = fs
        print(f'  {name:<26}{fs[0]:>14.4f}{fs[1]:>14.4f}{np.mean(fs):>9.4f}')

    # What a gate would actually replace: per fold, the better of the two
    # shipped alphas. That is the hardest honest baseline, because the two
    # selected picks already span alpha.
    best_ref = max(refs, key=lambda k: float(np.mean(refs[k])))
    shipped = {v: max(refs['alpha 0.5'][i], refs['alpha 0.75'][i])
               for i, v in enumerate(V.FOLDS)}
    print(NL + f'  best mean reference: {best_ref} ({np.mean(refs[best_ref]):.4f})')
    print(f'  per-fold best of the two shipped alphas: '
          f'{shipped[f0]:.4f} / {shipped[f1_]:.4f}')

    out = {'model': args.model, 'refs': refs, 'shipped_best': shipped,
           'grid': list(GRID), 'curves': {}}
    for kind in ('targeted', 'global'):
        tag = 'g1' if kind == 'targeted' else 'g2'
        for al in (0.5, 0.75):
            key = f'{tag} {kind} on alpha {al}'
            curve = {v: [] for v in V.FOLDS}
            counts = {v: {} for v in V.FOLDS}
            for m in GRID:
                for v in V.FOLDS:
                    pred = gate(Sw[al][v], fi, si, m, kind)
                    curve[v].append(f1w(truth[v], np.asarray(labels)[pred]))
                    counts[v][m] = pair_counts(truth[v], pred, labels)
            print(NL + f'  --- {key} ---')
            print(f'  {"m":>7}{"fold 2025-06":>14}{"fold 2025-09":>14}{"mean":>9}')
            for i, m in enumerate(GRID):
                a, b = curve[f0][i], curve[f1_][i]
                star = '  <- m=1, baseline' if abs(m - 1.0) < 1e-9 else ''
                print(f'  {m:>7.3f}{a:>14.4f}{b:>14.4f}{np.mean([a, b]):>9.4f}{star}')
            arg = {v: GRID[int(np.argmax(curve[v]))] for v in V.FOLDS}
            orc = {v: float(np.max(curve[v])) for v in V.FOLDS}
            cross = {}
            for i, v in enumerate(V.FOLDS):
                cross[v] = curve[v][GRID.index(arg[V.FOLDS[1 - i]])]
            out['curves'][key] = dict(curve=curve, argbest=arg, oracle=orc, cross=cross,
                                      counts={v: counts[v][arg[v]] for v in V.FOLDS})
            agree = 'YES' if arg[f0] == arg[f1_] else 'no'
            print(f'  each fold\'s own best m: {arg[f0]} and {arg[f1_]}   agree: {agree}')
            print(f'  oracle F1 (ceiling):    {orc[f0]:.4f} / {orc[f1_]:.4f}   '
                  f'over shipped best: {orc[f0] - shipped[f0]:+.4f} / '
                  f'{orc[f1_] - shipped[f1_]:+.4f}')
            print(f'  cross-fold F1 (honest): {cross[f0]:.4f} / {cross[f1_]:.4f}   '
                  f'over shipped best: {cross[f0] - shipped[f0]:+.4f} / '
                  f'{cross[f1_] - shipped[f1_]:+.4f}')

    print(NL + '  --- pre-registered verdict ---')
    verdict = {}
    for key, d in out['curves'].items():
        orc_gain = float(np.mean([d['oracle'][v] - shipped[v] for v in V.FOLDS]))
        agree = d['argbest'][f0] == d['argbest'][f1_]
        cross_ok = all(d['cross'][v] - shipped[v] >= KEEP_BAR for v in V.FOLDS)
        ok = bool(orc_gain >= KEEP_BAR and agree and cross_ok)
        verdict[key] = dict(oracle_gain=orc_gain, folds_agree=bool(agree),
                            cross_passes=bool(cross_ok), adopt=ok)
        print(f'  {key:<30} oracle {orc_gain:+.4f}   agree {"Y" if agree else "n"}'
              f'   cross {"Y" if cross_ok else "n"}   -> {"ADOPT" if ok else "drop"}')
    out['verdict'] = verdict
    closed = all(v['oracle_gain'] < KEEP_BAR for v in verdict.values())
    print(NL + ('  -> every oracle ceiling is under the +0.005 bar: the Fuel/Stable '
                'boundary is CLOSED.' if closed else
                '  -> an oracle ceiling clears the bar; the cross-fold rows decide.'))
    save(args.model, out)
    return out


TRACKS = {'gate': track_gate}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--block', nargs='+', default=['gate'])
    ap.add_argument('--train', default='data/train.csv')
    ap.add_argument('--model', default='lags20',
                    help='lags20 (the shipped classifier), r320, or bag20')
    a = ap.parse_args()
    ctx = W.Ctx(a.train)
    for b in a.block:
        print(NL + f'########## block {b} ##########')
        t0 = time.monotonic()
        TRACKS[b](ctx, a)
        print(f'[block {b} took {time.monotonic() - t0:.0f}s]')


if __name__ == '__main__':
    main()
