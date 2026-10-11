"""Sweep 9: renewal features for the CLV_fuel hurdle only.

Why only fuel, and why only renewal. The r1 renewal block is the one change in
this project that transferred LARGER than validation predicted -- fuel RMSE
-0.0012 and -0.0014 on the two folds, -0.0049 on the public board -- while the
same block applied to CLV_nonfuel reversed on public (-0.0022, -0.0001, +0.0015).
That asymmetry has a mechanism: r1 is built from fuel fill rhythm (gap
statistics, phase, projected fill counts and litres), so it has a causal story
for fuel volume and none for non-fuel basket spend. This sweep follows the one
direction with demonstrated out-of-sample transfer and leaves everything else
alone.

Nothing here touches the classifier or the non-fuel hurdle, so any submission
built from a result keeps its Opportunity and CLV_nonfuel columns byte-identical
to the selected pick. The only column at risk is CLV_fuel.

PRE-REGISTERED, written before any result was seen:

    A variant is adopted only if, at 20 seeds, fuel RMSE improves against the
    r1 incumbent on BOTH validation folds. If one is adopted and submitted, it
    replaces the selected pick only if its public CLV Fuel RMSE also improves
    on 0.588231347.

Screen at 5 seeds, confirm at 20.

    python src/sweep9.py --block fuel
    python src/sweep9.py --block fuel --confirm
"""
import argparse, json, time
from pathlib import Path
import numpy as np
import make_submission as M
import sweep2 as W
import sweep5 as S5
import sweep8 as S8
import validate as V

RESULTS = Path('preds/sweep9.json')
NL = chr(10)
TARGET = 'CLV_fuel'
# the 20-seed references, both already measured
BASE_20 = [0.5969, 0.5996]   # v3 regressions, base features only
R1_20 = [0.5957, 0.5982]     # the incumbent, shipped in submission_v5_fuel.csv
# the 5-seed base, from sweep 7's cache, for judging the screen
BASE_5 = [0.5975, 0.5994]

VARIANTS = [
    ('base (no renewal)', []),
    ('r1 fuel renewal [incumbent]', ['r1']),
    ('r2 all-basket renewal', ['r2']),
    ('r3 calendar phase', ['r3']),
    ('r1 + r2', ['r1', 'r2']),
    ('r1 + r3', ['r1', 'r3']),
    ('r1 + r2 + r3', ['r1', 'r2', 'r3']),
]


def rmse(pred, truth):
    return float(np.sqrt(((np.asarray(pred) - np.asarray(truth)) ** 2).mean()))


def fuel_rmse(ctx, extra, seeds):
    """Two-fold RMSE for the CLV_fuel hurdle alone. Half the work of a full
    regression run, because CLV_nonfuel is never fitted."""
    out = []
    for v in V.FOLDS:
        D = ctx.fold(v, extra=extra)
        y = D['ytr'][TARGET]
        pos = y.to_numpy() > 0
        gate = W.bag_binary(D['Xtr'], pos.astype(int), D['Xva'], seeds=list(seeds))
        mag = W.bag_reg(D['Xtr'][pos], y[pos], D['Xva'], seeds=list(seeds))
        out.append(rmse(np.clip(gate * mag, 0, None), D['yva'][TARGET]))
    return out


def save(key, obj):
    RESULTS.parent.mkdir(exist_ok=True)
    allr = json.loads(RESULTS.read_text()) if RESULTS.exists() else {}
    allr[key] = obj
    RESULTS.write_text(json.dumps(allr, indent=2, default=float))


def track_fuel(ctx, args):
    seeds = tuple(S5.CONFIRM if args.confirm else S5.SCREEN)
    tag = 'confirm' if args.confirm else 'screen'
    ref = R1_20 if args.confirm else None     # the screen judges against its own r1 row
    cuts = sorted(set(ctx.monthly + list(V.FOLDS) + [W.TEST_CUTOFF]))
    picked = VARIANTS
    if args.confirm and args.only:
        picked = [(n, b) for n, b in VARIANTS if '+'.join(b) in args.only or n in args.only]
    print(NL + f'--- CLV_fuel hurdle, renewal blocks, {len(seeds)} seeds ---')
    print(f'  20-seed references: base {BASE_20}, r1 incumbent {R1_20}')
    print(NL + f'  {"variant":<30}{"cols":>6}{"fold 2025-06":>14}{"fold 2025-09":>14}'
          f'{"mean":>9}')
    rows = {}
    for name, blocks in picked:
        extra = S8.renewal_extra(ctx, cuts, blocks) if blocks else None
        ncol = extra[V.FOLDS[0]].shape[1] if extra is not None else 0
        t0 = time.time()
        r = fuel_rmse(ctx, extra, seeds)
        rows[name] = dict(blocks=blocks, rmse=r, ncols=int(ncol),
                          secs=round(time.time() - t0))
        print(f'  {name:<30}{ncol:>6}{r[0]:>14.4f}{r[1]:>14.4f}{np.mean(r):>9.4f}'
              f'   [{time.time() - t0:.0f}s]', flush=True)

    inc = rows.get('r1 fuel renewal [incumbent]', {}).get('rmse') or ref or R1_20
    print(NL + '  against the r1 incumbent (lower is better on BOTH folds):')
    for name, d in rows.items():
        if name.startswith('r1 fuel renewal'):
            continue
        r = d['rmse']
        dl = [r[i] - inc[i] for i in (0, 1)]
        both = all(x < 0 for x in dl)
        d['delta'] = dl
        d['both_better'] = bool(both)
        print(f'  {name:<30}{dl[0]:>+14.4f}{dl[1]:>+14.4f}'
              f'   both folds better: {"YES" if both else "no"}')

    print(NL + '  pre-registered rule: adopt only if fuel RMSE improves on BOTH folds')
    print('  at 20 seeds against the r1 incumbent.')
    winners = [n for n, d in rows.items() if d.get('both_better')]
    if not winners:
        print('  -> nothing improves on both folds. r1 stands.')
    else:
        print(f'  -> improves on both folds: {winners}')
        if not args.confirm:
            print('     (5-seed screen; a 20-seed confirmation is required before adopting)')
    save(tag, rows)
    return rows


TRACKS = {'fuel': track_fuel}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--block', nargs='+', default=['fuel'])
    ap.add_argument('--train', default='data/train.csv')
    ap.add_argument('--confirm', action='store_true')
    ap.add_argument('--only', nargs='*', default=None,
                    help='with --confirm, restrict to these variants')
    a = ap.parse_args()
    ctx = W.Ctx(a.train)
    for b in a.block:
        print(NL + f'########## block {b} ##########')
        t0 = time.time()
        TRACKS[b](ctx, a)
        print(f'[block {b} took {time.time() - t0:.0f}s]')


if __name__ == '__main__':
    main()
