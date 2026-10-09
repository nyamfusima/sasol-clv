"""Sweep 8: renewal (fill-rhythm) features.

The sanity table runs before any model and is a gate: if the projected-minus-
previous fill count does not separate true Fuel growth among regular customers,
the premise is wrong and there is no point fitting anything.

    python src/sweep8.py --block sanity
    python src/sweep8.py --block r1 r2 r3
    python src/sweep8.py --block combo --confirm
"""
import argparse, json, time
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.metrics import f1_score
import features5 as F5
import make_submission as M
import snapshots as S
import sweep2 as W
import sweep5 as S5
import sweep8_tracks as TK
import validate as V

RESULTS = Path('preds/sweep8.json')
NL = chr(10)
FUEL = 'Existing-category growth: Fuel'
ALPHAS = (0.5, 0.75)
# 5-seed base-feature hurdle RMSEs, from sweep 7's cache. Held fixed so a
# classifier-only change moves the combined score through F1 alone.
BASE_RF, BASE_RN = [0.5975, 0.5994], [0.7352, 0.7450]


def save(key, obj):
    RESULTS.parent.mkdir(exist_ok=True)
    allr = json.loads(RESULTS.read_text()) if RESULTS.exists() else {}
    allr[key] = obj
    RESULTS.write_text(json.dumps(allr, indent=2, default=float))


def renewal_extra(ctx, cuts, blocks):
    """Cached renewal features, one parquet per block."""
    out = {}
    for b in blocks:
        p = Path(f'preds/renew_{b}.parquet')
        have = {}
        if p.exists():
            t = pd.read_parquet(p)
            for c, g in t.groupby('_cutoff', observed=True):
                have[str(c)] = g.drop(columns=['_cutoff']).set_index('ID')
        need = [c for c in cuts if c not in have]
        if need:
            d = S.load_data(ctx.train, ctx.cfg)
            for c in need:
                t0 = time.time()
                have[c] = F5.BUILDERS[b](d, c, ctx.snaps[c][0].index, ctx.cfg)
                print(f'  {b} {c}: {have[c].shape[1]} cols '
                      f'({time.time() - t0:.0f}s)', flush=True)
            (pd.concat([have[c].assign(_cutoff=c) for c in sorted(have)])
             .rename_axis('ID').reset_index().to_parquet(p, index=False))
        out[b] = {c: have[c] for c in cuts}
    merged = {}
    for c in cuts:
        f = out[blocks[0]][c]
        for b in blocks[1:]:
            f = f.join(out[b][c])
        merged[c] = f
    return merged


def join_extra(*dicts):
    cuts = set(dicts[0])
    for d in dicts[1:]:
        cuts &= set(d)
    return {c: dicts[0][c].join([d[c] for d in dicts[1:]]) for c in sorted(cuts)}


# --- the sanity gate ---------------------------------------------------------

def track_sanity(ctx, args):
    """Among customers whose fuel buying is regular enough for a projection to
    mean anything, does the projection's direction separate true Fuel growth?"""
    cuts = list(V.FOLDS)
    ext = renewal_extra(ctx, cuts, ['r1'])
    print(NL + '--- Sanity: true Fuel-growth rate by projected-minus-previous fills ---')
    print('    Population: 3-12 fuel fills in the previous quarter and gap CV < 0.5.')
    print('    No model involved; the projection is arithmetic on pre-cutoff rows.')
    rows = {}
    for v in V.FOLDS:
        y = ctx.snaps[v][1]
        f = ext[v].reindex(y.index)
        sel = (f.r1_prev_fills.between(3, 12) & (f.r1_gap_cv < 0.5)
               & f.r1_projdiff_med.notna())
        base = (y.Opportunity == FUEL)
        print(NL + f'  fold {v}: {int(sel.sum())} of {len(y)} customers qualify '
              f'({sel.mean():.1%}); Fuel-growth rate {base[sel].mean():.3f} '
              f'in that group against {base.mean():.3f} overall')
        for tag in ('med', 'm6'):
            dcol = f[f'r1_projdiff_{tag}']
            bins = pd.Series(np.where(dcol <= -1, '<= -1',
                                      np.where(dcol >= 1, '>= +1', '0')),
                             index=f.index)
            print(f'    projection from the {"median" if tag == "med" else "last-6"} '
                  f'gap:')
            print(f'      {"bucket":<9}{"n":>7}{"Fuel growth":>13}{"lift":>8}')
            g = {}
            for k in ('<= -1', '0', '>= +1'):
                m = sel & (bins == k)
                r = float(base[m].mean()) if m.sum() else float('nan')
                g[k] = (int(m.sum()), r)
                print(f'      {k:<9}{int(m.sum()):>7}{r:>13.3f}'
                      f'{r / base[sel].mean():>8.2f}')
            rows[f'{v}|{tag}'] = g
            lo, hi = g['<= -1'][1], g['>= +1'][1]
            print(f'      spread (>= +1 minus <= -1): {hi - lo:+.3f}, '
                  f'ratio {hi / lo:.2f}x' if lo else '')
    save('sanity', rows)
    print(NL + '  Read this as the gate: a monotone, repeated-across-folds spread')
    print('  means the premise holds; a flat split means stop the block.')
    return rows


def _blk(blocks, name):
    def run(ctx, args):
        return TK.clf_block(ctx, args, blocks, name, renewal_extra, join_extra,
                            save, RESULTS)
    return run


def track_combo(ctx, args):
    """Whichever individual blocks improved, together. Reads the screen results
    rather than being told which, so the combination follows the evidence."""
    allr = TK.load_results(RESULTS)
    picked = []
    for b in ('r1', 'r2', 'r3'):
        k = f'{b}_screen'
        if k not in allr:
            print(f'  {b}: not screened yet, skipping')
            continue
        g = max(allr[k]['verdicts'][str(al)]['gain'] for al in ALPHAS)
        if g > 0:
            picked.append(b)
        print(f'  {b}: best gain across alphas {g:+.5f} -> '
              f'{"in" if g > 0 else "out"}')
    if len(picked) < 2:
        print(f'  only {len(picked)} block(s) improved, so a combination of '
              f'{picked} is not a distinct variant. Nothing to run.')
        return None
    return TK.clf_block(ctx, args, picked, 'combo', renewal_extra, join_extra,
                        save, RESULTS)


def track_reg(ctx, args):
    return TK.reg_block(ctx, args, renewal_extra, save, RESULTS)


TRACKS = {'sanity': track_sanity, 'ref': _blk([], 'reference'),
          'r1': _blk(['r1'], 'r1'), 'r2': _blk(['r2'], 'r2'),
          'r3': _blk(['r3'], 'r3'), 'r3a': _blk(['r3a'], 'r3a'),
          'r3b': _blk(['r3b'], 'r3b'),
          'combo': track_combo, 'reg': track_reg}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--block', nargs='+', required=True)
    ap.add_argument('--train', default='data/train.csv')
    ap.add_argument('--confirm', action='store_true')
    a = ap.parse_args()
    ctx = W.Ctx(a.train)
    for b in a.block:
        print(NL + f'########## block {b} ##########')
        t0 = time.time()
        TRACKS[b](ctx, a)
        print(f'[block {b} took {time.time() - t0:.0f}s]')


if __name__ == '__main__':
    main()
