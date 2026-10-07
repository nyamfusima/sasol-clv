"""Sweep 5: maximise. Tracks R (regressions), L (label model), D (decision rule).

Conventions that make the tracks comparable:
  * Regression variants are measured on top of the hurdle and hold F1 at v3's
    decision-rule values, so touches=('rf','rn').
  * Label variants are measured as probabilities pushed through the SAME v3
    decision rule (partial prior matching, 4 large classes, alpha 0.5), which
    separates model effects from rule effects.
  * Screening runs at 5 seeds (42-46); anything promising is re-measured at 20
    (42-61). Seed sd is 0.00020 against a 0.0015 bar, so screening cannot hide a
    passing effect.

    python src/sweep5.py --track r1
"""
import argparse, json, time
from pathlib import Path
import numpy as np, pandas as pd
import calibrate as C
import make_submission as M
import snapshots as S
import sweep2 as W
import train_label as T
import validate as V

RESULTS = Path('preds/sweep5.json')
SCREEN = tuple(range(42, 47))
CONFIRM = tuple(range(42, 62))
# v3 = hurdle regressions (bag20) + partial prior matching at alpha 0.5
V3_F1 = [0.5022, 0.5173]
V3_RF = [0.5969, 0.5996]
V3_RN = [0.7350, 0.7451]
# the 5-seed hurdle baseline and the pieces already measured in sweep 2
FIVE_SEED = {
    'hurdle (sweep2 b1)':            dict(rf=[0.5975, 0.5994], rn=[0.7352, 0.7450]),
    '+ f3f4 features (freg)':        dict(rf=[0.5957, 0.6005], rn=[0.7336, 0.7448]),
    '+ catboost magnitude (b3)':     dict(rf=[0.5960, 0.5989], rn=[0.7339, 0.7444]),
    '+ lgbm/cat average (b3)':       dict(rf=[0.5965, 0.5989], rn=[0.7344, 0.7446]),
}


def score(f1s, rf, rn):
    return float(np.mean([M.combined(f1s[i], rf[i], rn[i]) for i in (0, 1)]))


def save(key, obj):
    RESULTS.parent.mkdir(exist_ok=True)
    allr = json.loads(RESULTS.read_text()) if RESULTS.exists() else {}
    allr[key] = obj
    RESULTS.write_text(json.dumps(allr, indent=2))


def judge_reg(rf, rn, base_rf, base_rn, f1s=None):
    """Standing rule on a regression-only change: +0.0015 mean score, both folds
    up, and neither touched component worse on either fold by more than 0.001."""
    f1s = f1s or V3_F1
    s, b = score(f1s, rf, rn), score(f1s, base_rf, base_rn)
    gain = s - b
    both = all(M.combined(f1s[i], rf[i], rn[i]) > M.combined(f1s[i], base_rf[i], base_rn[i])
               for i in (0, 1))
    worst = max(max(rf[i] - base_rf[i], rn[i] - base_rn[i]) for i in (0, 1))
    ok = gain >= 0.0015 and both and worst <= 0.001
    why = []
    if gain < 0.0015:
        why.append(f'gain {gain:+.5f}<0.0015')
    if not both:
        why.append('one fold down')
    if worst > 0.001:
        why.append(f'component -{worst:.4f}')
    return dict(score=s, gain=gain, keep='KEPT' if ok else 'dropped', note=' '.join(why))


def row(name, rf, rn, base_rf, base_rn, f1s=None):
    j = judge_reg(rf, rn, base_rf, base_rn, f1s)
    print(f'{name:<38}{rf[0]:>8.4f}{rf[1]:>8.4f}{rn[0]:>8.4f}{rn[1]:>8.4f}'
          f'{j["score"]:>10.5f}{j["gain"]:>+10.5f}  {j["keep"]} {j["note"]}')
    return dict(name=name, rf=rf, rn=rn, **j)


def header():
    print(f'{"variant":<38}{"rf f1":>8}{"rf f2":>8}{"rn f1":>8}{"rn f2":>8}'
          f'{"score":>10}{"gain":>10}  verdict')


# --- R1 ----------------------------------------------------------------------

def track_r1(ctx, args):
    """Bundle the individually sub-bar, same-direction regression changes:
    f3+f4 features for the regressors and a LightGBM/CatBoost magnitude
    average, at monthly spacing. The pieces were measured separately in sweep 2
    at 5 seeds; only the bundle is new."""
    cuts = sorted(set(ctx.monthly + list(V.FOLDS) + [W.TEST_CUTOFF]))
    extra = W.f_snapshots(ctx, ('f3', 'f4'), cuts)
    base = FIVE_SEED['hurdle (sweep2 b1)']

    print('\n--- 5-seed screen: pieces (from sweep 2) and the new bundles ---')
    header()
    rows = [row(n + ' [5s]', v['rf'], v['rn'], base['rf'], base['rn'])
            for n, v in FIVE_SEED.items()]
    bundles = {}
    for name, kind, ex in (('bundle f3f4 + lgbm/cat avg', 'hurdle_avg', extra),
                           ('bundle f3f4 + catboost mag', 'hurdle_cat', extra)):
        t0 = time.time()
        W.SEEDS = list(SCREEN)
        rf, rn = W.reg_rmses(ctx, kind, mode='monthly', extra=ex)
        bundles[name] = (rf, rn)
        rows.append(row(name + ' [5s]', rf, rn, base['rf'], base['rn']))
        print(f'    ({time.time() - t0:.0f}s)')

    # marginal effect of each addition, on the combined score
    print('\n--- marginal effect of each addition (5-seed, score) ---')
    s_base = score(V3_F1, base['rf'], base['rn'])
    for n, v in FIVE_SEED.items():
        if n == 'hurdle (sweep2 b1)':
            continue
        print(f'  {n:<34} alone {score(V3_F1, v["rf"], v["rn"]) - s_base:+.5f}')
    for n, (rf, rn) in bundles.items():
        print(f'  {n:<34} bundle {score(V3_F1, rf, rn) - s_base:+.5f}')
    f34 = FIVE_SEED['+ f3f4 features (freg)']
    for n, (rf, rn) in bundles.items():
        piece = ('+ lgbm/cat average (b3)' if 'avg' in n else '+ catboost magnitude (b3)')
        p = FIVE_SEED[piece]
        print(f'  {n:<34} marginal of f3f4 given {piece.split("(")[0].strip():<24}'
              f'{score(V3_F1, rf, rn) - score(V3_F1, p["rf"], p["rn"]):+.5f}')
        print(f'  {n:<34} marginal of {piece.split("(")[0].strip():<34}'
              f'{score(V3_F1, rf, rn) - score(V3_F1, f34["rf"], f34["rn"]):+.5f}')

    # confirm the best screened bundle at 20 seeds against bag20's hurdle
    best = max(bundles, key=lambda k: score(V3_F1, *bundles[k]))
    print(f'\n--- 20-seed confirmation of "{best}" against v3 regressions ---')
    kind = 'hurdle_avg' if 'avg' in best else 'hurdle_cat'
    t0 = time.time()
    W.SEEDS = list(CONFIRM)
    rf, rn = W.reg_rmses(ctx, kind, mode='monthly', extra=extra)
    header()
    rows.append(row('v3 regressions [20s, incumbent]', V3_RF, V3_RN, V3_RF, V3_RN))
    conf = row(best + ' [20s]', rf, rn, V3_RF, V3_RN)
    rows.append(conf)
    print(f'    ({time.time() - t0:.0f}s)')
    W.SEEDS = list(CONFIRM)
    save('r1', dict(rows=rows, confirmed=conf, bundle=best,
                    bundle_rf=rf, bundle_rn=rn))
    return conf


TRACKS = {'r1': track_r1}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--track', nargs='+', required=True)
    ap.add_argument('--train', default='data/train.csv')
    a = ap.parse_args()
    ctx = W.Ctx(a.train)
    for t in a.track:
        print(f'\n########## track {t} ##########')
        t0 = time.time()
        TRACKS[t](ctx, a)
        print(f'[track {t} took {time.time() - t0:.0f}s]')


if __name__ == '__main__':
    main()
