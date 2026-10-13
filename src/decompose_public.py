"""Decompose the sweep 11 public readings into a classifier half and a
regression half, and apply the amended pre-registered rule.

Three files were uploaded on 13 Oct 2026. Two of them are composed so that the
decomposition is exact rather than estimated:

    diag_v6_clf   new labels, CLV byte-identical to v5_fuel2  -> pure F1 change
    v6_reg        the same labels, new CLV                    -> pure RMSE change

So `diag_v6_clf - v5_fuel2` isolates the classifier and `v6_reg -
diag_v6_clf` isolates the hurdles, with no term estimated and nothing cancelled
approximately.

It also estimates the one thing the arithmetic cannot settle: whether a paired
difference measured on the ~1,650 public rows says anything about the ~3,838
private ones. For a paired RMSE comparison,

    mse_a - mse_b = (1/n) * sum_i (p_a - p_b)(p_a + p_b - 2y)
                  = (1/n) * sum_i d_i * s_i

where d_i is the known per-row prediction difference and s_i the sum of the two
residuals. The two models are close, so s_i ~ 2*r_i for a shared residual r
with E[r^2] = rmse^2; d is independent of r; hence

    se(mse difference)  ~ 2 * rmse * rms(d) / sqrt(n)
    se(rmse difference) ~ rms(d) / sqrt(n)          [since d(rmse) = d(mse)/(2*rmse)]

which needs only the per-row differences, which we have, and the public RMSE,
which is published. The independence of d and r is an assumption and is the one
soft step; everything else is arithmetic.

    python src/decompose_public.py
"""
import numpy as np
import pandas as pd

from check_submission import read

FUEL_NORM, NONFUEL_NORM = 0.74, 0.816
N_PUBLIC = 1650          # 30% of 5,488, the public split size

# --- published public readings ------------------------------------------------
V5_FUEL2 = dict(zid='nskGdBPf', score=0.309785160, f1=0.535300706,
                rmse_fuel=0.588056839, rmse_nonfuel=0.723339664)
V5_A050 = dict(zid='gmpjvZn9', score=0.307223445, f1=0.528896416)
V6_REG = dict(zid='fFWfbjda', score=0.308725795)
V6_A050 = dict(zid='8qCwUGQy', score=0.307465421)
V6_CLF = dict(zid='aUswGpyi', score=0.309807976, f1=0.535357746)

# Leaderboard, read off 13 Oct 2026 14:20 SAST. Our row is aUswGpyi.
BOARD = {
    '1st':  dict(score=0.320813915, f1=0.558513, fuel=0.588188, nonfuel=0.718452),
    '3rd':  dict(score=0.320394545, f1=0.548499, fuel=0.580036, nonfuel=0.717687),
    '10th': dict(score=0.315329722, f1=0.548815, fuel=0.588512, nonfuel=0.722460),
}
MY_RANK = 35

GATE = 0.3078
PAIRED_BAR = -0.0010
VAL_CLF_HALF, VAL_REG_HALF, VAL_BOTH = 0.00090, 0.00113, 0.00203


def rmse_half(rmse_fuel, rmse_nonfuel):
    return 0.3 * (1 - rmse_fuel / FUEL_NORM) + 0.3 * (1 - rmse_nonfuel / NONFUEL_NORM)


def line(k, v, w=34):
    print(f'  {k:<{w}}{v}')


def main():
    print('\n--- the fixed constant ---')
    v5_half = rmse_half(V5_FUEL2['rmse_fuel'], V5_FUEL2['rmse_nonfuel'])
    line('v5 RMSE half', f'{v5_half:.9f}')
    line('inverts v5 public F1 to', f'{(V5_FUEL2["score"] - v5_half) / 0.4:.9f}'
                                    f'   (Zindi {V5_FUEL2["f1"]:.9f})')

    print('\n--- the two halves, exact ---')
    clf_half = V6_CLF['score'] - V5_FUEL2['score']
    reg_half = V6_REG['score'] - V6_CLF['score']
    both = V6_REG['score'] - V5_FUEL2['score']
    f1_v6 = (V6_CLF['score'] - v5_half) / 0.4
    line('classifier half', f'{clf_half:+.9f}   (validation {VAL_CLF_HALF:+.5f})')
    line('  implied public F1', f'{f1_v6:.9f}   (Zindi {V6_CLF["f1"]:.9f})')
    line('  public F1 gain', f'{f1_v6 - V5_FUEL2["f1"]:+.9f}')
    line('regression half', f'{reg_half:+.9f}   (validation {VAL_REG_HALF:+.5f})')
    v6_half = V6_REG['score'] - 0.4 * f1_v6
    line('  v6 RMSE half', f'{v6_half:.9f}   (v5 {v5_half:.9f})')
    line('both halves together', f'{both:+.9f}   (validation {VAL_BOTH:+.5f})')
    line('  halves add to', f'{clf_half + reg_half:+.9f}   '
                            f'residual {both - clf_half - reg_half:+.2e}')

    print('\n--- the alpha 0.5 family ---')
    f1_a050 = (V6_A050['score'] - v6_half) / 0.4
    line('v6_reg_a050 implied F1', f'{f1_a050:.9f}')
    line('  over v5_fuel2_a050', f'{f1_a050 - V5_A050["f1"]:+.9f}')
    pred = 0.4 * f1_a050 + v5_half
    line('v6_clf_a050 PREDICTED public', f'{pred:.9f}')
    line('  over gmpjvZn9', f'{pred - V5_A050["score"]:+.9f}')

    print('\n--- sampling noise of the paired reading ---')
    a = read('submissions/submission_v6_reg.csv').set_index('ID')
    b = read('submissions/submission_v5_fuel2.csv').set_index('ID').reindex(a.index)
    se2 = 0.0
    for col, norm, rmse in (('CLV_fuel', FUEL_NORM, V5_FUEL2['rmse_fuel']),
                            ('CLV_nonfuel', NONFUEL_NORM, V5_FUEL2['rmse_nonfuel'])):
        d = a[col].to_numpy() - b[col].to_numpy()
        se_rmse = float(np.sqrt((d ** 2).mean()) / np.sqrt(N_PUBLIC))
        se_score = 0.3 * se_rmse / norm
        se2 += se_score ** 2
        line(f'{col} rms(d)', f'{np.sqrt((d ** 2).mean()):.6f}   '
                              f'-> se(rmse) {se_rmse:.6f}   se(score) {se_score:.6f}')
    se = float(np.sqrt(se2))
    line('se of the paired difference', f'{se:.6f}')
    gap = VAL_REG_HALF - reg_half
    line('validation minus measured', f'{gap:+.6f}   = {gap / se:.1f} standard errors')

    print('\n' + '--- the gap to the board, decomposed ---')
    mine = dict(score=V6_CLF['score'], f1=V6_CLF['f1'],
                fuel=V5_FUEL2['rmse_fuel'], nonfuel=V5_FUEL2['rmse_nonfuel'])
    print(f'  {"":<8}{"gap":>11}{"from F1":>11}{"from fuel":>11}'
          f'{"from nonfuel":>13}{"check":>11}')
    for k, o in BOARD.items():
        gp = o['score'] - mine['score']
        g_f1 = 0.4 * (o['f1'] - mine['f1'])
        g_fu = 0.3 * (mine['fuel'] - o['fuel']) / FUEL_NORM
        g_nf = 0.3 * (mine['nonfuel'] - o['nonfuel']) / NONFUEL_NORM
        print(f'  {k:<8}{gp:>+11.6f}{g_f1:>+11.6f}{g_fu:>+11.6f}{g_nf:>+13.6f}'
              f'{gp - g_f1 - g_fu - g_nf:>+11.1e}')
    print(f'  rank {MY_RANK}. our fuel RMSE {mine["fuel"]:.6f} is the best of these '
          f'four; our non-fuel {mine["nonfuel"]:.6f} is the worst.')

    print('\n--- the amended rule (notes.md, 13 Oct 13:18 SAST) ---')
    branch_ok = reg_half >= PAIRED_BAR
    if branch_ok:
        pick1, name = V6_REG, 'v6_reg + v6_reg_a050'
    else:
        pick1, name = V6_CLF, 'diag_v6_clf (promoted) + v6_clf_a050'
    line('regression half', f'{reg_half:+.9f} vs bar {PAIRED_BAR:+.4f}'
                            f'  -> {"transferred" if branch_ok else "DID NOT transfer"}')
    line('candidate pair', name)
    adopt = pick1['score'] >= GATE
    line(f'gate on {pick1["zid"]}', f'{pick1["score"]:.9f} >= {GATE}  -> '
                                    f'{"ADOPT" if adopt else "reject, picks stand"}')
    print()


if __name__ == '__main__':
    main()
