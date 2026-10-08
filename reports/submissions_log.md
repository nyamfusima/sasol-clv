# Submissions log

Validation = train on snapshots ending before 1 Sep 2025, score on the 1 Sep 2025 snapshot.

| Date | File | What | Val RMSE fuel | Val RMSE non-fuel | Val F1 | Public | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 2026-10-05 | submission_baseline_v1.csv | LGBM, 16 monthly snapshots, seed 42 | 0.6052 | 0.7489 | 0.5047 | 0.2974 | pipeline test |
| 2026-10-06 | submission_v2.csv | 5-seed bag (42-46), monthly, hurdle regressions, reference classifier | 0.5994 | 0.7450 | 0.5058 | **0.2991** | combined score 0.28553 |

| 2026-10-06 | submission_v3.csv | bag20 regressions + prior matching to the 2025-09-01 mix (4 large classes) at alpha 0.5 | 0.5996 | 0.7451 | 0.5173 | 0.3031 | score 0.28929; best candidate |

| 2026-10-08 | submission_v4_simple.csv | lag classifier + simple Fuel x1.25 rule + v3 regressions | 0.5996 | 0.7451 | 0.5093 | not submitted | score 0.28796; second-pick candidate, hedge against fitted rules |
| 2026-10-08 | submission_diag_alpha075.csv | v4_hybrid with prior matching at alpha 0.75 | 0.5996 | 0.7451 | - | not submitted | **diagnostic, not for selection**; the folds disagreed on alpha |
| 2026-10-07 | submission_v4_lags.csv | lag series on classifier + both hurdle regressors, 20 seeds, prior matching alpha 0.5 | 0.5972 | 0.7458 | 0.5169 | **0.3045** | score 0.29119; best candidate |
| 2026-10-07 | submission_v4_hybrid.csv | hybrid: lag labels + v3 regressions | 0.5996 | 0.7451 | 0.5169 | *0.30515 inferred* | score 0.29068; labels identical to v4_lags and CLV identical to v3, so its public score follows exactly; **+0.00065 over v4_lags** |
| 2026-10-07 | submission_diag_r1_bundle.csv | v3 labels + R1 bundle regressions (f3f4 features, CatBoost magnitude) | 0.6003 | 0.7443 | 0.5173 | 0.3022 | **diagnostic, not for selection** (track R best non-passer, +0.00086) |

| 2026-10-06 | submission_v2_bag20.csv | 20-seed bag (42-61), monthly, hurdle regressions | 0.5996 | 0.7451 | 0.5045 | 0.2987 | score 0.28502; flip rate 1.42% vs 2.91% for 5-seed |
| 2026-10-06 | submission_v2_prior.csv | bag20 + Fuel-growth prior x1.25 (cross-fold validated) | 0.5996 | 0.7451 | 0.5096 | 0.3035 | score 0.28667; best candidate |

| 2026-10-06 | probe_stable.csv | all rows "Stable", bagged-reference CLV | 0.5994 | 0.7450 | - | 0.1406 | leaderboard probe, not submitted |
| 2026-10-06 | probe_inactivity.csv | all rows "Inactivity", bagged-reference CLV | 0.5994 | 0.7450 | - | 0.1473 | leaderboard probe, not submitted |
| 2026-10-06 | probe_fuelgrowth.csv | all rows "Existing-category growth: Fuel", bagged-reference CLV | 0.5994 | 0.7450 | - | 0.1336 | leaderboard probe, not submitted |

| 2026-10-06 | submission_g_dec10.csv | v2 regressions; classifier with analog 2024-12-01 weighted x10 | 0.5994 | 0.7450 | 0.4959* | not submitted | block G diagnostic, not submitted |
| 2026-10-06 | submission_g_dec_only.csv | v2 regressions; classifier trained only on 2024-11/12, 2025-01 | 0.5994 | 0.7450 | 0.4580* | not submitted | block G diagnostic, not submitted |

\* F1 is the validation score of the equivalent fold configuration, not of the
test file itself. Both are worse than v2's 0.5058 and are diagnostics only.

Combined score = 0.4*F1 + 0.3*(1 - RMSE_fuel/0.74) + 0.3*(1 - RMSE_nonfuel/0.816).
RMSE/F1 columns above are fold 2 (1 Sep 2025), matching this log's convention.

| Stack | Mean score | Fold 2025-06 | Fold 2025-09 |
| --- | --- | --- | --- |
| single-seed baseline | 0.28164 | 0.28208 | 0.28120 |
| 5-seed bagged reference | 0.28342 | 0.28387 | 0.28296 |
| **submission_v2** (bagged + hurdle) | **0.28553** | 0.28561 | 0.28545 |

Public feedback: v2 scored **0.2991**. The single-seed baseline's public score
works out to 0.29741 from its reported components, so bagging + the hurdle moved
the public score by about +0.0017 where validation predicted +0.00389. **The
public result confirms the direction of the gain; its size is within public-board
noise.** The public split is roughly 1,650 customers, and one observation on that
many customers cannot establish how validation gains translate -- do not read a
ratio into it.

submission_v2 validation: 5,488 rows, IDs match data/test.csv, 11 distinct
labels all in label_config.json, both CLV columns >= 0 (min 0.0256).

Every file in `submissions/` is named after the run that produced it and has a
row in this table. An unnamed `submission.csv` left over from a baseline run was
deleted on 6 Oct (it was byte-identical to `submission_baseline_v1.csv`), and
`baseline.py` now defaults its `--out` to that named file so the leftover cannot
reappear.

No submission was produced on 5 Oct: the label v2 sweep (variants a-f) peaked at
a held-out mean F1 of 0.5011 vs the baseline's 0.4998, which fails the +0.010 bar,
so `submission_baseline_v1.csv` remains the live entry. Full table in `notes.md`.

All files are written with `float_format='%.17g'` and must be read with
`float_precision='round_trip'`: pandas' default CSV float parser is not
correctly rounded and shifts values by 1 ULP on every read-rewrite cycle. Every
file here is read-write byte-identical, and the bag20-derived files
(`submission_v2_prior.csv`, `submission_v3.csv`) have CLV columns bit-identical
to `submission_v2_bag20.csv`.

Block I (projected target mixes) produced no new submission: the projections are
more accurate targets but all score below the stale target, so `submission_v3.csv`
remains the best candidate at 0.28929.

## Label v2 sweep (validation, two folds -- not submitted)
| Variant | Fold 2025-06 | Fold 2025-09 | Mean | Kept |
| --- | --- | --- | --- | --- |
| base (reference) | 0.4949 | 0.5047 | 0.4998 | - |
| a growth-rule features | 0.4935 | 0.5051 | 0.4993 | dropped |
| b fine recency / activity | 0.4908 | 0.5020 | 0.4964 | dropped |
| c same season last year | 0.4908 | 0.5014 | 0.4961 | dropped |
| d two-stage | 0.4939 | 0.4972 | 0.4956 | dropped |
| e rule-derived | 0.3842 | 0.3846 | 0.3844 | dropped |
| f class-prior (held-out) | 0.4969 | 0.5053 | 0.5011 | kept, +0.0013 |

## Validation against public, per kept change

Public components are F1 / RMSE fuel / RMSE non-fuel. Every row is the delta
between two submitted files, so the public column is measured, not inferred.

| change | val delta | public delta | public dF1 | public d rmse_f | public d rmse_nf |
| --- | --- | --- | --- | --- | --- |
| bagging + hurdle (baseline -> v2) | +0.00389 | **+0.0017** | +0.0025 | -0.0013 | -0.0007 |
| 20-seed bag (v2 -> v2_bag20) | -0.00051 | -0.0004 | -0.0018 | -0.0004 | -0.0003 |
| Fuel x1.25 prior (bag20 -> v2_prior) | +0.00166 | **+0.0048** | **+0.0120** | 0.0000 | 0.0000 |
| partial prior matching (v2_prior -> v3) | +0.00262 | **-0.0004** | -0.0008 | 0.0000 | 0.0000 |
| lag series, all 3 models (v3 -> v4_lags) | +0.00190 | **+0.0014** | +0.0050 | +0.0012 | +0.0003 |
| R1 bundle regressions (v3 -> diag_r1_bundle) | +0.00086 | **-0.0009** | 0.0000 | worse | worse |

**Label gains transferred; small regression gains did not.**
- The two large label changes transferred and one amplified. Fuel x1.25 was
  +0.0041 F1 on validation and **+0.0120 on public**, roughly three times the
  size. The lag classifier was +0.0034 on validation and +0.0050 on public.
- The large structural regression change transferred: bagging + hurdle improved
  public RMSE on both targets (-0.0013, -0.0007).
- Every small regression refinement failed. The lag regressors made public RMSE
  **worse** (+0.0012, +0.0003) where validation said better, and the R1 bundle
  went from +0.00086 on validation to **-0.0009 on public**.
- The one label change that did not transfer is partial prior matching: +0.00262
  on validation, -0.0004 on public, with F1 going the wrong way by 0.0008. It
  was also the most heavily fitted change -- an IPF procedure re-estimated per
  dataset with alpha chosen cross-fold -- against Fuel x1.25, a single constant
  both folds independently agreed on. The simpler calibration travelled; the
  machinery did not.

So the dividing line is not label-versus-regression but **how much machinery a
change carries**. Single global constants and structural model changes survived
the public split; per-dataset procedures and sub-0.001 refinements did not.

### The hybrid would have beaten both final picks
`submission_v4_hybrid.csv` was never submitted, but its public score follows
exactly, because its labels are bit-identical to v4_lags and its CLV columns are
bit-identical to v3:

| | F1 | rmse fuel | rmse nonfuel | public |
| --- | --- | --- | --- | --- |
| v3 (selected) | 0.5239 | 0.5932 | 0.7233 | 0.3031 |
| v4_lags (selected) | 0.5289 | 0.5944 | 0.7236 | 0.3045 |
| **v4_hybrid** (not selected) | 0.5289 | 0.5932 | 0.7233 | **0.30515** |

+0.00065 over v4_lags and +0.00205 over v3. It keeps the lag classifier's
+0.0050 F1 and drops the lag regressors' +0.0012/+0.0003 RMSE damage. Validation
had it dominated by 0.00051 -- exactly backwards, and for the reason the table
above gives: the lag regressors were a sub-0.001 refinement and did not travel.

### The probes confirm the scoring formula
The three constant-label probes are scores, not F1 values. For a constant-class
submission the weighted F1 is 2p^2/(1+p) with p the class's true share, so each
probe gives 0.4*2p^2/(1+p) + R where R is the regression term they all share.
Solving with the independently supplied shares:

| probe | score | share | implied F1 | implied R |
| --- | --- | --- | --- | --- |
| Stable | 0.1406 | 0.276 | 0.11940 | 0.09284 |
| Inactivity | 0.1473 | 0.297 | 0.13602 | 0.09289 |
| Fuel growth | 0.1336 | 0.252 | 0.10144 | 0.09302 |

R agrees across all three to 0.00018, which can only happen if the 0.4/0.3/0.3
weighting and the 0.74/0.816 normalisers are both right. Inverting instead gives
shares 0.2757 / 0.2969 / 0.2524 against the supplied 0.276 / 0.297 / 0.252.
Together with the formula reproducing six public rows to within 0.00006, the
inferred constants are confirmed as far as they can be without publication.

## Reference baselines (validation)
| Method | RMSE fuel | RMSE non-fuel | F1 |
| --- | --- | --- | --- |
| Same as last quarter | 0.792 | 0.946 | 0.360 |
| Predict zero | 1.834 | 1.224 | - |

## Leaderboard reference
- Direction (higher or lower wins):
- 1st / 3rd / 10th:
- Exact closing time:

## Final selection
- Submission 1:
- Submission 2:
- Reasoning (validation + public evidence):
