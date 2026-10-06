# Submissions log

Validation = train on snapshots ending before 1 Sep 2025, score on the 1 Sep 2025 snapshot.

| Date | File | What | Val RMSE fuel | Val RMSE non-fuel | Val F1 | Public | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 2026-10-05 | submission_baseline_v1.csv | LGBM, 16 monthly snapshots, seed 42 | 0.6052 | 0.7489 | 0.5047 | ? | pipeline test |
| 2026-10-06 | submission_v2.csv | 5-seed bag (42-46), monthly, hurdle regressions, reference classifier | 0.5994 | 0.7450 | 0.5058 | **0.2991** | combined score 0.28553 |

| 2026-10-06 | probe_stable.csv | all rows "Stable", bagged-reference CLV | 0.5994 | 0.7450 | - | ? | leaderboard probe, not submitted |
| 2026-10-06 | probe_inactivity.csv | all rows "Inactivity", bagged-reference CLV | 0.5994 | 0.7450 | - | ? | leaderboard probe, not submitted |
| 2026-10-06 | probe_fuelgrowth.csv | all rows "Existing-category growth: Fuel", bagged-reference CLV | 0.5994 | 0.7450 | - | ? | leaderboard probe, not submitted |

| 2026-10-06 | submission_g_dec10.csv | v2 regressions; classifier with analog 2024-12-01 weighted x10 | 0.5994 | 0.7450 | 0.4959* | ? | block G diagnostic, not submitted |
| 2026-10-06 | submission_g_dec_only.csv | v2 regressions; classifier trained only on 2024-11/12, 2025-01 | 0.5994 | 0.7450 | 0.4580* | ? | block G diagnostic, not submitted |

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
