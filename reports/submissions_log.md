# Submissions log

Validation = train on snapshots ending before 1 Sep 2025, score on the 1 Sep 2025 snapshot.

| Date | File | What | Val RMSE fuel | Val RMSE non-fuel | Val F1 | Public | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 2026-10-05 | submission_baseline_v1.csv | LGBM, 16 monthly snapshots, seed 42 | 0.6052 | 0.7489 | 0.5047 | ? | pipeline test |
| 2026-10-06 | submission_v2.csv | 5-seed bag (42-46), monthly, hurdle regressions, reference classifier | 0.5994 | 0.7450 | 0.5058 | ? | not submitted; combined score 0.28553 |

Combined score = 0.4*F1 + 0.3*(1 - RMSE_fuel/0.74) + 0.3*(1 - RMSE_nonfuel/0.816).
RMSE/F1 columns above are fold 2 (1 Sep 2025), matching this log's convention.

| Stack | Mean score | Fold 2025-06 | Fold 2025-09 |
| --- | --- | --- | --- |
| single-seed baseline | 0.28164 | 0.28208 | 0.28120 |
| 5-seed bagged reference | 0.28342 | 0.28387 | 0.28296 |
| **submission_v2** (bagged + hurdle) | **0.28553** | 0.28561 | 0.28545 |

submission_v2 validation: 5,488 rows, IDs match data/test.csv, 11 distinct
labels all in label_config.json, both CLV columns >= 0 (min 0.0256).

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
