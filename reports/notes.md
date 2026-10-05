# Notes

## Data findings (5 Oct)
- 467,880 lines, 6,204 customers, 390 sites, 19 Mar 2024 to 30 Nov 2025.
- Test = 5,488 customers = everyone with 3+ baskets before 1 Dec 2025.
- `label_rules.py` reproduces the published `s` constants exactly, so we can label any past cutoff.
- 73% of rows are fuel. Per quarter about 29% of customers have zero fuel and 65% zero non-fuel.
- Labels (latest snapshot): Fuel growth 27%, Inactivity 27%, Stable 27%; each adoption class about 1% or less.
- Inactivity rises every quarter (7% to 27%); expect about 29% in test.
- Early snapshots have short history, so adoption is inflated there (`hist_days` feature covers this).

## Setup verified (5 Oct)
- Env: Python 3.13, pandas 3.0.6, numpy 2.5.3, scikit-learn 1.9.1, lightgbm 4.7.0.
- `python src/baseline.py` reproduces exactly: RMSE fuel 0.6052 | RMSE nonfuel 0.7489 | weighted F1 0.5047
  (13 train snapshots, 53,025 rows). Output is bit-identical to `submissions/submission_baseline_v1.csv`.
- Data copied from `~/Downloads/train (1).csv` / `test (1).csv` / `SampleSubmission.csv`; 467,880 rows, 5,488 test IDs.
- Note: `~/Downloads/Train.csv` / `Test.csv` belong to a different competition, do not use them.

## Plan
1. Label model: growth vs stable vs inactive. Key signal: was last quarter unusually low.
2. Derive the label from predicted per-category spend, blend with the direct classifier.
3. Two-stage models for the numbers (buys at all, then how much).
4. Seasonality: only one past Dec to Feb window (cutoff 1 Dec 2024).
5. Ensemble LightGBM, CatBoost and seeds. Test everything on validation before submitting.

## Label model v2 sweep (5 Oct) -- NO-GO, nothing shipped

Two time-based folds (`src/validate.py`): validate on 2025-06-01 and 2025-09-01,
each training only on snapshots whose 3-month outcome window ends on or before
that cutoff. Fold 2 is the baseline's own split, so it reproduces 0.5047 exactly.
Baseline reference: fold1 0.4949, fold2 0.5047, mean 0.4998.

| variant | fold 2025-06 | fold 2025-09 | mean | kept |
| --- | --- | --- | --- | --- |
| base (reference) | 0.4949 | 0.5047 | 0.4998 | - |
| a growth-rule features | 0.4935 | 0.5051 | 0.4993 | dropped |
| b fine recency / activity | 0.4908 | 0.5020 | 0.4964 | dropped |
| c same season last year | 0.4908 | 0.5014 | 0.4961 | dropped |
| d two-stage (active / class) | 0.4939 | 0.4972 | 0.4956 | dropped |
| e rule-derived from predicted spend | 0.3842 | 0.3846 | 0.3844 | dropped |
| f class-prior adjustment (held-out) | 0.4969 | 0.5053 | 0.5011 | kept, +0.0013 |

**Decision: NO-GO.** Best honest mean 0.5011 = +0.0013 over baseline, far short of
the +0.010 bar. No submission written; the baseline stays as the live entry.

### What we learned
- The new features are genuinely discriminative but do not change argmax
  decisions. `g_Fuel_need` reproduces the official Fuel-growth qualification for
  99.87% of customers (remainder is the strict/non-strict boundary), and fuel
  growth is clearly "a declining fuel buyer who bounces back": `slope6_fuel_r`
  -45 for Fuel growth vs +61 for Stable, `g_Fuel_prev_zero` 0.21 vs 0.009.
  LightGBM already extracts that from the baseline's `fuel_q1_vs_rate` /
  `fuel_q1_vs_q2`; the extra 34 columns only dilute `colsample_bytree=0.7`.
- (c) cannot be trained on this history. "Same season last year" needs 12 months
  of lookback, which only exists from the 2025-04-01 cutoff onwards, so it is
  missing in ALL 10 fold-1 training snapshots while being present at fold 1's
  validation cutoff. Revisit only if more history is released.
- (e) is sound but too noisy to use. Fed the TRUE future per-category spend,
  `rules_to_proba` reproduces `label_rules.raw_labels` exactly (0 mismatches of
  4961 and 5238), so all of its 0.38 comes from the spend predictions, not the
  rule logic. Per-category quarterly spend is simply not predictable enough.
  Its output is hard one-hot, so blending is monotonically worse at every weight
  and saturates to the pure rule at w>=0.5. A soft (probabilistic) version is
  the only version worth retrying.
- (f) is the only thing that works, and barely. In-fold gain +0.0093 vs held-out
  +0.0006/+0.0020: the weights mostly memorise their own fold. Regularising did
  not help either -- restricting the fit to the 10 classes with >=100 support
  gives +0.0012, and `class_weight='balanced'` is far worse at -0.0778.
  It does rebalance the big confusion: Fuel growth F1 0.4539 -> 0.4864 and
  growth:Other 0.337 -> 0.455, paid for with Stable 0.574 -> 0.546.
- Adoption classes stay at ~0.00 F1 under every variant. Boosting their priors
  raises predicted counts (adoption:Other 6 -> 48) without raising F1, i.e. pure
  false positives. At ~1% base rates there is no signal in these features.
- CAUTION, cost us a wrong headline once: averaging the two fold-fitted weight
  vectors and re-scoring both folds leaks each fold's own weights into its score
  (0.5069 instead of the honest 0.5011). Only ever quote the cross-fold number.

### Where the remaining gap probably is
1st place public F1 0.557 vs our 0.512. Adoption classes are ~7% of the weight at
F1 0.000, worth about +0.021 if they could reach 0.3 -- that is the largest
single block of unclaimed weight, and nothing in this sweep touched it.
Fuel growth 0.45 -> 0.60 would be worth about +0.040.

### Next, in priority order
1. Attack adoption directly: one binary model per adoption category on
   "never bought X, buys X next quarter", rather than hoping a 17-class softmax
   surfaces a 1% class. This is the only untried route to the biggest unclaimed
   block of weight.
2. Soft version of (e): calibrated P(qualify) per category from the predicted
   spend distribution instead of a point estimate, then blend.
3. Tune the classifier itself (depth, `min_child_samples`, more estimators) --
   the sweep changed features and decision rules but never the model capacity.

## Leakage checks run (5 Oct)
- Rebuilt features from data physically truncated at each cutoff (dropping
  148,398 rows at fold 1 and 76,717 at fold 2): 0 of 166 columns changed. All
  34 `g_*`, 6 `slope6_*`/`last6_*` and 21 other recency columns bit-identical.
- `fsp_*`/`fn_*` (future per-category spend) live only in `cat_targets.parquet`
  and appear in none of the 166 model-fed columns. Their sole consumer is the
  per-category spend regressors' training target, read from training snapshots
  only (`tgt_tr`); the validation-cutoff copy is deliberately never loaded.

## Leaderboard scoring formula (confirmed 5 Oct)

    Score = 0.4*F1 + 0.3*(1 - RMSE_fuel/0.74) + 0.3*(1 - RMSE_nonfuel/0.816)

Higher is better. `src/validate.py` exposes `combined()` and `score_row()`; every
decision from here uses that single number.

Gradients: +0.4 per unit F1, -0.4054 per unit RMSE_fuel, -0.3676 per unit
RMSE_nonfuel. A 0.005 RMSE drop is worth about as much as a 0.005 F1 gain, so the
three GO bars (+0.005 F1, -0.005 on each RMSE) are worth +0.00200, +0.00203 and
+0.00184 score -- deliberately comparable.

Reference points on this formula:
| What | F1 | RMSE fuel | RMSE nonfuel | Score |
| --- | --- | --- | --- | --- |
| baseline val fold 2025-06 | 0.4949 | 0.5998 | 0.7418 | 0.28208 |
| baseline val fold 2025-09 | 0.5047 | 0.6052 | 0.7489 | 0.28120 |
| our public | 0.512 | 0.595 | 0.724 | 0.29741 |
| 1st public | 0.557 | 0.589 | 0.722 | 0.31858 |

The public gap to 1st is 0.02117, and it is 85% label: F1 contributes +0.01800,
RMSE fuel +0.00243, RMSE nonfuel +0.00074. Regression work is cheap and does
count, but F1 is where the race is decided.

## Sweep 2 (5 Oct) -- 5-seed bagged, decided on the combined score

New reference: baseline features, monthly snapshots, seeds 42-46 averaged.
| | F1 | rmse fuel | rmse nonfuel | Score |
| --- | --- | --- | --- | --- |
| single-seed baseline | 0.4998 | 0.6025 | 0.7453 | 0.28164 |
| **5-seed bagged reference** | 0.5006 | 0.6005 | 0.7436 | **0.28342** |

Bagging is worth +0.00178, and nearly all of it is regression variance reduction
(rmse -0.0020 / -0.0018) rather than F1 (+0.0008). GO bars are measured against
the bagged reference, as they should be.

### A1 recency weighting (classifier) -- all dropped
| variant | fold 2025-06 | fold 2025-09 | mean F1 | delta | kept |
| --- | --- | --- | --- | --- | --- |
| reference | 0.4953 | 0.5058 | 0.5006 | - | reference |
| half-life 3mo | 0.4916 | 0.5028 | 0.4972 | -0.0034 | dropped |
| half-life 6mo | 0.4955 | 0.5044 | 0.5000 | -0.0006 | dropped |
| half-life 12mo | 0.4971 | 0.5042 | 0.5007 | +0.0001 | dropped |

Monotone in the half-life and converging to the reference from below: the longer
the half-life, the better, with the limit (uniform weights) being the reference.
There is no distribution shift here that down-weighting old snapshots corrects
for -- you only lose effective sample size. Do not revisit unless the label mix
starts moving faster than it has.

### A2 snapshot spacing (classifier) -- all dropped, but informative
| variant | snaps (fold 2) | fold 2025-06 | fold 2025-09 | mean F1 | delta | kept |
| --- | --- | --- | --- | --- | --- | --- |
| reference monthly | 13 | 0.4953 | 0.5058 | 0.5006 | - | reference |
| every 2 months | 7 | 0.4944 | 0.5072 | 0.5008 | +0.0002 | dropped |
| quarterly non-overlapping | 5 | 0.4937 | 0.5062 | 0.4999 | -0.0006 | dropped |

Neither clears the bar, but the real finding is that **cutting the training set
from 13 snapshots to 5 costs essentially nothing** (-0.0006). Monthly snapshots
overlap heavily -- the same customers, outcome windows sliding by one month -- so
the extra rows are near-duplicates carrying little independent information. The
training signal is already saturated at quarterly spacing.

Two consequences:
- Quarterly spacing is 2.6x cheaper for the same F1. Use it for fast iteration.
- It predicts block D will fail: if going from 13 to 5 snapshots is free, going
  from 13 to ~53 (weekly) should add nothing either. D is the direct test.

### A3 capacity grid (classifier) -- dropped on both selection routes
18 configs over num_leaves {7,15,31} x min_child_samples {30,100,300} x
colsample {0.4,0.7}, n_estimators by early stopping on fold 1 then fixed.

The grid is flat: fold-2 single-seed F1 spans 0.4937..0.5037, a 0.0099 range.
Early stopping picks 103..224 trees where the baseline uses 400, so the baseline
is over-trained -- yet the 5-seed bagged reference (fold 2 0.5058) beats every
single-seed config in the grid. Bagging is worth more here than tuning.

| selection route | config | fold 2 | vs ref fold 2 |
| --- | --- | --- | --- |
| on fold 1 (honest, fold 2 left clean) | leaves 15, mcs 30, cols 0.4, n 164 | 0.4951 | -0.0107 |
| on fold 2 (selection-biased) | leaves 31, mcs 100, cols 0.7, n 103 | 0.5037 | +0.0021* |

\* not a check: it is the max of 18 configs measured on the same fold.
Re-bagged on both folds it gives 0.4984 / 0.5019, mean 0.5002 vs the reference
0.5006 -- **dropped**. The apparent +0.0021 collapsed to -0.0004 once the
selection bias was removed, which is what selection bias looks like.

Lesson recorded: fold 1 is the only legitimate selection set here, because early
stopping already spends it. Anything chosen on fold 2 must be re-measured before
it can be believed.

### A4 CatBoost (classifier) -- dropped, averaging not triggered
| variant | fold 2025-06 | fold 2025-09 | mean F1 | delta | kept |
| --- | --- | --- | --- | --- | --- |
| reference (LightGBM, 5-seed bag) | 0.4953 | 0.5058 | 0.5006 | - | reference |
| catboost (5-seed bag, depth 6) | 0.4931 | 0.4936 | 0.4933 | -0.0072 | dropped |

The |catboost - lgbm| mean F1 gap is 0.0072, outside the 0.005 window, so the
probability-averaging arm was correctly not attempted. CatBoost is also 154s per
17-class fit vs about 22s for LightGBM -- 7x the cost for a worse result, so it
is not worth revisiting on the classifier. (Its regressors are a different
story, 9s per fit and level with LightGBM: see B3.)

### Block A summary -- every training-recipe lever is flat
| lever | best variant | delta mean F1 | verdict |
| --- | --- | --- | --- |
| A1 recency weighting | half-life 12mo | +0.0001 | dropped |
| A2 snapshot spacing | every 2 months | +0.0002 | dropped |
| A3 capacity grid | leaves31 mcs100 cols0.7 | -0.0004 | dropped |
| A4 CatBoost | - | -0.0072 | dropped |

Nothing in the training recipe moves the classifier. Combined with sweep 1
(features, decision rules) the picture is that **17-class argmax weighted F1 is
pinned near 0.50** on this feature set regardless of weighting, data density,
capacity or library. What is left that could move it: changing what gets
predicted (blocks C and E), or genuinely new information (block F).

## Open questions
- Does higher or lower win on the leaderboard?
- Does public score track validation?
