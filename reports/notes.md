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

## Open questions
- Does higher or lower win on the leaderboard?
- Does public score track validation?
