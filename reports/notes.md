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

### B1 hurdle regression -- fails the stated bar, but wins on the score
| target | fold 2025-06 | fold 2025-09 | mean | delta | bar -0.005 |
| --- | --- | --- | --- | --- | --- |
| reference CLV_fuel | 0.5977 | 0.6033 | 0.6005 | - | - |
| hurdle CLV_fuel | 0.5975 | 0.5994 | 0.5984 | -0.00207 | not met |
| reference CLV_nonfuel | 0.7397 | 0.7474 | 0.7436 | - | - |
| hurdle CLV_nonfuel | 0.7352 | 0.7450 | 0.7401 | -0.00348 | not met |

P(y>0) times a regressor fitted on positive rows only. Better on all four
fold x target cells -- the first consistent improvement found in two sweeps --
but neither target reaches the -0.005 bar.

The mechanism is confirmed rather than assumed: non-fuel is about 65% zeros per
quarter and fuel about 29%, and the hurdle helps non-fuel (-0.0035) more than
fuel (-0.0021), ordered exactly by zero-inflation.

**Resolved: KEPT.** The per-target -0.005 bars were set before the score formula
was known and are replaced from here on by a single rule:

> KEEP if (1) mean combined score improves by >= 0.0015 over the current best,
> (2) the combined score improves on BOTH folds, and (3) no component (F1,
> rmse_f, rmse_nf) gets worse on either fold by more than 0.001.

B1 under that rule:
- (1) mean score gain +0.00212 >= 0.0015 -- pass
- (2) both folds up: fold1 +0.00175, fold2 +0.00249 -- pass
- (3) worst component change +0.00000 (nothing gets worse at all) -- pass

Re-judging every earlier variant under the new rule changes no other verdict:
a1/a2/a3/a4 and all of b2 remain dropped, each failing on the gain and on at
least one fold. B1 is the only kept change in sweep 2 so far, and the hurdle is
now the current best regression stack that B3 and D are measured on top of.

### B2 recency / spacing (regressions) -- all dropped, and it inverts A2
Positive delta = higher RMSE = worse.

| variant | CLV_fuel delta | CLV_nonfuel delta |
| --- | --- | --- |
| recency half-life 3mo | +0.00216 | +0.00255 |
| recency half-life 6mo | +0.00069 | +0.00135 |
| recency half-life 12mo | +0.00056 | +0.00087 |
| spacing every 2 months | +0.00224 | +0.00155 |
| spacing quarterly | +0.00261 | +0.00279 |

Recency repeats A1's monotone pattern on both regressions and converges to the
reference from above. **Recency weighting is settled across all three targets:
it only costs effective sample size. Stop testing it.**

Spacing inverts A2. Quarterly cost the classifier just 0.0006 F1 but costs the
regressions +0.0026 / +0.0028 RMSE: the regressions want MORE snapshots, the
classifier does not care. This matches the bagging result (regressions -0.0020 /
-0.0018, F1 only +0.0008). The theme across both sweeps is that variance
reduction helps the continuous targets and barely moves argmax.

Two consequences:
- **The A2-based prediction that block D would fail was wrong**, because A2 only
  measured the classifier. Revised: D should be flat for the classifier but may
  genuinely help the regressions.
- **Block F runs quarterly**, which costs the regressions ~0.0026 / ~0.0028.
  Comparing F variants against the quarterly reference is still sound, but any F
  group that helps must be re-measured at monthly spacing before it goes into a
  submission.

### B3 CatBoost on top of the hurdle -- dropped; gains stack but decay fast
Measured on top of the kept hurdle, not the original reference.

| variant | rmse_fuel | rmse_nonfuel | score delta | kept |
| --- | --- | --- | --- | --- |
| b1 hurdle [current best] | 0.5984 | 0.7401 | - | reference |
| hurdle + catboost magnitude | 0.5974 (-0.00100) | 0.7391 (-0.00093) | +0.00075 | dropped |
| hurdle lgbm/catboost average | 0.5977 (-0.00072) | 0.7395 (-0.00057) | +0.00050 | dropped |

Both improve all four fold x target cells but reach under half the 0.0015 bar.

**Do gains stack? Yes, with steep diminishing returns.** The hurdle gave +0.00212
over the reference; CatBoost on top adds +0.00075. Zero-inflation was the
dominant effect and the choice of magnitude model is secondary.

Note for any future regression work: CatBoost magnitude alone (+0.00075) beats
the LightGBM/CatBoost average (+0.00050), so CatBoost's magnitude model is
genuinely better on positive rows rather than merely decorrelated -- averaging
dilutes it, which is the opposite of the usual rationale for averaging.

### C adoption binary + oracle relabel -- DROP THE IDEA (definitive)
| fold | AUC | base F1 | best oracle F1 | gain | optimal k |
| --- | --- | --- | --- | --- | --- |
| 2025-06 | 0.7396 | 0.4953 | 0.4960 | +0.0008 | 50 |
| 2025-09 | 0.7500 | 0.5058 | 0.5070 | +0.0012 | 75 |

Both folds under the 0.003 kill-switch, so the idea is dropped as agreed.

This is conclusive rather than merely negative. **AUC 0.745 means the ranking
works** -- a binary "will adopt something" model genuinely discriminates, so this
is not a modelling failure. But the ORACLE relabel, which picks k with full
sight of the validation labels and therefore cannot be beaten by any realisable
method, gains only +0.001 F1 (+0.0004 score). The oracle also chose to relabel
just 50-75 of about 5,000 customers, i.e. it found that relabelling more costs
more than it gains.

Mechanism: adoption is ~7% of support split across 11 classes, so a correct
prediction must be right twice (that they adopt at all, and which of 11), while
every relabel immediately costs precision on Stable and Inactivity at ~27%
support each. Weighted F1 cannot pay for adoption recall at these base rates.

**Do not spend more time on the adoption classes.** Any future attempt needs a
fundamentally different signal, not a better classifier or threshold.

### Where the F1 gap actually is -- revised
Sweep 1 guessed adoption was the largest unclaimed block (~7% of weight at F1
0.000). C shows that block is unreachable from these features. Reconstructing
1st place's 0.557 without adoption requires the big three classes to carry it:
Fuel growth 0.454 -> 0.60 is worth about +0.040 weighted and Stable 0.574 -> 0.65
about +0.020. So the race is **Fuel growth vs Stable discrimination**, which is
also exactly where the confusion matrix has always been worst (432 of 1435 true
Fuel growth predicted Stable, 362 of 1404 true Stable predicted Fuel growth).

### E rule-component classifiers -- both dropped
| variant | fold 2025-06 | fold 2025-09 | mean F1 | delta | kept |
| --- | --- | --- | --- | --- | --- |
| reference | 0.4953 | 0.5058 | 0.5006 | - | reference |
| e1 stacked event probabilities | 0.4920 | 0.5062 | 0.4991 | -0.0014 | dropped |
| e2 events through a simple rule | 0.4515 | 0.4492 | 0.4504 | -0.0502 | dropped |

Four binary rule events (inactive, Fuel qualifies as growth, any non-fuel
category qualifies, any adoption candidate), trained on labels derived from
label_rules and fed in as out-of-fold features built strictly forward in time.

e1 is flat-to-slightly-negative: fold 2 gains 0.0004 while fold 1 loses 0.0033.
The forward-in-time construction is part of the cost -- the earliest snapshots
have no earlier outcomes to train the event models on, so those rows carry NaN
event features, and fold 1 has fewer usable snapshots than fold 2. That the
better fold is the one with more event coverage is at least consistent, but the
effect is too small to act on.

e2 is much worse (-0.05), which is expected: collapsing to five hard outcomes
throws away every adoption sub-class and both minor growth classes.

**Combined with C, this closes the "change what gets predicted" route.** C showed
the adoption signal exists (AUC 0.745) but cannot be monetised. E shows that
handing the classifier an explicit "will Fuel qualify as growth" probability
does not move the Fuel-growth/Stable boundary either. The boundary is not
reachable by re-routing these features, so block F's new information is the only
remaining lever.

### D cutoff density, decided per target -- monthly wins all three
| target | fortnightly | weekly | winner |
| --- | --- | --- | --- |
| Opportunity (mean F1) | -0.0020 | -0.0017 | monthly |
| CLV_fuel (mean RMSE) | +0.00082 | +0.00103 | monthly |
| CLV_nonfuel (mean RMSE) | +0.00145 | +0.00079 | monthly |

**Monthly is a genuine optimum, not an endpoint.** Combining B2 and D within each
recipe:
- thinning (B2, direct): monthly 0.6005 -> quarterly 0.6031, +0.0026 worse
- densifying (D, hurdle): monthly 0.5984 -> weekly 0.5994, +0.0010 worse

**My prediction that weekly would help the regressions was wrong**, and the error
is instructive: B2 showed thinning hurt them and I extrapolated that densifying
would help. That is invalid across an optimum.

Likely mechanism: consecutive weekly cutoffs share about 97% of their outcome
window, so extra snapshots add no information while re-weighting the training
distribution toward long-tenured customers who are eligible in more snapshots --
at 4x the compute. Duplicating near-identical rows biases the sample rather than
enriching it.

The per-target rule selected the same spacing for all three targets, but the
targets were free to disagree and the answer is now measured rather than assumed.
**The "more training data" lever is closed in both directions.**

## Sweep 2 complete -- 1 kept of 17 variants
| block | best delta score | verdict |
| --- | --- | --- |
| A1 recency weighting | +0.00004 | dropped |
| A2 snapshot spacing | +0.00009 | dropped |
| A3 capacity grid | -0.00015 | dropped |
| A4 CatBoost classifier | -0.00289 | dropped |
| **B1 hurdle regressions** | **+0.00212** | **KEPT** |
| B2 recency / spacing (reg) | -0.00055 | dropped |
| B3 CatBoost regressions | +0.00075 | dropped |
| C adoption oracle relabel | +0.00039 | dropped |
| D cutoff density | -0.00029 | dropped |
| E rule-component classifiers | -0.00058 | dropped |

Score ladder: single-seed 0.28164 -> 5-seed bagged 0.28342 (+0.00178) ->
bagged + hurdle 0.28553 (+0.00212). Total +0.00389, about 18% of the 0.02117
public gap to 1st.

Both wins are regression-side and both are variance-reduction. **F1 has not
moved from ~0.50 across 13 attempts** spanning features, decision rules, sample
weighting, data density, model capacity, library choice, class priors,
two-stage structure, rule derivation, rule-event stacking and adoption
relabelling. The classifier ceiling on these features is real.

## Block F -- new information (6 Oct), quarterly spacing, all groups dropped

Quarterly reference (block F base) 0.28108 vs the monthly reference 0.28342:
quarterly costs 0.00234 score, almost all of it on fold 1 (10 monthly snapshots
drop to 4, against fold 2's 13 to 5). Building a separate quarterly reference was
necessary -- measured against monthly, every F group would have looked ~0.0023
worse than it is and all six would have been dropped spuriously.

| group | fold 1 score | fold 2 score | mean | delta vs ref | kept |
| --- | --- | --- | --- | --- | --- |
| reference quarterly | 0.2790 | 0.2831 | 0.28108 | - | reference |
| f1 customer id | 0.2775 | 0.2834 | 0.28045 | -0.00064 | dropped |
| f2 sites | 0.2772 | 0.2810 | 0.27909 | -0.00200 | dropped |
| f3 fuel type/price | 0.2787 | 0.2837 | 0.28123 | +0.00015 | dropped |
| f4 timing | 0.2796 | 0.2837 | 0.28165 | +0.00057 | dropped |
| f5 vouchers/discounts | 0.2782 | 0.2821 | 0.28015 | -0.00093 | dropped |
| f6 combined (f3+f4) | 0.2791 | 0.2826 | 0.28084 | -0.00024 | dropped |

### F1 alone (customer id)
| metric | fold 2025-06 | fold 2025-09 | mean | delta |
| --- | --- | --- | --- | --- |
| weighted F1 | 0.4895 | 0.5069 | 0.4982 | -0.00177 |
| rmse fuel | 0.6037 | 0.6034 | 0.6035 | +0.00044 |
| rmse nonfuel | 0.7440 | 0.7473 | 0.7457 | -0.00069 |
| combined score | 0.2775 | 0.2834 | 0.28045 | -0.00064 |

The ID diagnostic was real but redundant. Inactivity by ID decile spans
0.147..0.373 and ID correlates with first-seen date at Pearson 0.543, yet adding
the two columns costs 0.0018 F1. `tenure` and `hist_days` already carry the
"new customer" part, and deciles 0-6 are left-censored so their ID order holds
no date information at all -- only deciles 7-9 are genuinely later sign-ups.
A strong marginal association with the target is not the same as information the
model does not already have.

### Per-component split -- the interesting part
| group | delta F1 | delta rmse_fuel | delta rmse_nonfuel |
| --- | --- | --- | --- |
| f1 customer id | -0.00177 | +0.00044 | -0.00069 |
| f2 sites | -0.00460 | +0.00109 | -0.00079 |
| f3 fuel type/price | -0.00167 | **-0.00068** | **-0.00147** |
| f4 timing | -0.00123 | **-0.00072** | **-0.00210** |
| f5 vouchers/discounts | -0.00127 | +0.00051 | +0.00060 |
| f6 combined (f3+f4) | -0.00335 | -0.00070 | -0.00222 |

**Every group hurts F1, without exception.** f3 and f4 genuinely help both
regressions, and their positive combined score is a tug-of-war the classifier
loss wins.

Falsification check: **f6 is worse than f3 or f4 alone** (+0.00015, +0.00057 ->
-0.00024). Independent signal would roughly add; going negative says the
classifier dilution compounds faster than the regression gains accumulate.

f2 is the biggest classifier casualty (-0.0046) despite site inactivity rates
spanning 0.025..0.488 across 382 sites. `home_site` as a 390-level numeric code
is the likely culprit -- a high-cardinality identifier LightGBM can split
arbitrarily on, which is exactly how to overfit a flat signal.

### F-REG: f3+f4 on the regressions only, monthly -- DROPPED, no v3
Run 6 Oct on top of the kept hurdle, classifier left on base features so
touches=('rf','rn') and condition (3) ignores F1.

| | fold 2025-06 | fold 2025-09 | mean | delta |
| --- | --- | --- | --- | --- |
| hurdle [current best] score | 0.2856 | 0.2855 | 0.28553 | - |
| freg f3+f4 score | 0.2869 | 0.2850 | 0.28597 | +0.00044 |
| rmse fuel | 0.5957 | 0.6005 | 0.5981 | -0.00028 |
| rmse nonfuel | 0.7336 | 0.7448 | 0.7392 | -0.00088 |

Fails all three conditions: gain +0.00044 < 0.0015; fold 2 score down
(0.2850 vs 0.2855); and rmse_fuel on fold 2 worsens by 0.0011, over the 0.001
component slack.

**My quarterly extrapolation of +0.0011 was optimistic.** The regression gains
shrink with more data: rf -0.00070 -> -0.00028 and rn -0.00222 -> -0.00088 going
from quarterly to monthly. f3/f4 were partly compensating for quarterly's data
scarcity, and with the full monthly training set the base hurdle is already
strong enough that the extra columns mostly stop mattering. Removing the
quarterly penalty did not help because it applied to base and variant alike and
largely cancelled. Same diminishing-returns shape as B3 (+0.00212 -> +0.00075)
and D.

**submission_v2 (0.28553) remains the best stack. No v3 was built.**

### The original lead as stated before it was run
Your per-target principle extends from spacing to feature sets: a group could be
given to the regressions only, leaving the classifier on base features. The
arithmetic on f6's regression effects is
0.3*0.00070/0.74 + 0.3*0.00222/0.816 = 0.0011, still under the 0.0015 bar, so it
does not pass even in its best framing. But that is measured at quarterly, which
penalises the regressions by ~0.0026, so a monthly re-measurement of
"f3+f4 on the regressions only" is the single untested combination that could
plausibly cross. Nothing was kept, so no re-measurement was required.

### Probe files (diagnostic, not submitted)
`probe_stable.csv`, `probe_inactivity.csv`, `probe_fuelgrowth.csv`: every row
labelled with one constant class, CLV columns from the bagged reference
regressions (identical across all three files). Each validated at 5,488 rows,
IDs matching data/test.csv, a single label present and in the config, and both
CLV columns >= 0.

## Block G -- seasonal analog weighting (6 Oct), all 7 variants dropped

Analog = the training snapshot exactly 12 months before the cutoff. History
behind each analog: fold 1 (2024-06-01) 73 days, fold 2 (2024-09-01) 165 days,
test (2024-12-01) 256 days. Fold 1's analog is also one-sided -- 2024-05-01 is
outside our cutoff set -- so G3 trains on 2 snapshots for fold 1 against 3 for
fold 2. The established cutoff set was left unchanged to keep every earlier
block comparable.

### Opportunity (vs bagged reference 0.5006)
| variant | fold 2025-06 | fold 2025-09 | mean F1 | delta | kept |
| --- | --- | --- | --- | --- | --- |
| incumbent | 0.4953 | 0.5058 | 0.5006 | - | reference |
| g1 analog x3 | 0.4971 | 0.5034 | 0.5003 | -0.00029 | dropped |
| g1 analog x10 | 0.4956 | 0.4962 | 0.4959 | -0.00461 | dropped |
| g1 analog x30 | 0.4926 | 0.4864 | 0.4895 | -0.01109 | dropped |
| g2 analog+near x3 | 0.4935 | 0.5029 | 0.4982 | -0.00234 | dropped |
| g2 analog+near x10 | 0.4978 | 0.4977 | 0.4978 | -0.00278 | dropped |
| g3 analog+near only | 0.4529 | 0.4632 | 0.4580 | -0.04251 | dropped |
| g4 season-analog features | 0.4960 | 0.5056 | 0.5008 | +0.00023 | dropped |

### CLV_fuel (vs kept hurdle 0.5984) and CLV_nonfuel (vs 0.7401)
| variant | rmse_fuel | delta | rmse_nonfuel | delta |
| --- | --- | --- | --- | --- |
| g1 analog x3 | 0.5982 | -0.00018 | 0.7396 | -0.00050 |
| g1 analog x10 | 0.5982 | -0.00026 | 0.7402 | +0.00011 |
| g1 analog x30 | 0.6022 | +0.00377 | 0.7423 | +0.00221 |
| g2 analog+near x3 | 0.5979 | -0.00048 | 0.7390 | -0.00107 |
| g2 analog+near x10 | 0.6006 | +0.00214 | 0.7397 | -0.00039 |
| g3 analog+near only | 0.6428 | +0.04434 | 0.7839 | +0.04378 |
| g4 season-analog features | 0.5982 | -0.00024 | 0.7401 | +0.00006 |

### The dose-response is the real result
On the classifier, harm scales monotonically with the analog weight:
x3 -0.00029, x10 -0.00461, x30 -0.01109. If the analog snapshot carried extra
seasonal signal, up-weighting it would help at *some* dose. Instead every
increase makes things worse, in order. That is a clean refutation rather than a
null: the analog is just another snapshot, and weighting it up only destroys
effective sample size.

### Which fold to trust -- and it inverts the expected caveat
The brief warned that fold 1's thin analog would make fold 1 understate the
idea. The data runs the other way. Classifier deltas by fold:

| variant | fold 1 (73-day analog) | fold 2 (165-day analog) |
| --- | --- | --- |
| g1 analog x3 | +0.0018 | -0.0024 |
| g1 analog x10 | +0.0003 | -0.0096 |
| g1 analog x30 | -0.0027 | -0.0194 |
| g2 analog+near x3 | -0.0018 | -0.0029 |
| g2 analog+near x10 | +0.0025 | -0.0081 |

Fold 1 is noisy and mildly positive; fold 2 is consistently negative and
monotone in dose. **The better-supported analog is the one that rejects the idea
more firmly**, so fold 1 overstates rather than understates here. The test
analog has 256 days behind it, more than fold 2's 165, and the trend across
73 -> 165 days of analog history points further negative, not positive. I trust
fold 2, and the conclusion is stronger for it.

### G3 and G4
G3 (train only on analog + near-analogs) is catastrophic across all three
targets: F1 -0.0425, rmse_fuel +0.0443, rmse_nonfuel +0.0438. Expected at 2-3
training snapshots instead of 10-13, and it re-confirms D's finding that
training-set size dominates any seasonal alignment.

G4 is flat everywhere (F1 +0.00023, rmse_fuel -0.00024, rmse_nonfuel +0.00006)
for the reason flagged before running it: its "same calendar quarter one year
earlier" window only exists from cutoff 2025-04-01 onward, so it is NaN in all
10 of fold 1's training snapshots and present in only 3 of fold 2's 13, while
being populated at both validation cutoffs. Sweep 1's variant (c) failed at
-0.0037 by the same mechanism. `cut_quarter` is also near-collinear with the
existing `cut_month`. Seasonality cannot be learned from 20 months of history;
it needs a second year.

### Diagnostic submissions (not submitted)
Both keep v2's CLV columns byte-identical by construction (copied, then
asserted equal) and change only Opportunity.
- `submission_g_dec10.csv`: all 16 snapshots, analog 2024-12-01 weighted x10.
  Agrees with v2 on 90.09% of customers. Mix: Stable 0.399, Inactivity 0.322,
  Fuel growth 0.221.
- `submission_g_dec_only.csv`: classifier trained only on 2024-11-01,
  2024-12-01, 2025-01-01. Agrees with v2 on 82.65%. Mix: Stable 0.402,
  Inactivity 0.338, Fuel growth 0.201.

Both shift the mix toward Stable and Inactivity and away from Fuel growth, which
is the direction validation says is wrong -- Fuel growth is the class the model
already under-predicts. On validation the equivalent configurations score
-0.0046 (x10) and -0.0425 (analog-only) on F1, so neither is a candidate.

## Open questions
- Does higher or lower win on the leaderboard?
- Does public score track validation?
