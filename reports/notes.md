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

Provenance (updated 7 Oct): the normalisation **mechanism** is confirmed -- Zindi
staff confirmed it in the discussion "Clarification Request: Leaderboard Public
Score Discrepancy" (25 Sep 2026) and the Info page now reads "normalised RMSE ...
normalised Weighted F1". The weights 0.3 / 0.3 / 0.4 are official. The two
**constants** 0.74 and 0.816 remain unpublished: they were backed out here from
four public leaderboard rows (nine-decimal agreement), another participant in
that thread independently recovered the same values, and Zindi's multi-metric
policy makes the starter notebook's RMSE scores the likely source. Label:
inferred constants, confirmed mechanism. All verdicts are also reported per
component so none of them rests on the constants.

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

## Consolidation (6 Oct)

### 1. One reproducible entry point
`src/make_submission.py` goes from `data/train.csv` + `data/test.csv` to a
submission with no cached artifacts (all 17 snapshots built in memory), seeds
fixed, printing per-fold and mean validation. It reproduces
`submissions/submission_v2.csv` **byte-for-byte** (`cmp` clean, not merely within
1e-9) and prints the recorded numbers exactly: fold1 0.28561, fold2 0.28545,
mean 0.28553. **Runtime 646 s** on 12 logical cores.

### 2. Seed stability
Five 5-seed builds, each the v2 stack on a different disjoint seed set:

| seeds | fold 2025-06 | fold 2025-09 | mean score |
| --- | --- | --- | --- |
| 42-46 (= v2) | 0.28561 | 0.28545 | **0.28553** |
| 47-51 | 0.28538 | 0.28520 | 0.28529 |
| 52-56 | 0.28542 | 0.28536 | 0.28539 |
| 57-61 | 0.28519 | 0.28504 | 0.28511 |
| 62-66 | 0.28510 | 0.28483 | 0.28496 |

Spread: range 0.00057, sd 0.00020, mean 0.28526. **v2 is the maximum of the five
draws**, so the 0.28553 headline sits at the optimistic end of what a 5-seed bag
delivers. Seeds 42-46 were fixed before any measurement, so this is not
post-hoc selection -- but it does mean 0.28553 should not be read as the expected
score of the approach, which is about 0.28526.

Opportunity flip rate between 5-seed builds: **2.91%** of test customers (10
pairs, 2.73-3.01%).

### 20-seed bag (42-61)
| | F1 | rmse fuel | rmse nonfuel | score |
| --- | --- | --- | --- | --- |
| fold 2025-06 | 0.4937 | 0.5969 | 0.7350 | 0.28524 |
| fold 2025-09 | 0.5045 | 0.5996 | 0.7451 | 0.28481 |
| mean | 0.4991 | - | - | **0.28502** |

Flip rate between two **disjoint** 20-seed bags (42-61 vs 62-81): **1.42%**,
51% lower than the 5-seed rate -- almost exactly the 1/sqrt(4) expected from
averaging four times as many seeds.

**Verdict: bag20 fails the keep rule, by 0.00001.** The gap to v2 is -0.00051
against a 0.0005 window. Two considerations pull opposite ways and both are
recorded rather than resolved in favour of the convenient one:
- *Against the strict reading*: v2 is the luckiest of five draws. Against the
  expected 5-seed score (0.28526) bag20 is only -0.00023, inside the window.
- *Supporting it*: bag20's F1 is systematically low, not just below the lucky
  draw. Fold-2 F1 across the five builds is 0.5058/0.5058/0.5061/0.5056/0.5044
  and bag20 gives 0.5045 -- below four of five. Its regressions are equal or
  marginally better. So heavier bagging genuinely lowers weighted F1, the same
  asymmetry seen throughout: variance reduction helps the continuous targets and
  pulls argmax toward majority classes.

Recommendation: keep it as the conservative candidate -- the cost is about 2.5
seed-sd and the flip rate halves -- but it is a judgement call, not a rule pass.

### 3. Class-prior adjustment -- the first F1 gain in the project
Fitted on bag20's out-of-fold probabilities with 3 free weights (Stable, Fuel
growth, growth:Other), each fold fitted independently and scored only on the
other.

| fit on | Stable | Fuel | Other | held-out F1 |
| --- | --- | --- | --- | --- |
| 2025-06 | 1.10 | **1.25** | 2.00 | fold 2025-09: 0.5045 -> 0.5081 (+0.0036) |
| 2025-09 | 0.80 | **1.25** | 1.00 | fold 2025-06: 0.4937 -> 0.4937 (+0.0000) |

**Only the Fuel-growth weight replicated.** Stable came out on opposite sides of
1.0 in the two fits (1.10 vs 0.80) and growth:Other moved 2.00 vs 1.00, so both
are fold-specific noise. Keeping only the replicated parameter *improves* the
held-out result, which is what should happen if the others were noise:

| weights | fold 2025-06 | fold 2025-09 | mean F1 |
| --- | --- | --- | --- |
| bag20, none | 0.4937 | 0.5045 | 0.4991 |
| full 3-parameter, held out | 0.4937 (+0.0000) | 0.5081 (+0.0036) | 0.5009 (+0.0018) |
| **Fuel x1.25 only, held out** | **0.4968 (+0.0031)** | **0.5096 (+0.0051)** | **0.5032 (+0.0041)** |

Both Fuel-only numbers are genuinely held out: each fold's own fit chose 1.25
independently, and each is scored on the other fold. Under the standing rule,
measured on bag20: score 0.28502 -> **0.28667**, +0.00165 (clears 0.0015), both
folds up (0.28650 / 0.28685), only F1 touched. **KEPT.**

Why this works where 21 earlier F1 attempts failed: it is not a model change at
all. The confusion matrix has said from the start that Fuel growth is
under-predicted (1,209 predicted against 1,435 true on fold 2), and this moves
exactly that decision boundary by one number. Every earlier attempt tried to give
the model better information or a different structure; this one accepts the
model's ranking and corrects its threshold.

### Candidate summary
| File | Validation score | Mean F1 | Note |
| --- | --- | --- | --- |
| `submission_baseline_v1.csv` | 0.28164 | 0.4998 | single seed, reference |
| `submission_v2.csv` | 0.28553 | 0.5006 | 5-seed bag + hurdle; public 0.2991 |
| `submission_v2_bag20.csv` | 0.28502 | 0.4991 | 20-seed, flip rate 1.42% vs 2.91% |
| **`submission_v2_prior.csv`** | **0.28667** | **0.5032** | bag20 + Fuel x1.25, cross-fold validated |

`submission_v2_prior.csv` is both the highest-scoring and the most stable
candidate: it inherits bag20's halved flip rate and adds a one-parameter
adjustment validated on each fold by the other.

## Block H -- decision calibration (6 Oct). Largest gain since the hurdle.

All of block H works on saved bag20 probabilities: per-class multiplicative
weights applied before the argmax, no model refitting. Base is bag20 + Fuel
growth x1.25 (score 0.28668, F1 0.4968 / 0.5096).

**Class mix drifts monotonically**, which shapes every result here. Inactivity
rises in 16 of 16 snapshots (0.0725 at 2024-06 to 0.2742 at 2025-09) while Stable
falls (0.3654 to 0.2680) and adoption halves (0.1395 to 0.0703). So any target
mix taken from the last fully observed snapshot is stale by construction:

| fold | target from | target Inactivity | true Inactivity | 4-class drift |
| --- | --- | --- | --- | --- |
| 2025-06-01 | 2025-03-01 | 0.222 | 0.258 | 0.106 |
| 2025-09-01 | 2025-06-01 | 0.258 | 0.274 | 0.053 |

### H1 Fuel-growth weight curve (nothing fitted, just evaluated)
| weight | fold 2025-06 | fold 2025-09 | mean F1 |
| --- | --- | --- | --- |
| 1.000 | 0.4937 | 0.5045 | 0.4991 |
| 1.125 | 0.4966 | **0.5099** | **0.5033** |
| 1.250 | **0.4968** | 0.5096 | 0.5032 |
| 1.375 | 0.4957 | 0.5070 | 0.5014 |
| 1.500 | 0.4940 | 0.5053 | 0.4996 |
| 1.625 | 0.4884 | 0.5045 | 0.4965 |
| 1.750 | 0.4828 | 0.5001 | 0.4914 |
| 1.875 | 0.4764 | 0.4952 | 0.4858 |
| 2.000 | 0.4644 | 0.4887 | 0.4765 |

Peak at 1.25 on fold 1 and 1.125 on fold 2 -- **adjacent grid points, so the two
folds agree on the location to within one step**. The optimum is a plateau: mean
F1 is within 0.001 of the peak across 1.125-1.250, then falls away steeply
(-0.027 by x2.0). That asymmetry is useful: over-weighting is far more dangerous
than under-weighting, so 1.25 sits safely on the flat part. Nothing clears the
bar because the base already *is* the plateau.

### H2 prior matching by IPF, no fitted parameters -- all dropped
| variant | fold 2025-06 | fold 2025-09 | mean F1 | delta |
| --- | --- | --- | --- | --- |
| all 17 classes | 0.4933 | 0.5080 | 0.5007 | -0.0026 |
| 4 large only | 0.4961 | 0.5100 | 0.5030 | -0.0002 |
| all 17, mean of last 3 | 0.4923 | 0.5075 | 0.4999 | -0.0033 |

IPF matches the target mix essentially exactly (mix error 0.000-0.005), so the
method works mechanically -- it just does not help. Matching a stale target
moves the predictions toward last quarter's proportions, and the drift means
that is the wrong place to move them.

### H3 partial matching -- 5 of 12 variants KEPT, best result in the project
Weights from H2 raised to a power alpha.

| variant | fold 2025-06 | fold 2025-09 | mean F1 | score | delta | kept |
| --- | --- | --- | --- | --- | --- | --- |
| 4 large, alpha 0.25 | 0.4984 | 0.5129 | 0.5056 | 0.28765 | +0.00097 | dropped |
| **4 large, alpha 0.5** | **0.5022** | **0.5173** | **0.5098** | **0.28929** | **+0.00262** | **KEPT** |
| 4 large, alpha 0.75 | 0.4999 | 0.5156 | 0.5078 | 0.28851 | +0.00183 | KEPT |
| 4 large, alpha 1.0 | 0.4961 | 0.5100 | 0.5030 | 0.28661 | -0.00007 | dropped |
| all 17, alpha 0.5 | 0.5028 | 0.5153 | 0.5090 | 0.28901 | +0.00233 | KEPT |
| all 17, alpha 0.75 | 0.5001 | 0.5145 | 0.5073 | 0.28831 | +0.00163 | KEPT |
| last-3, alpha 0.5 | 0.5017 | 0.5150 | 0.5084 | 0.28874 | +0.00206 | KEPT |

**alpha = 0.5 is cross-fold validated**: each fold's own alpha search picked 0.5
independently, so scoring 0.5 on the other fold is a genuine held-out test, and
both held-out folds improve (+0.0054 on fold 1, +0.0077 on fold 2). There is a
clear interior optimum -- 0.5032 at no correction, 0.5056 at 0.25, 0.5098 at 0.5,
0.5078 at 0.75, 0.5030 at 1.0.

Why an interior optimum rather than full matching? Not only staleness. Forcing
the predicted marginal to equal *any* target costs accuracy when the model's
ranking is imperfect, so the best point trades calibration against ranking. Full
matching (alpha=1) gives back the entire gain even though it hits the target mix
most precisely -- matching the mix is not the objective, weighted F1 is.

Residual selection to be honest about: alpha is cross-fold validated, but the
choice among the three target variants was made by comparing both folds. The
three best-alpha variants score 0.28929 / 0.28901 / 0.28874 -- a 0.00055 spread,
and all three pass the rule -- so that choice is low-stakes.

### H4 one-at-a-time weights on top of Fuel x1.25 -- nothing kept
| class | fold-1 fit | fold-2 fit | same direction | both held-out folds up |
| --- | --- | --- | --- | --- |
| Stable | 0.90 | 0.80 | yes | no |
| Inactivity | 1.00 | 1.00 | n/a (no change wanted) | no |
| growth:Other | 2.00 | 1.25 | yes | no |

Stable and growth:Other agreed on direction but neither improved both held-out
folds, and Inactivity's search wanted no change at all. **Yet H3, which moves all
four large classes jointly, gains +0.0066 mean F1.** The adjustment is genuinely
joint: shifting Stable down only helps when Inactivity and Fuel move at the same
time, which one-at-a-time search cannot find.

### submission_v3.csv
Test-time rule, no labels from the thing being predicted: IPF the test
probabilities to the observed 2025-09-01 class mix over the four large classes,
raise those weights to 0.5, argmax. Regressions are bag20's, unchanged.

| | v3 predicted | 2025-09 observed |
| --- | --- | --- |
| Stable | 0.333 | 0.268 |
| Inactivity | 0.295 | 0.274 |
| Fuel growth | 0.259 | 0.274 |
| growth:Other | 0.060 | 0.053 |
| adoption (all) | 0.006 | 0.070 |

Adoption stays near zero because only the four large classes are matched, which
is consistent with block C: the adoption classes are unmonetisable under
weighted F1 even with an oracle.

Note on the CLV columns: they match bag20 to within 1 ULP (max 4.4e-16, 90 of
5,488 rows). In memory they are exactly equal and that is asserted before
writing; the difference is pandas' default CSV float formatting not
round-tripping the final bit. The official `label_rules.py` writes with
`float_format='%.17g'` for exactly this reason. Irrelevant at scoring precision.

### Candidate summary after block H
| File | Score | Mean F1 | Note |
| --- | --- | --- | --- |
| `submission_baseline_v1.csv` | 0.28164 | 0.4998 | single seed |
| `submission_v2.csv` | 0.28553 | 0.5006 | 5-seed + hurdle; public 0.2991 |
| `submission_v2_bag20.csv` | 0.28502 | 0.4991 | 20-seed, flip rate 1.42% |
| `submission_v2_prior.csv` | 0.28668 | 0.5032 | bag20 + Fuel x1.25 |
| **`submission_v3.csv`** | **0.28929** | **0.5098** | bag20 + prior matching at alpha 0.5 |

v3 is +0.00376 over v2 and +0.00262 over v2_prior. Every gain in the project now
comes from three places: seed bagging, the hurdle regressions, and decision
calibration. No feature, model or structural change ever helped.

## Block I -- better targets for partial prior matching (7 Oct). Nothing passes.

Incumbent: H3, stale target, 4 large classes, alpha 0.5, score 0.28929.

### I1 projected target mixes: they ARE more accurate
Sum of absolute error against the TRUE fold mix over the four large classes:

| method | fold 2025-06 | fold 2025-09 |
| --- | --- | --- |
| stale (last observed) | 0.106 | 0.053 |
| linear over last 4 | 0.123 | 0.079 |
| linear over last 6 | 0.075 | 0.055 |
| last + mean QoQ change | **0.044** | 0.059 |
| damped (half the linear step) | 0.082 | **0.050** |

On fold 1 the quarter-on-quarter projection cuts target error by 58%
(0.106 -> 0.044). The projections do what they were meant to do.

### I2 ...and every one of them scores WORSE
Best alpha per target, against the incumbent 0.28929:

| target | best alpha | mean F1 | score | delta | cross-fold alpha |
| --- | --- | --- | --- | --- | --- |
| stale | 0.5 | 0.5098 | 0.28929 | - | 0.5 / 0.5 AGREE |
| lin4 | 0.5 | 0.5079 | 0.28854 | -0.00075 | 0.5 / 0.5 AGREE |
| lin6 | 0.75 | 0.5085 | 0.28877 | -0.00052 | 0.75 / 0.75 AGREE |
| qoq | 0.75 | 0.5078 | 0.28852 | -0.00078 | 0.75 / 0.5 DISAGREE |
| damped | 0.75 | 0.5102 | 0.28946 | +0.00017 | 0.75 / 0.5 DISAGREE |

All dropped. `damped` at alpha 0.75 is nominally +0.00017 but its cross-fold
alpha disagreed (0.75 against 0.5), so it is not selectable, and the gain is an
eighth of the bar in any case.

**The alpha hypothesis is confirmed**: a better target does push the optimum
toward 1.0 (stale picks 0.5; lin6, qoq and damped all pick 0.75). The mechanism
behaves exactly as predicted. It just does not convert into F1.

**What this settles.** Matching the true class mix is not why prior matching
works. The stale target wins *despite* being the least accurate of the five, and
the three most accurate targets all score below it. So alpha is not a staleness
correction -- it is trading the model's ranking against its marginal, and the
best trade sits well short of full matching no matter how good the target is.
Improving the target is therefore the wrong axis; this line is closed.

### I3 calibration before matching -- dropped
A single global temperature is algebraically identical to alpha:

    argmax_c p_c^(1/T) * w_c = argmax_c [(1/T) log p_c + log w_c]
                             = argmax_c [log p_c + T log w_c] = argmax_c p_c * w_c^T

so temperature T with weights w is exactly weights w**T, which is the alpha
mechanism. Demonstrated rather than fitted: T=2.0 with alpha=0.5 gives F1
0.4976 / 0.5046, identical to T=1.0 with alpha=1.0. There is nothing to gain
here that the alpha grid has not already searched.

Isotonic (one monotone map per class, fitted on an earlier fully observed
snapshot) is genuinely different and all five variants are dropped, from -0.00138
(lin6) to -0.00376 (lin4).

### I4 distance to the external reference shares -- report only
Computed after selection was frozen, and it corroborates the choice rather than
contradicting it.

| method | Stable | Inactivity | Fuel | sum abs err |
| --- | --- | --- | --- | --- |
| REFERENCE | 0.276 | 0.297 | 0.252 | - |
| **stale** | 0.268 | 0.274 | 0.274 | **0.053** |
| damped | 0.255 | 0.283 | 0.278 | 0.060 |
| qoq | 0.238 | 0.312 | 0.262 | 0.062 |
| lin6 | 0.236 | 0.300 | 0.275 | 0.066 |
| lin4 | 0.243 | 0.292 | 0.282 | 0.068 |

The stale target is the CLOSEST to the reference and every projection is
further away: the drift did not continue linearly into the test quarter --
reference Stable 0.276 is *above* the 2025-09 observation of 0.268, reversing
the trend the projections extrapolated. Same failure mode as on validation.

Caveats on these three numbers, which matter for how much weight they carry:
- They describe the **public split only**, roughly 1,650 customers. Sampling
  error at p=0.28 is about +/-0.011, so the 0.053-vs-0.060 gap is not decisive.
- They were used for a distance report and nothing else. Selection ran on the
  two folds and was frozen before I4 was computed, by construction in
  `src/project_mix.py`. Letting them choose a variant would be fitting to the
  public test set, which is the standard route to a public-to-private collapse.

## The 1-ULP CSV issue -- the real cause was the READER, not the writer

The requested fix was `float_format='%.17g'` on write. That was applied to all
ten CSV writers (label_rules.py already had it), but **it was not the cause**.

Writing was already correct: pandas' default `to_csv` emits
`1.9039983171278114`, which is the exact shortest round-tripping text. The loss
is in `read_csv`, whose default float parser is fast but not correctly rounded.
Measured: writing a value and reading it back is inexact under
`float_precision=None` **and** under `'high'`, and exact only under
`'round_trip'`.

So every read-then-rewrite of a submission shifted CLV values by 1 ULP. That is
how `submission_v3.csv` and `submission_v2_prior.csv` came to differ from
`submission_v2_bag20.csv` by 4.4e-16 despite being built by copying its columns.

Fixed by adding `float_precision='round_trip'` to every submission reader in
`src/`, and the two derived files were regenerated. All ten files in
`submissions/` are now read-write byte-identical, and v3 and v2_prior have CLV
columns bit-identical to bag20 (max difference exactly 0.0).

Also replaced a vacuous assertion found while debugging this: the scripts
asserted that the written frame's CLV equalled the frame it had just been copied
from, which is trivially true and tested nothing. It is now a post-write check
that re-reads both files from disk and compares.

Numerically none of this matters at scoring precision -- 4.4e-16 on a value of
1.9 cannot move an RMSE at four decimals. It matters for reproducibility claims:
without it, "identical to bag20" was not true of the files on disk.

## Sweep 5 (7 Oct) -- maximise

Seed policy: variants are screened at 5 seeds (42-46) and anything promising is
re-measured at 20 (42-61). Seed sd is 0.00020 against a 0.0015 bar, so screening
cannot hide a passing effect, and it cuts compute about fourfold.

**Correction (9 Oct).** This section originally claimed "every reported keep
decision is at 20 seeds". That was true of the classifier tracks and false of the
regression tracks: `W.SEEDS = list(CONFIRM)` could not override a frozen
`seeds=SEEDS` default argument, so every `reg_rmses` call ran at 5 seeds. See
"A seed bug found in sweep 8" below for the fix and the full blast radius. No
kept decision changes; the rows below marked 20-seed on the regression side are
5-seed measurements.

Incumbent: v3 = hurdle regressions (bag20) + partial prior matching at alpha 0.5,
score 0.28929, F1 0.5022 / 0.5173, rmse 0.5969 / 0.5996 and 0.7350 / 0.7451.

### R1 bundle test -- dropped, and the additions are partly redundant
5-seed screen, score gain against the 5-seed hurdle:

| variant | rmse fuel | rmse nonfuel | score gain |
| --- | --- | --- | --- |
| hurdle (sweep2 b1) | 0.5975 / 0.5994 | 0.7352 / 0.7450 | - |
| + f3f4 features | 0.5957 / 0.6005 | 0.7336 / 0.7448 | +0.00047 |
| + catboost magnitude | 0.5960 / 0.5989 | 0.7339 / 0.7444 | +0.00075 |
| + lgbm/cat average | 0.5965 / 0.5989 | 0.7344 / 0.7446 | +0.00052 |
| bundle f3f4 + lgbm/cat avg | 0.5951 / 0.6003 | 0.7330 / 0.7445 | +0.00081 |
| **bundle f3f4 + catboost mag** | 0.5949 / 0.6003 | 0.7325 / 0.7443 | **+0.00096** |

**The bundle is sub-additive.** f3f4 alone is +0.00047 and catboost magnitude
alone +0.00075, summing to +0.00122, but together they give +0.00096 -- 79% of
the sum. Marginals confirm it from both sides: f3f4 adds only +0.00021 once
catboost is in (45% of its standalone effect) and catboost adds +0.00049 once
f3f4 is in (65%). The two changes are mining overlapping signal, so bundling
sub-bar regression tweaks does not accumulate the way adding independent
improvements would.

Note also that the CatBoost magnitude alone beats the LightGBM/CatBoost average
in both the single and the bundled form (+0.00075 against +0.00052; +0.00096
against +0.00081), which reproduces the B3 finding: CatBoost's magnitude model is
genuinely better on positive rows rather than merely decorrelated, so averaging
dilutes it.

Confirmation of the better bundle against v3 -- labelled 20-seed at the time,
actually **5-seed** (see the sweep 8 seed-bug section); v3's own numbers in
this table are genuine 20-seed, so this row compares a 5-seed challenger
against a 20-seed incumbent:

| | fold 1 score | fold 2 score | mean | gain |
| --- | --- | --- | --- | --- |
| v3 regressions | 0.288673 | 0.289905 | 0.28929 | - |
| bundle f3f4 + catboost mag | 0.290410 | 0.289893 | 0.29015 | +0.00086 |

**Dropped**: +0.00086 is well under the bar, and fold 2 is down by 0.000012 --
essentially flat, but it fails the both-folds condition outright. The seed
mislabelling does not rescue it: the both-folds failure is intrinsic, and a
20-seed re-measurement would have to move the mean by +0.00064 and flip fold 2
to pass. Fold 1 alone
gains +0.00174, so the entire mean gain comes from one fold, which is exactly
the asymmetry the both-folds rule exists to catch.

Running total on the regression side across all sweeps: the hurdle (+0.00212)
captured the structural win, and everything since -- catboost magnitude
(+0.00075), f3f4 features (+0.00047), their bundle (+0.00086) -- has been
scraping the same residual.

### R2 monthly decomposition -- dropped decisively, and the reason is structural

Construction: for horizon h in {0,1,2} the target is the raw total in the single
calendar month [c+h, c+h+1). The horizon is a FEATURE, because at predict time
all three months must be forecast from history available at the cutoff -- we
cannot use history up to 1 Jan to predict January. A 1-month outcome window also
needs only one month of future data, so cutoffs extend to 2025-11-01 instead of
the quarterly 2025-09-01; feature-only snapshots were built at 2025-10-01 and
2025-11-01 for this. Fold 2's training set grows from 53,025 rows to 174,341.

Correctness check first: three consecutive monthly totals reproduce the cached
quarterly targets to 9e-13 (float accumulation only), so the decomposition is
measuring the intended quantity.

| variant | rmse fuel | rmse nonfuel | score | delta |
| --- | --- | --- | --- | --- |
| quarterly hurdle (base) | 0.5975 / 0.5994 | 0.7352 / 0.7450 | 0.28919 | - |
| monthly, no shrink | 0.7244 / 0.6965 | 1.0071 / 0.9811 | 0.15039 | -0.13879 |
| monthly + cross-fold shrink | 0.6715 / 0.6582 | 0.8323 / 0.8206 | 0.23052 | -0.05867 |
| average(quarterly, monthly) | 0.6346 / 0.6240 | 0.8161 / 0.8111 | 0.24968 | -0.03951 |
| average(quarterly, monthly+shrink) | 0.6185 / 0.6152 | 0.7634 / 0.7643 | 0.27298 | -0.01621 |

**The Jensen bias is enormous and I initially left it uncorrected.** The hurdle
predicts P(y>0)*E[y|y>0], a conditional MEAN in raw space. The metric is RMSE on
ln(1+Y)/s, and ln(1+mean) > E[ln(1+actual)] for skewed Y, so transforming a mean
prediction overshoots. Correcting it with a cross-fold shrink factor recovers
+0.080 of score -- more than every improvement found in this project combined --
which shows the first number was mostly measuring my own bias, not the method.

Shrink factors, fitted on one fold and applied to the other:

| target | fold | applied k | own-fold optimum | true optimum (wide grid) |
| --- | --- | --- | --- | --- |
| CLV_fuel | 2025-06 | 0.55 | 0.50 | 0.52 |
| CLV_fuel | 2025-09 | 0.50 | 0.55 | 0.56 |
| CLV_nonfuel | 2025-06 | 0.40 | 0.40 | **0.18** |
| CLV_nonfuel | 2025-09 | 0.40 | 0.40 | **0.19** |

My shrink grid started at 0.40, which was **binding for non-fuel**: the real
optimum is near 0.18, so that arm was under-corrected. Re-checked on the cached
raw predictions with a 0.05-1.50 grid. k ~ 0.18 means the raw-mean prediction
overshoots by about 5.5x in the relevant sense, which fits non-fuel being the
more zero-inflated target (65% zeros quarterly, more monthly).

**The verdict survives the grid error.** With an ORACLE per-fold shrink -- each
fold's own optimum, an upper bound no honest procedure can reach -- the score is
0.24758, still -0.04161 behind the quarterly hurdle. So 3x the training rows and
two extra months of usable cutoffs do not come close to compensating.

Why the structure loses: the quarterly hurdle fits z = ln(1+Y)/s directly and is
therefore unbiased for the metric. Any route that predicts raw totals and
transforms afterwards inherits a bias that a single scalar cannot remove,
because the right shrink depends on each customer's dispersion, not just the
population's. Monthly targets are also far more zero-inflated than quarterly
ones, pushing the per-month gates toward 0.5 and making the product a noisier
point estimate. More data on the wrong loss loses to less data on the right one.

### R3 loss and space -- dropped; the stage-2 loss is not a lever
| variant | rmse fuel | rmse nonfuel | score | delta |
| --- | --- | --- | --- | --- |
| current stage 2 (recomputed) | 0.5975 / 0.5994 | 0.7352 / 0.7450 | 0.28921 | +0.00002 |
| huber | 0.5972 / 0.5994 | 0.7351 / 0.7449 | 0.28928 | +0.00009 |
| quantile median | 0.5985 / 0.5992 | 0.7351 / 0.7447 | 0.28910 | -0.00009 |
| tweedie on raw totals | 0.6732 / 0.6641 | 0.8489 / 0.8433 | 0.22177 | -0.06742 |
| avg(current, huber) | 0.5973 / 0.5994 | 0.7351 / 0.7450 | 0.28926 | +0.00007 |
| avg(current, quantile) | 0.5975 / 0.5989 | 0.7350 / 0.7448 | 0.28936 | +0.00017 |
| avg(current, tweedie) | 0.6194 / 0.6172 | 0.7688 / 0.7710 | 0.27019 | -0.01900 |

**Squared loss, Huber and quantile-median all land within 0.0002 score of each
other.** That is a null with a mechanism: the metric IS RMSE on z = ln(1+y)/s, so
squared loss on z is already exactly matched to it. Huber and quantile buy
robustness to heavy tails, and the log transform has already removed them. Do
not revisit the stage-2 loss.

Tweedie reproduces R2's failure for R2's reason -- it is another raw-space route,
so it inherits the Jensen bias. Its shrink factors were perfectly stable
cross-fold (applied = own-fold optimum on all four cells: fuel 0.45/0.45,
non-fuel 0.40/0.40), so the shrink is well estimated and still cannot repair a
bias that depends on each customer's dispersion. Non-fuel's 0.40 is again at the
grid's lower bound, immaterial at a -0.067 gap.

Consistency check: recomputing the incumbent stage 2 from scratch gave 0.28921
against the 0.28919 carried over from sweep 2, a 0.00002 difference explained by
4-decimal rounding in the stored baseline.

### R4 model diversity -- dropped; two independent GBDTs agree
| variant | rmse fuel | rmse nonfuel | score | delta |
| --- | --- | --- | --- | --- |
| lightgbm (base) | 0.5975 / 0.5994 | 0.7352 / 0.7450 | 0.28921 | +0.00002 |
| xgboost alone | 0.5969 / 0.5995 | 0.7353 / 0.7463 | 0.28902 | -0.00017 |
| extratrees alone | 0.6119 / 0.6063 | 0.7457 / 0.7499 | 0.28203 | -0.00716 |
| ridge on signed-log features | 0.6691 / 0.6681 | 0.7501 / 0.7539 | 0.25637 | -0.03282 |
| lgbm + xgboost blend | 0.5973 / 0.5991 | 0.7351 / 0.7454 | 0.28923 | +0.00004 |
| lgbm + extratrees blend | 0.5981 / 0.5992 | 0.7353 / 0.7447 | 0.28916 | -0.00003 |
| lgbm + ridge blend | 0.5986 / 0.5991 | 0.7348 / 0.7446 | 0.28917 | -0.00002 |

**XGBoost comes within 0.00017 of LightGBM.** Two independent implementations with
different split finding and regularisation reaching the same number is much
stronger evidence than one library's tuning ceiling: the gradient-boosted fit on
these features is saturated. The weaker families rule out the alternative
explanation that GBDT is overfitting -- ExtraTrees is -0.0072 and ridge -0.0328,
so the capacity is being used, not wasted.

Blends give nothing (all within 0.00004 of base) and the cross-fold weights
disagree for xgboost (0.5 against 0.1) and extratrees (0.1 against 0.2). Only
ridge's weights agreed, at 0.1/0.1, for -0.00002.

### R5 monthly lag series for the regressors -- dropped at +0.00061
| variant | rmse fuel | rmse nonfuel | score | delta |
| --- | --- | --- | --- | --- |
| hurdle (base) | 0.5975 / 0.5994 | 0.7352 / 0.7450 | 0.28919 | - |
| + lag series (62 columns) | 0.5967 / 0.5972 | 0.7344 / 0.7458 | 0.28980 | +0.00061 |

Fuel improves on both folds (-0.0008, -0.0022); non-fuel improves fold 1 and
gives back 0.0008 on fold 2, inside the component slack. It satisfies conditions
(2) and (3) and fails only on magnitude -- the cleanest sub-bar regression result
in the project. Lags (+0.00061) beat the f3/f4 block (+0.00047), so the
regressions do retain a little feature headroom, just not 0.0015 from any one
block, and R1 showed these blocks are sub-additive when combined.

### Track R complete -- nothing passes
| track | best arm | delta score |
| --- | --- | --- |
| R1 bundle | f3f4 + catboost magnitude | **+0.00086** |
| R5 lag series | 62 lag columns | +0.00061 |
| R3 loss and space | avg(current, quantile median) | +0.00017 |
| R4 model diversity | lgbm + xgboost blend | +0.00004 |
| R2 monthly decomposition | average with shrink | -0.01621 |

Taken together the regression side is closed, and for three separate reasons
rather than one: the loss is already matched to the metric (R3), the model fit is
saturated across independent implementations (R4), and the feature headroom that
remains is both small and sub-additive (R1, R5). Only a different target
construction could change that, and R2 shows the obvious one loses badly to the
loss mismatch it introduces.

### L1 lag series for the classifier -- +0.00139 alone, the first feature block
### ever to improve F1
| variant | F1 fold 1 | F1 fold 2 | mean | score | delta |
| --- | --- | --- | --- | --- | --- |
| bag20 [20s, incumbent] | 0.5022 | 0.5173 | 0.5098 | 0.28929 | - |
| base features [5s] | 0.5031 | 0.5134 | 0.5082 | 0.28869 | -0.00060 |
| + lag series [5s] | 0.5074 | 0.5160 | 0.5117 | 0.29008 | +0.00079 |
| + lag series [20s] | 0.5095 | 0.5169 | 0.5132 | 0.29068 | +0.00139 |

As a classifier-only change this fails by 0.00011 on the gain and also on the
both-folds condition (fold 2 dips 0.0004). Two things make it more than a near
miss. The harness validates -- bag20's own probabilities through the recomputed
rule reproduce v3 exactly. And the effect is stable across seed counts: measured
like for like at 5 seeds the lag gain is 0.29008 - 0.28869 = +0.00139, identical
to the 20-against-20 comparison.

**Every one of block F's five feature groups hurt F1 (-0.0012 to -0.0046); lags
help.** The difference looks structural: lags give monthly resolution on the
target quantities themselves (fuel litres, non-fuel rands, basket counts), where
the base features carry only quarterly aggregates. Block F added
side-information -- sites, timing, vouchers, customer id. Temporal detail on the
same series is a different kind of addition.

## v4_lags -- KEPT. The lag series as ONE change across all three models

| | F1 f1 | F1 f2 | rmse_f f1 | rmse_f f2 | rmse_n f1 | rmse_n f2 |
| --- | --- | --- | --- | --- | --- | --- |
| v3 (incumbent) | 0.5022 | 0.5173 | 0.5969 | 0.5996 | 0.7350 | 0.7451 |
| v4 lags | 0.5095 | 0.5169 | 0.5967 | 0.5972 | 0.7344 | 0.7458 |
| delta | +0.0073 | -0.0004 | -0.0002 | -0.0024 | -0.0006 | +0.0007 |

Per-fold combined score 0.28867 / 0.28991 -> **0.29191 / 0.29048**, mean
**0.29119** against 0.28929.

| condition | value | pass |
| --- | --- | --- |
| mean gain >= 0.0015 | **+0.00190** | yes |
| combined score up on both folds | +0.00324, +0.00057 | yes |
| worst touched component <= 0.001 | rmse_nonfuel -0.0007 | yes |

**Why one change passes where two halves failed.** As a classifier-only change
the lag series was +0.00139; as a regression-only change (R5) it was +0.00061.
Neither clears the bar. Applied to all three models the gains land on DIFFERENT
components of the score and therefore add, giving +0.00190. Contrast R1, where
two changes both aimed at the regressions came out sub-additive at 79% of their
parts because they were mining the same residual. Measuring a single change
against its own component baseline understates it whenever the change touches
more than one component.

Caveat worth carrying into the final pick: fold 1 contributes +0.00324 and fold
2 only +0.00057, and fold 2's F1 actually dips 0.0004. It satisfies the rule, but
it passes on fold 1's strength with fold 2 merely not objecting.

### Track D on the lag classifier's probabilities -- alpha stays 0.5
| alpha | F1 fold 1 | F1 fold 2 | mean |
| --- | --- | --- | --- |
| 0.25 | 0.5024 | 0.5116 | 0.5070 |
| **0.5** | 0.5095 | **0.5169** | 0.5132 |
| 0.75 | **0.5141** | 0.5137 | 0.5139 |
| 1.0 | 0.5063 | 0.5088 | 0.5075 |

Cross-fold the folds DISAGREE -- fold 1's own search picks 0.75 and fold 2's
picks 0.5 -- so alpha stays 0.5 as agreed. alpha 0.75's marginally better mean
(0.5139) is fold 1's +0.0046 cancelling fold 2's -0.0032, the kind of average
that does not survive out of sample.

Useful by-product: **the decision rule is now independently re-validated on a
different probability model.** alpha=0.5 was selected cross-fold on bag20's
probabilities in block H and survives re-fitting on the lag classifier's, an
interior optimum in the same place both times.

`submissions/submission_v4_lags.csv`: 5,488 rows, IDs match, labels in config,
CLV >= 0 (min 0.0264), IPF mix error 0.000. Predicted mix Stable 0.327,
Inactivity 0.297, Fuel growth 0.261, growth:Other 0.064. Opportunity agrees with
v3 on 93.06% of customers.

### Candidate summary
| File | Score | Mean F1 | Note |
| --- | --- | --- | --- |
| `submission_baseline_v1.csv` | 0.28164 | 0.4998 | single seed |
| `submission_v2.csv` | 0.28553 | 0.5006 | 5-seed + hurdle; public 0.2991 |
| `submission_v2_bag20.csv` | 0.28502 | 0.4991 | 20-seed |
| `submission_v2_prior.csv` | 0.28668 | 0.5032 | + Fuel x1.25 |
| `submission_v3.csv` | 0.28929 | 0.5098 | + partial prior matching |
| **`submission_v4_lags.csv`** | **0.29119** | **0.5132** | + lag series on all three models |

Four kept changes in the project: seed bagging (+0.00178), hurdle regressions
(+0.00212), partial prior matching (+0.00262 over Fuel x1.25, itself +0.00115
over bag20), and the lag series (+0.00190). Total 0.28164 -> 0.29119, +0.00955.

### submission_v4_hybrid.csv -- lag labels with v3's regressions
Opportunity from `submission_v4_lags.csv`, CLV columns from `submission_v3.csv`
(bag20 hurdle without lag features). Pure composition of two existing files, so
no refit was needed, and its configuration is exactly what the L1 20-seed row
already measured: the lag classifier under the alpha=0.5 rule with v3's
regressions.

| stack | score | mean F1 | vs v4_lags |
| --- | --- | --- | --- |
| v3 | 0.28929 | 0.5098 | -0.00190 |
| **v4_hybrid** | **0.29068** | 0.5132 | -0.00051 |
| v4_lags | 0.29119 | 0.5132 | - |

Provenance verified after writing: Opportunity bit-identical to v4_lags, CLV
columns bit-identical to v3, Opportunity differing from v3 on 6.94% of customers.

The hybrid is dominated on validation -- it gives up the +0.00051 the lag
features contribute to the regressions while keeping their +0.00139 on the
label. It is the conservative option: the lag regressors are a change to a
component that has already been through four sweeps of tuning, whereas the lag
classifier is the first thing in the project to move F1 at all. Keeping v3's
regressions means the two candidates differ in exactly one component, which also
makes them a cleaner pair for the two final picks than two variants differing in
several places.

### Track L complete -- nothing passes on top of v4_lags (0.29119)

| variant | F1 fold 1 | F1 fold 2 | mean F1 | delta score |
| --- | --- | --- | --- | --- |
| v4_lags (incumbent) | 0.5095 | 0.5169 | 0.5132 | - |
| L2 lgbm + mlp blend | 0.5133 | 0.5158 | 0.5146 | +0.00055 (weights disagree) |
| L2 lgbm + xgboost blend | 0.5105 | 0.5176 | 0.5141 | +0.00035 (weights disagree) |
| L2 lgbm + extratrees blend | 0.5105 | 0.5161 | 0.5133 | +0.00003 (weights agree) |
| L2 lgbm + catboost blend | 0.5111 | 0.5151 | 0.5131 | -0.00003 |
| L2 lgbm + logistic blend | 0.5132 | 0.5124 | 0.5128 | -0.00018 |
| L5 regression-informed features | 0.5082 | 0.5151 | 0.5117 | -0.00061 |
| L2 catboost alone | 0.5058 | 0.5119 | 0.5089 | -0.00173 |
| L4 one-vs-rest, group marginals | 0.5059 | 0.5060 | 0.5060 | -0.00289 |
| L4 one-vs-rest, 5 groups | 0.5048 | 0.5034 | 0.5041 | -0.00363 |
| L2 mlp alone | 0.5003 | 0.5022 | 0.5013 | -0.00478 |
| L2 extratrees alone | 0.4989 | 0.5000 | 0.4995 | -0.00549 |
| L2 logistic alone | 0.4806 | 0.4962 | 0.4884 | -0.00991 |
| L3 fine target, 48 classes | 0.4423 | 0.4220 | 0.4321 | -0.03243 |

#### L2: you can have a different model or a good model, not both
Pairwise probability correlation with the lag-feature LightGBM, against each
family's standalone score:

| family | corr, all cells | corr, 4 large classes | alone |
| --- | --- | --- | --- |
| logistic | 0.9272 | 0.8649 | -0.00991 |
| mlp | 0.9386 | 0.8908 | -0.00478 |
| extratrees | 0.9755 | 0.9382 | -0.00549 |
| catboost | 0.9858 | 0.9656 | -0.00173 |
| xgboost | 0.9974 | 0.9939 | +0.00001 |

The relationship is monotone and unhelpful: the only family that matches
LightGBM's accuracy (XGBoost, F1 0.5132 against 0.5132) is at correlation 0.9939
and so has nothing to contribute, while every decorrelated family is materially
worse. A useful blend needs a partner that is both decorrelated and comparably
accurate, and on this problem no family is. The best blend is lgbm+mlp at
+0.00055, using the second-most-different family, but mlp's standalone deficit
forces a small weight and the cross-fold weights disagree (0.1 against 0.3).
Only extratrees' weights agreed, for +0.00003.

#### L3 and L4 bracket the label structure, and 17 classes is the optimum
| change to the label structure | delta score |
| --- | --- |
| more resolution (48 fine classes) | -0.03243 |
| same resolution, independent binaries | -0.00289 |
| less resolution (5 groups) | -0.00363 |
| single 17-class softmax | best |

L3's collapse (-0.081 mean F1) is sample starvation: 48 classes from the same
~50k rows, when the adoption sub-classes were already down to 10-60 examples at
17. So the 17-class target is not hiding useful distinctions -- it is already at
or past the resolution the data supports, consistent with block C finding the
rare classes unmonetisable even with an oracle.

L4 answers its own question cleanly: five binary models estimate the group
marginals WORSE than one softmax. Preserving the within-group resolution helps
(+0.0019 over the plain version) but both lose to the joint model, because a
softmax shares statistical strength across classes and enforces sum-to-one by
construction, where independent binaries estimate in isolation and then have to
recover coherence by normalising.

#### L5 with good coverage, so a genuine null
Coverage was healthy -- 76% and 83% of training rows, 100% of validation rows --
unlike block E, whose analogous features were missing from all of fold 1's
training snapshots. So -0.00061 on both folds is a real null, and the reason is
that these are not new information: **the hurdle is built from the same base
features the classifier already has**, so its gate and magnitude outputs are
deterministic functions of columns already in the matrix. Feeding them back adds
a noisy one-seed summary of what is already present.

### What now explains every classifier result in the project
| change | new information? | outcome |
| --- | --- | --- |
| L1 lag series | yes, monthly resolution absent from quarterly aggregates | +0.00139 |
| L5 regression-informed | no, derived from existing features | -0.00061 |
| block F, five groups | side-information, redundant with tenure or high-cardinality noise | -0.0012 to -0.0046 |
| L2 family swaps and blends | no | -0.0099 to +0.00055 |
| L3, L4 restructuring | no | -0.0324 to -0.0029 |
| sweep 1 features a-c, e | no | -0.0037 to -0.115 |

**The classifier improves only when given information it genuinely lacks.**
Restructuring the label space, swapping model families, and re-deriving features
it can already compute have now failed 14 times between them. That is the case
for sweep 6 testing new information and nothing else.

## Sweep 6 (7-8 Oct) -- new information only. Nothing passes; no v5.

Six blocks screened as classifier features on top of the lag features, at 5
seeds, under the fixed v3 decision rule. Incumbent v4_lags = 0.29119 (20 seeds).
A 5-seed lags-only base is included so "did this block improve?" is judged like
for like: seeds alone are worth 0.00060 (0.29059 -> 0.29119), enough to mark a
helpful block as a failure if compared against the 20-seed incumbent.

| block | cols | F1 fold 1 | F1 fold 2 | mean F1 | score | vs 5-seed base |
| --- | --- | --- | --- | --- | --- | --- |
| **lags only [5s base]** | 0 | 0.5074 | 0.5160 | 0.5117 | **0.29059** | - |
| N1 own label history | 30 | 0.5101 | 0.5155 | 0.5128 | 0.29103 | +0.00044 |
| N3 per-category recency/age | 51 | 0.5112 | 0.5139 | 0.5125 | 0.29092 | +0.00033 |
| N5 weekly non-fuel and baskets | 26 | 0.5093 | 0.5147 | 0.5120 | 0.29070 | +0.00011 |
| N4 basket structure | 16 | 0.5075 | 0.5151 | 0.5113 | 0.29042 | -0.00017 |
| N2 per-category monthly lags | 132 | 0.5062 | 0.5158 | 0.5110 | 0.29031 | -0.00028 |
| N6 monthly lags 13-18 | 24 | 0.5074 | 0.5136 | 0.5105 | 0.29010 | -0.00049 |

All six land within 0.0009 of each other and of the base. **Not a dilution
pattern**: block width does not order the results (sizes 30, 51, 26, 16, 132, 24
against that ranking, with 24 columns last and 51 second), so this is six blocks
none of which carries meaningful signal, rather than signal being swamped by
width.

### N7: the three improving blocks combined -- sub-additive at 73%
| | score | vs 5-seed base |
| --- | --- | --- |
| + n1 + n3 + n5 | 0.29124 | +0.00065 |
| sum of the three individually | - | +0.00089 |

+0.00006 against the 20-seed incumbent. Same sub-additivity as R1's regression
bundle (79%): overlapping information, so stacking marginal blocks does not
accumulate.

### The all-three-models check, which settles it
The lag series passed sweep 5 because it improved the classifier AND the
regressions, and those land on different score components and therefore add.
Testing the same framing here:

| | rmse fuel | rmse nonfuel |
| --- | --- | --- |
| v4_lags | 0.5967 / 0.5972 | 0.7344 / 0.7458 |
| + n1 + n3 + n5 | 0.5955 / 0.5996 | 0.7363 / 0.7475 |

The blocks **hurt** the regressions: non-fuel worse on both folds, fuel worse on
fold 2, for -0.00090 on the regression side alone. The comparison is like for
like -- R5's 5-seed and v4_lags' 20-seed regression numbers agree to four
decimals, so seed count is not confounding it.

So the combined change is +0.00065 - 0.00090 = about -0.00025, worse than
v4_lags. No configuration of sweep 6 passes and the 20-seed confirmation was not
run, because the arithmetic cannot reach 0.0015 from here.

**This corrects an over-general reading of v4_lags.** Treating something as "one
change across all three models" is not a free uplift. It helps only when the
change improves every component it touches. Lags did (+0.00139 and +0.00061);
these blocks improve the classifier marginally and damage the regressions, so the
same framing converts a marginal gain into a net loss.

### What sweep 6 tells us about the information ceiling
Sweep 6 was built on the one finding that had worked: the classifier improves
only when given information it genuinely lacks. These six blocks were chosen to
be exactly that -- the customer's own past labels, per-category monthly history,
per-category recency, basket structure, weekly non-fuel series, and a longer
lookback. All six are new information in the literal sense, none of it derivable
from the existing columns, and none of it helps.

So the lag series was not an instance of a general rule that new information
helps. It was specific: monthly resolution on the four series that ARE the
prediction targets (fuel litres, fuel rands, non-fuel rands, baskets). Widening
to other quantities, finer categories, or longer history adds nothing. The
information ceiling is not about having more columns; it is that next-quarter
behaviour at this horizon is close to unpredictable beyond recent volume and
recency.

No diagnostic was written: the best non-passing block (N1, +0.00044) is also
negative against the 20-seed incumbent, so it has no configuration worth probing.

## Final-pick hardening (8 Oct)

### make_submission.py recipes
One entry point, five recipes, all from the raw CSVs with no cached artifacts.
Dependencies are pandas, numpy, scikit-learn and lightgbm only -- no CatBoost,
XGBoost or pyarrow on the submission path.

| recipe | seeds | classifier lags | regression lags | decision rule |
| --- | --- | --- | --- | --- |
| v2 | 42-46 | no | no | argmax |
| v3 | 42-61 | no | no | prior matching, alpha 0.5 |
| **v4_hybrid** (primary) | 42-61 | **yes** | no | prior matching, alpha 0.5 |
| v4_lags | 42-61 | yes | yes | prior matching, alpha 0.5 |
| v4_simple | 42-61 | yes | no | Fuel x1.25 |

`--recipe v4_hybrid` reproduces the validation numbers exactly: F1 0.5095 /
0.5169, rmse_fuel 0.5969 / 0.5996, rmse_nonfuel 0.7350 / 0.7451, score
**0.29068**. Runtime about **53 minutes** on 12 logical cores.

### Two measurement problems found and fixed
**The script's own timer was wrong.** It reported 38,162s because the system
clock jumped about 35,005s mid-run and durations were computed with
`time.time()`. Now measured with `time.monotonic()`, which is immune to clock
adjustments. The true runtime (about 53 min) is corroborated by a direct
sampling of the process -- 118 CPU-seconds over 20 seconds of wall time, so 5.9
cores busy.

**Byte-exact reproduction is impossible across both lineages, and this is not
nondeterminism.** The recipe reproduced v4_hybrid's labels on 100.0000% of rows
but its CLV columns differed by exactly 1 ULP (4.441e-16) on 1,582 and 2,203
rows. The cause is summation order: v4_hybrid's CLV descended from
`stability.py`, which averaged per-seed arrays with `np.mean([...], axis=0)`
(pairwise summation), while `make_submission` accumulates `out += ...` in a loop
(sequential). The arithmetic is identical; the rounding is not. No single
implementation can be byte-exact for both lineages, which is why the v2 recipe
*is* byte-exact -- v2 came from the loop form.

Resolved by promoting the recipe's output to `submissions/submission_v4_hybrid.csv`,
so the artifact we might submit is the one the documented recipe produces.
Consequence to record: its CLV columns are now within 1 ULP of
`submission_v3.csv` rather than bit-identical to them. That does not affect the
inferred public score, since RMSE at four decimals cannot see 4e-16, and
`submission_v3.csv` itself is untouched (it is already submitted and selected).

### submission_v4_simple.csv -- the second-pick hedge
The Fuel-growth weight was re-validated on the lag classifier's probabilities
rather than carried over from bag20's:

| weight | fold 2025-06 | fold 2025-09 | mean |
| --- | --- | --- | --- |
| 1.000 | 0.4966 | 0.5051 | 0.5009 |
| 1.125 | 0.5004 | 0.5068 | 0.5036 |
| **1.250** | **0.5036** | 0.5093 | **0.5064** |
| 1.375 | 0.5023 | **0.5095** | 0.5059 |
| 1.500 | 0.5019 | 0.5047 | 0.5033 |

Own-fold optima are 1.25 (fold 1) and 1.375 (fold 2), so the folds **disagree**
and the weight stays at the 1.25 default. Held out, 1.375-from-fold-2 gives
0.5023 on fold 1 and 1.25-from-fold-1 gives 0.5093 on fold 2. Note the weight is
less stable here than on bag20's probabilities, where both folds independently
chose 1.25 -- the disagreement is one grid step and worth 0.0005 of mean F1, so
the default is safe, but it is no longer a clean two-fold replication.

| candidate | validation F1 | validation score |
| --- | --- | --- |
| v4_hybrid (prior matching) | 0.5095 / 0.5169 | **0.29068** |
| v4_simple (Fuel x1.25) | 0.5036 / 0.5093 | 0.28796 |

v4_simple is -0.00271 on validation, which is the point. The transfer table
quantifies the hedge: prior matching cost -0.0004 on public where validation
promised +0.0026, a -0.0030 gap, while Fuel x1.25 gained +0.0048 against a
promised +0.0017. If that pattern repeats on the lag probabilities, v4_simple's
0.0068 validation F1 deficit could largely close on the private split.

Unlike v4_hybrid, v4_simple's public score cannot be inferred -- its labels are
new, so no submitted file shares them. It is a genuine hedge, not an arbitrage.

### submission_diag_alpha075.csv -- diagnostic, not for selection
v4_hybrid with alpha 0.75 instead of 0.5, regressions identical, so only the
decision rule differs. The folds disagreed on alpha (fold 1 preferred 0.75, fold
2 preferred 0.5), which is why 0.5 was kept, and the public board would be a
third reading.

Predicted class mixes against the probe-derived public shares (0.276 Stable,
0.297 Inactivity, 0.252 Fuel growth):

| file | Stable | Inactivity | Fuel growth | sum abs err |
| --- | --- | --- | --- | --- |
| v4_hybrid, alpha 0.5 | 0.327 | 0.297 | 0.261 | 0.060 |
| diag, alpha 0.75 | 0.300 | 0.286 | 0.270 | 0.053 |
| v4_simple, Fuel x1.25 | 0.336 | 0.299 | 0.299 | 0.109 |

alpha 0.75 sits marginally closer to the public mix. Read that weakly: block I
established that matching the mix is **not** what makes prior matching work --
the stale target won there despite being the least accurate of five -- so mix
proximity is not evidence of a better decision rule.

All three new files have CLV columns bit-identical to v4_hybrid, so the set
differs only in the decision rule. Re-deriving alpha 0.5 from the cached test
probabilities reproduces v4_hybrid's labels on 100.0000% of rows, confirming the
composition path and the recipe agree.

### Clean-clone verification (8 Oct)
Fresh `git clone` from GitHub at commit 3e5588d, new venv from
`requirements-lock.txt`, the three CSVs copied in, then
`python src/make_submission.py --recipe v4_hybrid`.

- output **byte-identical** to the committed `submission_v4_hybrid.csv`
- validation reproduced exactly: F1 0.5095 / 0.5169, rmse 0.5969 / 0.5996 and
  0.7350 / 0.7451, score 0.29068
- measured runtime **4785 s (80 min)** with `time.monotonic()`

That also corrects a number I had inferred rather than measured. The working-tree
run reported 38,162 s because the system clock was adjusted mid-run; I estimated
the true time at 53 min by subtracting the apparent jump, and the measured value
is 80 min. The subtraction was the wrong instrument -- a monotonic measurement in
a clean environment is the right one.

## Correction to the record (8 Oct): v4_hybrid was submitted and is selected

Earlier entries described `submission_v4_hybrid.csv` as never submitted with an
inferred public score, and the final picks as v4_lags and v3. Both were wrong.

- `submission_v4_hybrid.csv` **was** submitted on 8 Oct: public **0.305152978**,
  components F1 0.528896416 / RMSE fuel 0.593163991 / RMSE non-fuel 0.723339664,
  rank 49, Zindi ID Bb8Js9A2.
- The selected picks are **v4_hybrid and v3**, not v4_lags and v3.
- The submitted version was the pre-promotion file (CLV copied bit-for-bit from
  v3). The committed version is the `--recipe v4_hybrid` output, whose CLV
  differs by 1 ULP because the two code paths sum per-seed predictions in
  different orders. A resubmission of the recipe output is in flight so the
  selected file matches what the documented recipe produces; the score should be
  identical to every reported digit, since 4.4e-16 on individual predictions
  moves an RMSE by about 1e-16.

### The scoring constants are exact, not approximate
This is the first public row published with full-precision components, and our
formula reproduces it with residual **0.000000000**; solving for a common scale
factor on the normalisers gives k = 1.000000000. So 0.74 and 0.816 are exact.
The +/-0.00006 residuals recorded earlier against six public rows were an
artefact of those rows being published to four decimals, not evidence that the
constants were off. The README's "inferred constants, confirmed mechanism"
framing can now be strengthened: the mechanism was confirmed by Zindi staff and
the constants are confirmed by exact arithmetic.

I also briefly mis-stated this in working: a hand calculation suggested a
systematic 7e-5 residual, which came from my own division error, not from the
formula. The computed check is the authority.

## The alpha decision, with all three readings (8 Oct)

The damping exponent on the prior-matching weights is the single parameter the
two validation folds never agreed on. The public board was taken as a
pre-registered third reading, and it favoured 0.75.

| reading | n | alpha 0.5 | alpha 0.75 | delta F1 |
| --- | --- | --- | --- | --- |
| validation fold 2025-06 | 4,961 | 0.5095 | 0.5141 | **+0.0046** |
| validation fold 2025-09 | 5,238 | 0.5169 | 0.5137 | **-0.0032** |
| public board | ~1,650 | 0.528896 | 0.535301 | **+0.0064** |

- unweighted mean of the three: **+0.0026**
- sample-size weighted pooled: **+0.0014**
- validation-only mean: +0.0007

Public score 0.305152978 (alpha 0.5, rank 49) against **0.307714693** (alpha
0.75, rank 37), a delta of **+0.00256** with the regressions byte-identical, so
the whole difference is the decision rule.

### Why this counts as evidence and where it stops
It counts because the threshold was **pre-registered**: alpha 0.75 was submitted
as a diagnostic with F1 > 0.533 named in advance as the bar, and it cleared it at
0.5353. That is a genuine out-of-sample test rather than picking the best of two
after seeing both.

It stops short of settling the matter for three reasons.
- The public split is roughly 1,650 customers. A class share near 0.28 has a
  standard error of 0.011 there, so a +0.0064 weighted-F1 difference is of the
  same order as sampling noise. One pre-registered reading is still one reading.
- The folds genuinely disagree, and not by a little: fold 1 says +0.0046 and
  fold 2 says -0.0032. Pooling to +0.0014 averages a real conflict rather than
  resolving it.
- **We are now choosing a fitted parameter using public feedback**, which is
  precisely the mechanism behind public-to-private collapse. The transfer table
  already showed the most heavily fitted change we made -- partial prior
  matching -- losing its entire validation edge on public. Tuning alpha on
  public invites the same failure one level down, against a private split of
  about 3,838 customers that nothing has been fitted to.

### Why carrying both is the right response
The selection is **v4_alpha075 and v4_hybrid (alpha 0.5)**. That hedges the
disagreement directly: if alpha 0.75's public edge is real it is the better pick,
and if it is noise then alpha 0.5 -- the value fold 2 preferred and the default
the pre-registration protected -- is still selected. Two picks spanning the one
parameter the evidence is split on is a better use of them than two picks
differing in something the evidence agrees about.

Note also what block I established about this parameter: alpha is not a
staleness correction. Full matching (alpha = 1.0) gives back the entire gain even
though it hits the target mix most precisely, and alpha 1.0 scores 0.5063/0.5088
on validation, below both 0.5 and 0.75. So the useful range is interior and
fairly flat between 0.5 and 0.75, which is consistent with the three readings
landing within 0.006 F1 of each other in both directions.

### Scoring constants reconfirmed
alpha 0.75 is the second public row published with full-precision components, and
the formula reproduces it with residual 1e-9 (rounding of the last digit):
`0.4*0.535300706 + 0.3*(1 - 0.593163991/0.74) + 0.3*(1 - 0.723339664/0.816)
= 0.307714694` against a reported 0.307714693. Two exact rows now, both clean.

### Corrected figure and the resubmission result
`v4_alpha075`'s validation score is **0.29095** (folds 0.29343 / 0.28847), not
0.29139 as first written -- recomputed from its fold F1s (0.5141 / 0.5137) with
v3's regressions. Against v4_hybrid's 0.29067 that is +0.00028 on validation,
beside +0.00256 on public. The validation and public readings agree in sign and
differ by an order of magnitude in size, which is what a genuinely marginal
parameter looks like.

The v4_hybrid resubmission closed the 1-ULP question. `Bb8Js9A2` (CLV copied
bit-for-bit from v3) and `rjQUYHF9` (CLV from the recipe's own hurdle fit) differ
by 4.4e-16 on some rows and scored **identically to all nine reported digits**,
every component included. The two per-seed summation lineages are therefore
indistinguishable to the metric, and the resubmission bought byte-provenance
rather than score -- which is what was predicted before it was sent.

## v4_simple submitted: three decision rules, one probability model (8 Oct)

`submission_v4_simple.csv` (Zindi ID **8ii1ZeJp**) scored public **0.304851105**
(F1 0.528141734, regressions byte-identical to the other two candidates). The
pre-registered bands were: above 0.5353 F1 would replace the alpha 0.5 pick,
0.515 to 0.5353 keeps the current pair. It landed mid-band at 0.5281, so the
selection is unchanged -- A37bufRz and rjQUYHF9.

| rule | validation F1 | public F1 | public score |
| --- | --- | --- | --- |
| simple, Fuel x1.25 | 0.5064 | 0.5281 | 0.304851105 |
| prior matching, alpha 0.5 | 0.5132 | 0.5289 | 0.305152978 |
| **prior matching, alpha 0.75** | **0.5139** | **0.5353** | **0.307714693** |

**The rank order transferred exactly. The step sizes swapped.**

| step | validation | public |
| --- | --- | --- |
| simple -> alpha 0.5 | **+0.0068** | +0.0008 |
| alpha 0.5 -> alpha 0.75 | +0.0007 | **+0.0064** |

Validation attributed almost the whole gain to adopting prior matching and
treated the damping value as a detail; public reverses that attribution almost
exactly. Validation chose the right family of rule and the wrong lever within it.

The first step is additionally inside noise and flips sign with the probability
model: Fuel x1.25 beat alpha 0.5 by 0.0008 on bag20 probabilities (0.5247 against
0.5239) and lost to it by 0.0008 on the lag probabilities (0.5281 against
0.5289). With SE about 0.011 for a 0.28 share on ~1,650 customers, that is a coin
flip.

### Correcting the transfer rule
The rule recorded earlier -- simple constants and structural changes transfer,
fitted procedures and sub-0.001 refinements do not -- rested mainly on one
comparison (v2_prior -> v3, +0.00262 validation against -0.0004 public). Three
rules on one probability model show that was too strong. The accurate statement:
the machinery of prior matching is worth about nothing over a single constant
(+/-0.0008, sign varying), while one scalar inside it is worth +0.0064; and
validation mis-ranked those two by an order of magnitude each way. Validation was
trustworthy for picking the family and untrustworthy for tuning inside it, which
is precisely why alpha needed a pre-registered third reading.

This also means the hedge did its job without being used. v4_simple was carried
in case fitted rules failed to transfer; it showed instead that the fitted
machinery is neutral rather than harmful, and that the parameter inside it is
where the value sits. Keeping both alpha values as the picks remains the right
call on that evidence.

## Fine alpha grid on the cached lag probabilities (8 Oct, no refits)

Prior matching with the stale target, swept from 0.40 to 1.00 in steps of 0.05,
on the cached 20-seed lag-classifier fold probabilities. Nothing was selected on
this; it was run to see the shape.

### Four large classes matched
| alpha | fold 2025-06 | fold 2025-09 | mean |
| --- | --- | --- | --- |
| 0.40 | 0.5087 | 0.5145 | 0.5116 |
| 0.45 | 0.5086 | 0.5160 | 0.5123 |
| 0.50 | 0.5095 | **0.5169** | 0.5132 |
| 0.55 | 0.5115 | 0.5167 | 0.5141 |
| **0.60** | 0.5122 | 0.5163 | **0.5143** |
| 0.65 | 0.5115 | **0.5169** | 0.5142 |
| 0.70 | 0.5126 | 0.5154 | 0.5140 |
| 0.75 | **0.5141** | 0.5137 | 0.5139 |
| 0.80 | 0.5124 | 0.5133 | 0.5128 |
| 0.85 | 0.5102 | 0.5115 | 0.5109 |
| 0.90 | 0.5089 | 0.5110 | 0.5100 |
| 0.95 | 0.5070 | 0.5109 | 0.5089 |
| 1.00 | 0.5063 | 0.5088 | 0.5075 |

**The mean is a broad plateau.** Within 0.001 of the best mean: **alpha
0.55-0.75**, five of thirteen grid points, peaking at **0.60**. Outside that it
falls away steadily, losing 0.0068 by alpha 1.00.

**The folds disagree about where the optimum sits, not about the shape.** Fold
2025-06 peaks at 0.75 with a one-point plateau; fold 2025-09 peaks at 0.65 with a
plateau spanning 0.45-0.65. The two curves are roughly parallel -- fold 1 runs
0.0058 below fold 2 at alpha 0.40 and 0.0026 below at 1.00 -- and the difference
changes sign between 0.70 and 0.80, which is exactly where the two selected picks
straddle.

**Where our two picks sit.** alpha 0.75 is at the **upper edge** of the mean
plateau and alpha 0.50 is just **below its lower edge** (0.5132 against the best
0.5143, 0.0011 short). Neither is the validation-mean optimum; **alpha 0.60 is**,
and it has never been submitted. Reporting that, not acting on it -- the public
board was used once as a pre-registered reading and spending another submission
to chase a 0.0004 validation-mean difference inside a plateau would be the kind
of tuning the pre-registration existed to prevent.

### Six classes matched (adding adoption:Other and growth:Beverages)
This was specified in sweep 5 track D and never run until now.

| alpha | fold 2025-06 | fold 2025-09 | mean |
| --- | --- | --- | --- |
| 0.40 | 0.5070 | 0.5132 | 0.5101 |
| 0.45 | 0.5078 | 0.5143 | 0.5110 |
| 0.50 | 0.5083 | 0.5142 | 0.5113 |
| **0.55** | 0.5100 | 0.5154 | **0.5127** |
| 0.60 | 0.5094 | **0.5159** | 0.5126 |
| 0.65 | 0.5090 | 0.5151 | 0.5121 |
| 0.70 | 0.5106 | 0.5138 | 0.5122 |
| 0.75 | 0.5118 | 0.5102 | 0.5110 |
| 0.80 | **0.5129** | 0.5091 | 0.5110 |
| 0.85 | 0.5120 | 0.5085 | 0.5102 |
| 0.90 | 0.5109 | 0.5083 | 0.5096 |
| 0.95 | 0.5079 | 0.5079 | 0.5079 |
| 1.00 | 0.5069 | 0.5073 | 0.5071 |

Plateau within 0.001 of its best: alpha 0.55-0.70, peaking at 0.55.

**Matching six classes is worse than four at every single alpha.** Best mean
0.5127 against 0.5143, and the gap holds across the whole grid. Adding
`New category adoption: Other` and `Existing-category growth: Beverages` as
matched classes pins two targets the model cannot hit -- adoption classes were
shown unmonetisable even with an oracle in block C -- and the constraint costs
accuracy on the four classes that matter. Track D's open question is answered:
no, do not match more classes.

### No diagnostic file written
The pre-registered condition was to write a submission only if the plateau
clearly extends beyond 0.75 on **both** folds. It does not:

- fold 2025-06's own plateau is 0.75-0.75 and it drops 0.0017 by alpha 0.80;
- fold 2025-09 is already 0.0032 past its own best at alpha 0.75 and keeps
  falling.

Both folds decline beyond 0.75, so nothing was written.

## Sweep 7, block E1: expected-F1 decoding (8 Oct)

A parameter-free decision rule. Instead of weighting classes, choose the
assignment that maximises the model's OWN expected weighted F1,

    sum_c (n_c/N) * 2*TP_c / (n_c + m_c)

with m_c the rows assigned to c and TP_c the sum of P[i,c] over them. True labels
never enter the rule, so there is nothing to fit. Run on the cached 20-seed lag
classifier probabilities for both folds; no refits.

### Results
| rule | fold 2025-06 | fold 2025-09 | mean |
| --- | --- | --- | --- |
| argmax | 0.4966 | 0.5051 | 0.5009 |
| prior matching, alpha 0.5 | 0.5095 | **0.5169** | 0.5132 |
| prior matching, alpha 0.75 | **0.5141** | 0.5137 | **0.5139** |
| expected-F1, n_c model-implied | 0.5090 | 0.5145 | 0.5118 |
| expected-F1, n_c = stale mix | 0.5113 | 0.5126 | 0.5119 |
| expected-F1, n_c averaged | 0.5106 | 0.5133 | 0.5119 |

**Dropped on the pre-stated keep rule.** The bar was mean F1 at least alpha
0.75's 0.5139 and neither fold more than 0.001 below the better of alpha 0.5 /
0.75 on that fold (0.5141 / 0.5169). All three variants miss the mean by
0.0020-0.0021 and both fold conditions.

### It recovers 85% of the gain with no parameter
That is the result worth keeping. The three choices of n_c are
indistinguishable from each other (0.5118-0.5119, a 0.0001 spread) and all sit
about four fifths of the way from argmax to the best fitted rule:

| variant | fold 2025-06 | fold 2025-09 | mean |
| --- | --- | --- | --- |
| model-implied | 71% | 109% | 84% |
| stale mix | 84% | 86% | 85% |
| averaged | 80% | 95% | 85% |

The denominator is the mean argmax-to-alpha-0.75 gap, 0.50085 to 0.51392, which
is +0.01307.

So the bulk of what prior matching buys is not information about the target mix
at all -- it is the generic fact that **weighted F1 rewards predicting rare
classes more often than argmax does**, and a rule derived from the metric alone
finds most of it. The remaining 0.002 is the part that needs the stale label
mix and a damping exponent, and it is exactly the part the two folds disagree
about.

### The optimiser is not the limitation
The fixed point was checked against an exact one-row-at-a-time coordinate ascent
on the same objective:

| variant | fold | sweeps | objective | rows moved by exact ascent | after |
| --- | --- | --- | --- | --- | --- |
| model-implied | 2025-06 | 29 | 0.536186 | 7 | 0.536191 |
| model-implied | 2025-09 | 200 (cap) | 0.547413 | 45 | 0.547499 |
| stale mix | 2025-06 | 24 | 0.527872 | 41 | 0.527951 |
| stale mix | 2025-09 | 200 (cap) | 0.541281 | 63 | 0.541462 |
| averaged | 2025-06 | 29 | 0.532323 | 20 | 0.532363 |
| averaged | 2025-09 | 200 (cap) | 0.544398 | 69 | 0.544561 |

The ascent never lowered the objective, and it raised it by at most 0.00018 by
moving at most 69 of 5,238 rows. Fold 2025-09 did not reach an exact fixed point
in 200 sweeps in any variant -- it cycles, the damping engages and the best-seen
assignment is returned -- but since exact ascent from that point finds under
0.0002, the cap is not why the rule loses. **The objective is being maximised
properly; it is simply not the right objective.** Expected F1 is computed from
the model's own probabilities, so maximising it inherits their miscalibration.
Note the objective values differ by 0.008 across the three n_c choices while
their real F1 differs by 0.0001: the objective's level carries no information
about realised F1.

### The two rules move different degrees of freedom
Effective per-class multipliers at convergence (fold 2025-09), against the IPF
weights. Both columns are normalised by their geometric mean -- `ipf()` does this
internally because argmax is scale-invariant -- so they are directly comparable.

| class | a_c | IPF^0.5 | IPF^0.75 |
| --- | --- | --- | --- |
| Stable * | 0.693 | 0.564 | 0.423 |
| Inactivity * | 0.742 | 0.490 | 0.343 |
| Existing-category growth: Fuel * | 0.729 | 0.592 | 0.455 |
| Existing-category growth: Other * | 0.732 | 0.816 | 0.737 |
| the 13 unmatched classes | 0.815 to 1.381 | 1.168 | 1.168 |

(* the four matched classes.) Relative spread across those four: **a_c 0.069
against IPF 0.529 at alpha 0.5 and 0.804 at alpha 0.75.** The b_c offsets are at
most 0.00007, so the decoding rule, though formally affine, is effectively
multiplicative.

The structure is almost complementary. Expected-F1 decoding applies a **nearly
flat** discount to the four large classes and spends all its variation on the 13
rare ones (0.82 to 1.38), where IPF holds every unmatched class at one value by
construction. IPF does the reverse. That is why they land close in score by
different routes, and it suggests the two are not substitutes -- but combining
them means re-introducing alpha, so it is not a parameter-free option.

### A closer class mix again fails to mean a better F1
Predicted shares of the four large classes on fold 2025-09, with the L1 distance
to the true mix:

| rule | Stable | Inactivity | Fuel | Other | L1 to truth | fold-2 F1 |
| --- | --- | --- | --- | --- | --- | --- |
| TRUE | 0.268 | 0.274 | 0.274 | 0.053 | - | - |
| argmax | 0.394 | 0.309 | 0.235 | 0.047 | 0.206 | 0.5051 |
| alpha 0.5 | 0.354 | 0.283 | 0.254 | 0.057 | 0.119 | **0.5169** |
| alpha 0.75 | 0.330 | 0.270 | 0.262 | 0.057 | 0.082 | 0.5137 |
| expected-F1, model-implied | 0.328 | 0.270 | 0.280 | 0.056 | **0.073** | 0.5145 |
| expected-F1, stale mix | 0.336 | 0.266 | 0.279 | 0.058 | 0.086 | 0.5126 |
| expected-F1, averaged | 0.333 | 0.269 | 0.279 | 0.057 | 0.079 | 0.5133 |

The rule with the **closest** mix to truth (model-implied, 0.073) is beaten on
that fold by the rule with the **second-worst** mix (alpha 0.5, 0.119). Across
the five non-argmax rules the rank correlation between mix distance and fold-2
F1 is **0.10** -- nil, and the sign is the wrong way. Mix distance separates
argmax from everything else and then stops being informative.

This is the third independent confirmation that alpha is not a staleness
correction: block I found full matching (alpha 1.0) gives the entire gain back
despite hitting the target mix most precisely; the fine grid found the optimum
interior and flat; and now a rule that hits the Fuel share almost exactly
(0.280 against a true 0.274, where alpha 0.75 sits at 0.262) still scores lower.
Whatever the damped weights are doing, it is not getting the marginal
distribution right.

### Verdict
Dropped. Nothing written, no recipe added. The transferable finding is that
**85% of the decision-rule gain is available with no fitted parameter at all**,
which bounds how much of our 0.0131 argmax-to-alpha improvement could plausibly
be fragile: at most the last 0.0020.

## Sweep 7, block E2: two regression ideas, both dropped at the screen (8 Oct)

Both run at 5 seeds on base features (no lags), which is the regression recipe
the two selected picks actually ship. Scores below hold F1 at the lag
classifier's fold values (0.5095 / 0.5169) so every delta is the regression
change alone. The same-seed reference is the pure hurdle at **0.29059**.

### E2a: convex blend of the hurdle and a direct regressor
`pred = (1-lambda)*hurdle + lambda*direct`, both on the transformed target.

**CLV_fuel**
| lambda | fold 2025-06 | fold 2025-09 | mean |
| --- | --- | --- | --- |
| 0.00 (hurdle) | 0.5975 | **0.5994** | 0.5984 |
| 0.25 | 0.5967 | 0.5997 | **0.5982** |
| 0.50 | **0.5965** | 0.6005 | 0.5985 |
| 0.75 | 0.5968 | 0.6017 | 0.5993 |
| 1.00 (direct) | 0.5977 | 0.6033 | 0.6005 |

**CLV_nonfuel**
| lambda | fold 2025-06 | fold 2025-09 | mean |
| --- | --- | --- | --- |
| 0.00 (hurdle) | **0.7352** | **0.7450** | **0.7401** |
| 0.25 | 0.7357 | 0.7451 | 0.7404 |
| 0.50 | 0.7366 | 0.7456 | 0.7411 |
| 0.75 | 0.7380 | 0.7463 | 0.7422 |
| 1.00 (direct) | 0.7397 | 0.7474 | 0.7436 |

Non-fuel is unambiguous: RMSE rises monotonically in lambda on **both** folds, so
the hurdle is the best point of the whole family and the direct regressor costs
+0.0045 / +0.0024. Nothing to blend.

Fuel is the interesting case, and it is a clean illustration of why lambda had to
be chosen cross-fold. Fold 2025-06 has a shallow interior minimum at lambda
0.25-0.50 worth -0.0010; fold 2025-09 is monotone increasing, +0.0039 by lambda
1.00. **The two folds disagree about the direction, not just the size.** Chosen
honestly:

| chosen on | lambda | held-out fold | RMSE | against hurdle |
| --- | --- | --- | --- | --- |
| fold 2025-06 | 0.50 | 2025-09 | 0.6005 | **+0.0011** |
| fold 2025-09 | 0.00 | 2025-06 | 0.5975 | +0.0000 |

So one direction costs 0.0011 and the other selects the hurdle itself. Combined
score **0.29036, -0.00023** against the same-seed hurdle: dropped, and it also
fails the component slack (rmse_fuel worse by 0.0011) and the both-folds
condition. Note the mean-optimal lambda for fuel (0.25, mean 0.5982 against
0.5984) is a 0.0002 difference -- inside noise -- and the cross-fold protocol
correctly refuses to bank it.

### E2b: forward-in-time isotonic calibration of the hurdle gate
The gate P(y>0) was recalibrated with an isotonic fit trained on the last
snapshot whose 3-month outcome window closes before the fold cutoff
(2025-03-01 for fold 2025-06, 2025-06-01 for fold 2025-09), then multiplied by
the unchanged magnitude stage.

**It made reliability worse on all four target-fold combinations.**

| target / fold | mean abs decile gap, raw | calibrated | RMSE change |
| --- | --- | --- | --- |
| CLV_fuel / 2025-06 | 0.0099 | 0.0378 | +0.0048 |
| CLV_fuel / 2025-09 | 0.0091 | 0.0144 | +0.0011 |
| CLV_nonfuel / 2025-06 | 0.0212 | 0.0482 | +0.0080 |
| CLV_nonfuel / 2025-09 | 0.0146 | 0.0209 | +0.0021 |

Combined score **0.28754, -0.00305**: dropped, the largest regression loss of
the sweep.

#### Why, measured rather than assumed
The gate's bias is **not stationary across one quarter**, and it is already small
at the folds:

| target / fold | gate bias at the source | gate bias at the fold | after calibration |
| --- | --- | --- | --- |
| CLV_fuel / 2025-06 | +0.035 | +0.003 | **-0.037** |
| CLV_fuel / 2025-09 | +0.003 | -0.001 | -0.006 |
| CLV_nonfuel / 2025-06 | +0.048 | +0.008 | **-0.040** |
| CLV_nonfuel / 2025-09 | +0.008 | +0.005 | -0.002 |

(bias = mean predicted P(y>0) minus the realised positive rate.)

The 2025-03-01 source over-predicts by +0.035 and +0.048; the same construction
at 2025-06-01 over-predicts by +0.003 and +0.008. The calibrator therefore
learns a correction an order of magnitude too large and applying it **flips** a
+0.003 bias to -0.037. Where the two cutoffs happen to agree (the 2025-09 fold,
source bias +0.003 against fold bias -0.001) the damage is correspondingly small
but still negative.

The isotonic fit is not at fault -- it is accurate where it is fitted. In-sample
reliability at its own source is 0.0017-0.0042, better than the raw gate manages
at the fold. The entire loss is transfer:

| target / fold | isotonic in-sample at source | raw gate at fold | transplanted |
| --- | --- | --- | --- |
| CLV_fuel / 2025-06 | 0.0042 | 0.0099 | 0.0378 |
| CLV_fuel / 2025-09 | 0.0017 | 0.0091 | 0.0144 |
| CLV_nonfuel / 2025-06 | 0.0040 | 0.0212 | 0.0482 |
| CLV_nonfuel / 2025-09 | 0.0035 | 0.0146 | 0.0209 |

**The general condition this fails:** a forward-in-time calibrator helps only if
the miscalibration is stable over the gap it is carried across. Here the
residual bias (0.003 to 0.008 at the folds) is *smaller than its own
quarter-to-quarter variation* (0.035 to 0.048 one quarter earlier), so any such
calibrator necessarily imports more error than it removes. The likely reason the
source is so much worse is simply less training data -- 24,850 rows at
2025-03-01 against 38,409 at 2025-06-01 -- and the bias shrinks as history
accumulates, which is precisely the wrong direction for carrying a correction
forward.

A useful by-product: **LightGBM's gate is already well calibrated out of
sample.** Raw decile gaps of 0.009-0.021 with mixed signs, and a mean bias of
+0.003 to +0.008 on 4,961-5,238 held-out customers, leave essentially nothing
for a calibration stage to collect. This closes gate calibration as a lever, not
just this implementation of it.

### No 20-seed confirmation was run
The protocol was a 5-seed screen followed by a 20-seed confirmation *for
anything that passes the screen*. Neither block passes, and neither is close:
E2a is 0.0017 short of the +0.0015 bar and E2b is 0.0045 short. For E2a the
lambda = 0 point **is** the hurdle, so more seeds improve both arms of the
comparison and cannot move the difference by 0.0017 -- seed bagging was worth
+0.00178 in total across all three models when first measured. Spending roughly
20 minutes of fits to reconfirm a negative at higher precision was not
worthwhile, so nothing was written and no diagnostic file was produced.

### Sweep 7 summary
| block | change | mean score | against reference | verdict |
| --- | --- | --- | --- | --- |
| E1 | expected-F1 decoding (3 variants) | F1 0.5118-0.5119 | -0.0020 vs alpha 0.75 | dropped |
| E2a | hurdle/direct blend, cross-fold lambda | 0.29036 | -0.00023 | dropped |
| E2b | forward-in-time isotonic gate calibration | 0.28754 | -0.00305 | dropped |

Three for three negative, and the recipe table is unchanged. Running total:
**four kept changes out of 60 variants across seven sweeps.** The two selected
picks (`v4_alpha075`, `v4_hybrid`) are untouched.

## Sweep 8, renewal (fill-rhythm) features (9 Oct) -- all dropped

The premise: fuel is a refill good, so a customer who fills every 11 days makes
a predictable number of fills next quarter, and one fill more or fewer decides a
label the official rule defines in rands. Three blocks, built in
[`src/features5.py`](../src/features5.py) from rows strictly before each cutoff:
r1 fuel fills, r2 all qualifying baskets plus a no-purchase estimate, r3
calendar phase. Nothing reads the outcome window except its calendar LENGTH,
which is known at prediction time.

### The sanity gate passed
Run before any model, as a stop condition. True Fuel-growth rate by (projected
minus previous) fills, among customers with 3-12 fuel fills last quarter and gap
CV below 0.5 -- pure arithmetic, no model:

| bucket | fold 2025-06 n | rate | lift | fold 2025-09 n | rate | lift |
| --- | --- | --- | --- | --- | --- | --- |
| <= -1 | 19 | 0.053 | 0.22 | 18 | 0.111 | 0.39 |
| 0 | 25 | 0.160 | 0.67 | 20 | 0.150 | 0.53 |
| >= +1 | 60 | 0.333 | 1.39 | 50 | 0.400 | 1.41 |

Monotone on both folds, and it replicates with the last-6-fills gap (+0.184 and
+0.323 spread). The strict filter covers only ~2% of customers, but that is the
filter, not the features: **CV < 0.5 is the binding clause** (5.1-5.6% of
customers; median gap CV is about 1.0, which is what a memoryless process looks
like). The projection itself is defined for **97%**, and the separation survives
on the full population at 0.137 / 0.219 / 0.289 and 0.143 / 0.256 / 0.300, still
monotone, with ~18% of rows in the two informative buckets.

Single-feature AUC for Fuel growth puts the strength in the **ratio** forms, not
the difference: `r1_projratio_med` 0.644 / 0.637 and `r1_spend_ratio_med` 0.641 /
0.639, against `r1_projdiff_med` 0.543 / 0.536 and `r1_gap_med` 0.554 / 0.554.
The arithmetic reconstruction of the official rule is high-recall and
low-precision -- `r1_hits_line_med` fires on 68% of customers with precision 0.31
against a 0.26 base rate -- because projected fills times median rand clears the
1.25x line too easily.

### Classifier results, both alphas, against a lags-only reference
5-seed reference: alpha 0.5 mean 0.5117, alpha 0.75 mean 0.5118.

| block | cols | a0.5 mean | gain | a0.75 mean | gain |
| --- | --- | --- | --- | --- | --- |
| r1 fuel renewal | +34 | 0.5124 | +0.00028 | 0.5109 | -0.00034 |
| r2 all baskets | +23 | 0.5108 | -0.00036 | 0.5118 | -0.00000 |
| **r3 calendar phase** | +9 | 0.5120 | +0.00010 | **0.5149** | **+0.00124** |
| combo r1+r3 | +43 | 0.5112 | -0.00021 | 0.5120 | +0.00010 |

r3 was the only block worth confirming: at alpha 0.75 it beat the reference on
both folds and failed only the magnitude bar, by 0.00026.

**The 20-seed confirmation dissolved it.** Reference alpha 0.5 mean 0.5132,
alpha 0.75 mean 0.5139 (the cached 20-seed lags-only classifier, not refitted):

| | fold 2025-06 | fold 2025-09 | mean | gain |
| --- | --- | --- | --- | --- |
| r3, alpha 0.5 | 0.5105 | 0.5162 | 0.5134 | +0.00006 |
| r3, alpha 0.75 | 0.5147 | 0.5148 | 0.5147 | +0.00033 |

The screen's +0.00124 was mostly **reference noise, not signal**: at 5 seeds the
lags-only reference scored 0.5116 on fold 1 at alpha 0.75, against 0.5141 at 20
seeds. r3's own fold-1 value barely moved (0.5157 to 0.5147); the reference moved
0.0025. This is the clearest instance in the project of a screen gain being an
artefact of the baseline rather than the change, and the two-stage protocol
catching it.

The combination is also **sub-additive to the point of being negative**: r1 and
r3 each nominally positive at their better alpha, together -0.00021 at alpha 0.5.
43 mostly redundant columns dilute rather than accumulate, reproducing sweep 5's
sub-additivity finding on the regression side.

### Why r1 failed: it is a compression of the lag series, not new information
Binary Fuel-growth target, 1 seed, AUC on the held-out fold:

| features | fold 2025-06 | fold 2025-09 | mean |
| --- | --- | --- | --- |
| lag series only (62 cols) | 0.7292 | 0.7264 | 0.7278 |
| **r1 renewal only (34 cols)** | 0.7324 | 0.7247 | **0.7286** |
| lags + r1 (96 cols) | 0.7329 | 0.7268 | 0.7298 |
| r1 adds over lags | +0.0037 | +0.0004 | **+0.0021** |

34 renewal columns reproduce the discrimination of the whole 62-column lag
series, and their union adds only +0.0021 AUC. The renewal construction is a
**more compact encoding of signal the lags already carry**, which is a real
result about representation and a useless one for scoring: +0.0021 binary AUC is
consistent with the +0.00028 weighted F1 r1 actually delivered. This is the same
verdict as sweep 6's six "new information" blocks, now with a direct measurement
of the redundancy rather than an inference from the score.

### r3's gain had a second explanation, pre-registered before it was measured
Of r3's 9 columns, only 2 are per customer. The other 7 are identical for every
customer at a cutoff, take **9 distinct values across the 17 cutoffs**, and pair
the test cutoff uniquely with **2024-12-01** -- the same calendar month:

| | eom_prev | eom_out | eom_diff | days_diff |
| --- | --- | --- | --- | --- |
| 2024-12-01 | 22 | 21 | -1 | -1 |
| **2025-12-01 (test)** | **22** | **21** | **-1** | **-1** |

So a tree can use them to identify the cutoff and specialise on one December
window rather than learn anything about renewal. A decomposition into r3a
(customer shares) and r3b (window counts) was implemented to separate the two,
and then **not run**: the 20-seed gain is +0.00033, so there is nothing left to
decompose. Recorded because the decomposition exists in
[`src/features5.py`](../src/features5.py) and a reviewer will see it unused.

Design wart worth owning: the three month-start (`bom`) columns are constant
across all 17 cutoffs and cannot be split on at all. They were dead weight in
every r3 run.

### The population is drifting away from this block's premise
Eligibility is "3+ baskets ever", so the cohort accumulates customers who
qualify once and go quiet:

| | 2024-06 | 2025-03 | 2025-09 | 2025-12 (test) |
| --- | --- | --- | --- | --- |
| eligible customers | 2,745 | 4,649 | 5,238 | 5,488 |
| zero fuel fills in the prior quarter | 0.5% | 16.5% | 26.1% | **28.2%** |
| median prior-quarter fills | 8 | 5 | 4 | 4 |
| median gap CV | 0.76 | 0.96 | 1.02 | **1.05** |

By the test cutoff **28% of customers have no fuel fill in the prior quarter at
all** and the median customer's gap CV is 1.05. A fill-rhythm feature set is
structurally limited on that population and the trend runs against it.

The labels trend with it: Inactivity 0.072 -> 0.274 over the 16 cutoffs
(+0.176/year), Stable 0.365 -> 0.268 (-0.104/year), Fuel roughly flat
(-0.032/year). A linear read puts Inactivity near 0.318 at the test cutoff, so
the stale target understates it by about 0.044. This changes no decision -- block
I tested five projected targets and dropped all five, recording that the stale
target won despite being the least accurate of the five -- but it identifies the
**cause** of that drift, which the record previously carried as an unexplained
premise.

### The one near-miss: r1 for the regressions
R1 on top of the base-feature hurdle, at a genuine 20 seeds, against the shipped
v3 regressions:

| | fold 2025-06 | fold 2025-09 |
| --- | --- | --- |
| rmse fuel, base | 0.5969 | 0.5996 |
| rmse fuel, + r1 | **0.5957** | **0.5982** |
| rmse non-fuel, base | 0.7350 | 0.7451 |
| rmse non-fuel, + r1 | **0.7328** | **0.7450** |

Score 0.29159, **+0.00093**, better on both folds with no component worse.
**Dropped** on the magnitude bar, missing it by 0.00057.

All four RMSE measurements improve, on both folds, at 20 seeds. That makes it the
strongest sub-bar regression result in the project -- against the kept hurdle
(+0.00212) and seed bagging (+0.00178), and ahead of sweep 5's best non-passer.
Nothing was written: sweep 8's write condition was a classifier change passing,
and none did.

### Sweep 8 summary
| block | change | best result | verdict |
| --- | --- | --- | --- |
| sanity | arithmetic separation check | monotone, both folds | **passed, block continued** |
| r1 | fuel renewal, classifier | +0.00028 (a0.5, 5s) | dropped |
| r2 | all-basket renewal, classifier | -0.00000 (a0.75, 5s) | dropped |
| r3 | calendar phase, classifier | +0.00033 (a0.75, **20s**) | dropped |
| combo | r1+r3, classifier | +0.00010 (a0.75, 5s) | dropped |
| reg | r1 on the hurdle | +0.00093 (**20s**, both folds) | dropped, closest miss |

Four kept changes out of 66 variants across eight sweeps. The recipe table is
unchanged and the selected picks are untouched.

## A seed bug found in sweep 8, and what it does and does not affect

While confirming the regression result I noticed a 20-seed run finishing in the
same 134 s as its 5-seed screen and returning identical RMSEs. The cause:

```python
def hurdle_reg(Xtr, y, Xva, params=None, w=None, seeds=SEEDS, cat=False):
```

A `seeds=SEEDS` default binds the **list object** at definition time, so the
`W.SEEDS = list(CONFIRM)` idiom used across sweep 5 rebinds the module name and
leaves the default untouched. Every such call ran at 5 seeds. Fixed at the root
in [`src/sweep2.py`](../src/sweep2.py): the four bagging helpers now take
`seeds=None` and resolve the module attribute at call time, and `reg_predict` /
`reg_rmses` thread an explicit `seeds` through.

### Not affected
- **`make_submission.py`**, where `seeds` is a required positional in both
  `hurdle` and `bag_reg`. Every submission it produced used its stated seed count.
- **Both selected picks.** `submission_v4_hybrid.csv` and
  `submission_v4_alpha075.csv` are byte-identical to `make_submission.py` output
  (verified, and the recipe output was promoted to the committed artifact), so
  their CLV columns are genuine 20-seed.
- **`V3_RF` / `V3_RN`**, the shipped regression reference. These came from
  `stability.py`, which fits each seed once and composes the bag explicitly;
  `preds/stability.json`'s `bag20` entry gives 0.59693739 / 0.59957643 and
  0.73500345 / 0.74506159, matching the recorded constants exactly.
- Every **classifier** number, in sweep 8 and in `write_v4_lags`, where seeds
  were passed explicitly as a keyword.

### Affected, and corrected below
- sweep 5's R1 "20-seed confirmation" of the regression bundle (+0.00086) was a
  **5-seed** measurement. It stays dropped: it failed the both-folds condition
  outright, with fold 2 down 0.000012.
- sweep 5's "regressors: hurdle + lag features, 20 seeds" was **5-seed**, as were
  the CLV columns of `submission_v4_lags.csv` and
  `submission_diag_r1_bundle.csv`, both of which print and log "20 seeds".
- the claim "Every reported keep decision is at 20 seeds" was false for
  regression tracks.

### The v4_lags comparison was confounded, but only slightly
`submission_v4_lags.csv` changed **two** things against v3 at once -- lag
features in the regressions **and** 5 seeds instead of 20. Measured now at
matched seed counts against the genuine 20-seed base:

| lag-feature regressions | rmse fuel | rmse non-fuel | score | vs base 20s |
| --- | --- | --- | --- | --- |
| base, 20 seeds (shipped v3) | 0.5969 / 0.5996 | 0.7350 / 0.7451 | 0.29067 | - |
| + lag features, 5 seeds | 0.5967 / 0.5972 | 0.7344 / 0.7458 | 0.29118 | +0.00051 |
| + lag features, 20 seeds | 0.5957 / 0.5972 | 0.7347 / 0.7455 | 0.29139 | +0.00072 |

The 5-seed row reproduces the logged v4_lags validation numbers (fold 2: 0.5972
and 0.7458) exactly, confirming which measurement went into the record.

**The seed-count effect alone is +0.00021** -- an order of magnitude smaller
than the confound would need to be to matter. So the mislabelling is a
documentation error, not a substantive one:

- the logged v4_lags validation RMSEs barely move at 20 seeds (fold 2 non-fuel
  0.7458 to 0.7455, fuel unchanged at 0.5972);
- lag features in the regressions are **positive on validation at both seed
  counts** (+0.00051 and +0.00072), so validation never disliked them;
- the public board still disliked them (+0.0012 fuel, +0.0003 non-fuel), and
  +0.00021 of seed effect does not account for that.

**This corrects an overstatement made while the bug was being investigated**, not
the original finding. The record's reading -- that validation liked the lag
regressors and public did not, which is why the selected picks drop them --
survives intact, and the decision to ship `v4_hybrid` over `v4_lags` was never in
question (two real files; v4_hybrid scored 0.3052 against 0.3045).

What the episode does change is confidence in the *procedure* rather than the
conclusion: a "20-seed confirmation" that silently ran at 5 seeds went unnoticed
across three sweeps, and was only caught because a runtime looked implausible.
The fix makes the seed count explicit at every regression call site; the general
lesson is that a mutable module-level default is a silent-failure channel, and
the tell was available all along in the wall-clock time.

## The v5_reg diagnostic, pre-registered (10 Oct)

`--recipe v5_reg` is `v4_alpha075` with one change: the hurdle regressions see
the base features plus the r1 renewal block of
[`src/features5.py`](../src/features5.py), 131 regression features against 97,
at the same 20 seeds. The label side is untouched, so the `Opportunity` column is
identical to `submission_v4_alpha075.csv` on all 5,488 rows -- verified, not
assumed.

| | fold 2025-06 | fold 2025-09 | mean |
| --- | --- | --- | --- |
| v4_alpha075 score | 0.29343 | 0.28847 | 0.29095 |
| **v5_reg score** | **0.29473** | **0.28907** | **0.29188** |
| delta | +0.00130 | +0.00060 | **+0.00093** |
| rmse fuel | 0.5957 (was 0.5969) | 0.5982 (was 0.5996) | |
| rmse non-fuel | 0.7328 (was 0.7350) | 0.7450 (was 0.7451) | |

All four RMSE measurements improve, on both folds, and F1 is unchanged by
construction. The +0.00093 reproduces the sweep 8 measurement exactly, which is
the point of building it through the recipe rather than trusting the sweep
harness.

**Why it is a diagnostic and not a candidate.** +0.00093 is under the 0.0015 keep
bar. The project's transfer record is that sub-0.001 regression refinements
either vanish or reverse on the public board, which is exactly the band this sits
in, so it is carried with a pre-registered adoption rule rather than selected:

> adopt only if public RMSE improves on **both** targets and the public score
> gain is at least +0.0005 over A37bufRz (0.307714693)

Requiring both targets to improve is the part that matters. Fold 2's non-fuel
gain is 0.0001 -- essentially nothing -- so a public reading that improves fuel
while non-fuel drifts would be consistent with noise rather than signal, and the
rule refuses it.

### Verification
- **Determinism**: two runs, the second with `--skip-validation`, are
  byte-identical (sha256 `ac71ce631f10c6bc...`, 399,080 bytes).
- **Label identity**: `Opportunity` matches `submission_v4_alpha075.csv` on
  5,488 of 5,488 rows. Since that artifact predates the recipe refactor, this is
  also an end-to-end check that the refactor preserved the classifier path.
- **CLV**: every row changes, as it must -- mean fuel 1.45712 against 1.45659,
  mean non-fuel 0.72057 against 0.72160, maximum per-row shift 0.60 and 0.33.
  Minimum CLV 0.02672.
- **Dependency envelope unchanged**: `features5` imports only numpy, pandas and
  `features`, so the submission path is still pandas / numpy / scikit-learn /
  lightgbm.
- Runtime 4,680 s with validation, 1,689 s for the final fit alone.

### A stale claim found while regression-testing the refactor
The refactor generalised `make_submission`'s single lags flag into an explicit
extra-feature dict per model. To check it had not disturbed existing recipes I
rebuilt `v2` and compared bytes -- and it differed. The cause was not the
refactor:

- the difference is exactly **4.441e-16**, one ULP, on 1,524 and 2,222 CLV rows
  with `Opportunity` identical on all 5,488 -- the documented pairwise-against-
  sequential summation signature;
- `submission_v2.csv` dates from 2026-10-07 13:09, the same un-promoted batch as
  `v3` and `v2_bag20`, where every promoted artifact is dated 10-08;
- and the refactor was verified equivalent directly: `stack()` and `feats()` are
  the only functions changed, they are pure, and both return bit-identical
  frames on the `extra=None` and `extra=lags` paths.

So the reproduce table's "byte-identical" entry for `v2` was wrong and is now
corrected to "1 ULP, not byte-identical". The artifact was **not** promoted:
rewriting it would change the bytes behind an already recorded public score for a
file that is neither selected nor worth a submission to re-confirm.

Two smaller record fixes made at the same time: `submissions_log.md` still
carried `v4_alpha075`'s superseded validation figure 0.29139 in its status table
where the README and these notes had the corrected **0.29095**; and the README's
pairwise-lineage caveat named only `v3` and `v2_bag20`, now also `v2`.

The transferable lesson is about the check rather than the bug. A verification
step aimed at one thing found a different, older error, which is an argument for
re-running the reproduce table rather than treating it as settled once written.

## v5_reg's public result: the pre-registration refuses it (10 Oct)

`p7G9LVTM` scored **0.309155505** -- CLV Fuel 0.588231347, CLV Non Fuel
0.724859897, Opportunity 0.535300706.

The rule, fixed before submission, was: adopt only if public RMSE improves on
**both** targets and the score gain is at least +0.0005 over `A37bufRz`
(0.307714693).

| component | A37bufRz | p7G9LVTM | delta | test |
| --- | --- | --- | --- | --- |
| CLV Fuel RMSE | 0.593163991 | 0.588231347 | **-0.00493** | pass |
| CLV Non Fuel RMSE | 0.723339664 | 0.724859897 | **+0.00152** | **FAIL** |
| score | 0.307714693 | 0.309155505 | **+0.00144** | pass |

**Not adopted.** The score gain is +0.00144, nearly triple the +0.0005 margin,
and two of three conditions pass -- and the rule still refuses it. That is the
pre-registration working as intended rather than failing: had the margin alone
decided, a non-fuel regression would have been adopted on the strength of a fuel
gain.

### What the three readings say per target
| target | fold 2025-06 | fold 2025-09 | public |
| --- | --- | --- | --- |
| rmse fuel | -0.0012 | -0.0014 | **-0.0049** |
| rmse non-fuel | -0.0022 | -0.0001 | **+0.0015** |

Fuel moves the same direction on all three and about four times harder on public
than on either fold -- the first time in this project that a regression change
has transferred *larger* than validation predicted. Non-fuel was already
effectively zero on fold 2 and reverses on public.

Read structurally, that is what the block should do. r1 is built from fuel fill
rhythm: gap statistics, phase, projected fill counts, projected litres. It has a
mechanism for fuel volume and none for non-fuel basket spend, where it was
presumably picking up a weak correlate that did not survive out of sample. The
sweep 8 verdict -- +0.00093 combined, under the bar -- was an average over one
target the features explain and one they do not.

This also sharpens the project's transfer rule once more. The earlier reading was
that sub-0.001 regression refinements vanish or reverse on public. Fuel here is
-0.0012 and -0.0014 on validation, squarely in that band, and it transferred at
-0.0049. The distinction is not the size of the validation effect but whether
there is a mechanism tying the feature to the target: a 0.001 effect with a
causal story transferred, and a 0.002 effect without one reversed.

## `v5_fuel`: splitting the renewal block by target (10 Oct, post-hoc)

p7G9LVTM separated cleanly by target -- fuel improved on all three readings,
non-fuel reversed on public -- so `--recipe v5_fuel` keeps the renewal block for
the **CLV_fuel hurdle only** and returns CLV_nonfuel to base features. This
needed per-target regression feature selection in `make_submission.py`:
`reg_renew_targets` names the targets that get the block, and `reg_fit` stacks
one matrix per distinct feature set rather than one per target, so recipes where
both targets share features pay nothing extra.

| | fold 2025-06 | fold 2025-09 | mean |
| --- | --- | --- | --- |
| F1 | 0.5141 | 0.5137 | 0.5139 |
| rmse fuel | 0.5957 | 0.5982 | 0.5970 |
| rmse non-fuel | 0.7350 | 0.7451 | 0.7400 |
| score | 0.29390 | 0.28906 | **0.29148** |

+0.00053 over `v4_alpha075`, roughly half of `v5_reg`'s +0.00093, since the
non-fuel gain validation liked is deliberately surrendered.

### Labelled post-hoc, and why that label is the important part
The fuel evidence predates any submission: both folds said -0.0012 and -0.0014.
What public added was the *reason to split by target*, and that is a decision
made after seeing the held-out result. Calling it post-hoc is not a formality --
it is the difference between the v5_reg reading (a pre-registered test that
failed, so nothing is adopted) and this one (a recipe built to fit the public
components we have already seen).

Its public score is therefore **determined rather than predicted**: all three
columns are byte-identical to published files, giving
0.4*0.535300706 + 0.3*(1 - 0.588231347/0.74) + 0.3*(1 - 0.723339664/0.816)
= **0.309714415**, +0.002000 over A37bufRz. That certainty is real and it is
also exactly why it should not be mistaken for evidence: a number computed from
the public split cannot speak to the private one. The fuel half carries two
out-of-sample time folds behind it; the decision to drop non-fuel carries one
public reading of about 1,650 customers, where 0.0015 RMSE is inside noise.

### Verification
399040 bytes, sha256 `e861bd187aa5ebb2...`. All four checks pass: a second run with
`--skip-validation` is byte-identical; `Opportunity` and `CLV_nonfuel` match
`submission_v4_alpha075.csv` on 5,488 of 5,488 rows; `CLV_fuel` matches
`submission_diag_v5_reg.csv` on 5,488 of 5,488 rows. The composition is exact, so
the recipe and the three-way identity are two descriptions of the same file.

Not submitted.

## Open questions
- Does higher or lower win on the leaderboard?
- Does public score track validation?
