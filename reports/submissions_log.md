# Submissions log

Validation = train on snapshots ending before 1 Sep 2025, score on the 1 Sep 2025 snapshot.

| Date | File | What | Val RMSE fuel | Val RMSE non-fuel | Val F1 | Public | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 2026-10-05 | submission_baseline_v1.csv | LGBM, 16 monthly snapshots, seed 42 | 0.6052 | 0.7489 | 0.5047 | 0.2974 | pipeline test |
| 2026-10-06 | submission_v2.csv | 5-seed bag (42-46), monthly, hurdle regressions, reference classifier | 0.5994 | 0.7450 | 0.5058 | **0.2991** | combined score 0.28553 |

| 2026-10-06 | submission_v3.csv | bag20 regressions + prior matching to the 2025-09-01 mix (4 large classes) at alpha 0.5 | 0.5996 | 0.7451 | 0.5173 | 0.3031 | score 0.28929; best candidate |

| 2026-10-08 | submission_v4_simple.csv | lag classifier + simple Fuel x1.25 rule + v3 regressions | 0.5996 | 0.7451 | 0.5093 | **0.3049** | Zindi ID **8ii1ZeJp**. Public 0.304851105 = F1 0.528141734 / rmse 0.593163991 / 0.723339664. Pre-registered bands: >0.5353 would have replaced the alpha 0.5 pick, 0.515-0.5353 keeps the pair. Landed mid-band at 0.5281, so **selection unchanged** |
| 2026-10-08 | submission_v4_alpha075.csv | v4_hybrid with prior matching at alpha 0.75 (same file as submission_diag_alpha075.csv) | 0.5996 | 0.7451 | 0.5139 | **0.3077** | Zindi ID **A37bufRz**, rank 37. Public 0.307714693 = F1 0.535300706 / rmse 0.593163991 / 0.723339664. Pre-registered as a diagnostic with bar F1>0.533, cleared it, so promoted to **selected**. Folds disagreed on alpha; see notes. `submission_diag_alpha075.csv` and `submission_v4_alpha075.csv` are the SAME file, kept under both names so the pre-registration history stays readable; `--recipe v4_alpha075 --skip-validation` reproduces it byte-identically |
| 2026-10-10 | submission_diag_v5_reg.csv | v4_alpha075 with the r1 renewal block added to the hurdle regressions (131 regression features, 20 seeds). Opportunity column **identical on all 5,488 rows** to submission_v4_alpha075.csv | 0.5982 | 0.7450 | 0.5139 | not submitted | validation **0.29188** (+0.00093 over v4_alpha075); **diagnostic, pre-registered: adopt only if public RMSE improves on BOTH targets and the public score gain is at least +0.0005 over A37bufRz (0.307714693)** |
| 2026-10-07 | submission_v4_lags.csv | lag series on classifier + both hurdle regressors, prior matching alpha 0.5. **Classifier 20 seeds; CLV columns were built at 5 seeds, not 20** -- a frozen default argument swallowed the seed override (found 9 Oct, see notes). Re-measured at matched seeds the difference is +0.00021 in score, so the row stands | 0.5972 | 0.7458 | 0.5169 | **0.3045** | score 0.29119; best candidate at the time, not selected |
| 2026-10-07 | submission_v4_hybrid.csv | hybrid: lag labels + v3 regressions | 0.5996 | 0.7451 | 0.5169 | **0.3052** | submitted twice, identical score 0.305152978 = F1 0.528896416 / rmse 0.593163991 / 0.723339664, rank 49. **Bb8Js9A2** was the pre-promotion composition; **rjQUYHF9** is the `--recipe v4_hybrid` output and is the **selected** one. validation 0.29068 |
| 2026-10-07 | submission_diag_r1_bundle.csv | v3 labels + R1 bundle regressions (f3f4 features, CatBoost magnitude). CLV built at **5 seeds, not 20** -- same frozen-default bug | 0.6003 | 0.7443 | 0.5173 | 0.3022 | **diagnostic, not for selection** (track R best non-passer, +0.00086 at 5 seeds) |

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

## Final picks and the v4_hybrid lineage

**Selected on Zindi: `A37bufRz` (v4_alpha075) and `rjQUYHF9` (v4_hybrid).**
The two picks differ only in the prior-matching damping exponent, which is the
one parameter the evidence is split on.

| file | Zindi ID | public | validation | status |
| --- | --- | --- | --- | --- |
| `submission_v4_alpha075.csv` | **A37bufRz** | **0.307714693** (rank 37) | 0.29095* | **selected** |
| `submission_v4_hybrid.csv` | **rjQUYHF9** | **0.305152978** (rank 49) | 0.29068 | **selected** |
| `submission_v4_hybrid.csv` (pre-promotion) | Bb8Js9A2 | 0.305152978 | 0.29068 | superseded |
| `submission_v4_lags.csv` | - | 0.3045 | 0.29119 | submitted |
| `submission_v3.csv` | - | 0.3031 | 0.28929 | submitted |
| `submission_v4_simple.csv` | 8ii1ZeJp | 0.304851105 | 0.28796 | submitted, mid-band, not selected |
| `submission_diag_v5_reg.csv` | - | not submitted | **0.29188** | pre-registered diagnostic |

\* the folds disagree about alpha 0.75: +0.0046 F1 on fold 1, -0.0032 on fold 2.

**The resubmission confirmed the 1-ULP prediction.** Bb8Js9A2 (CLV copied from
v3) and rjQUYHF9 (CLV from the recipe's own hurdle fit) differ by 4.4e-16 on some
rows and scored **identically to all nine reported digits**, including every
component. So the two summation lineages are indistinguishable to the metric, and
the resubmission bought byte-provenance rather than score.

**Two lineages of the same file.** The version submitted on 8 Oct was built by
composition: the lag classifier's labels with CLV columns copied bit-for-bit from
`submission_v3.csv`. The version now committed is the output of
`--recipe v4_hybrid`, whose CLV comes from its own hurdle fit and differs from the
v3 lineage by exactly 1 ULP (4.4e-16) on some rows, because the two code paths
sum per-seed predictions in different orders.

A 4.4e-16 shift in individual predictions moves an RMSE by about 1e-16 and the
score by less than that, so the resubmission aligns provenance rather than
changing anything measurable. Its public score should match 0.305152978 to every
reported digit.

### Three decision rules on one set of probabilities
All three use the same lag classifier and byte-identical regressions, so every
difference is the decision rule alone.

| rule | validation F1 | public F1 | public score | Zindi ID |
| --- | --- | --- | --- | --- |
| simple, Fuel x1.25 | 0.5064 | 0.5281 | 0.304851105 | 8ii1ZeJp |
| prior matching, alpha 0.5 | 0.5132 | 0.5289 | 0.305152978 | rjQUYHF9 |
| **prior matching, alpha 0.75** | **0.5139** | **0.5353** | **0.307714693** | A37bufRz |

**The rank order transferred exactly; the step sizes swapped.**

| step | validation | public |
| --- | --- | --- |
| simple -> alpha 0.5 | **+0.0068** | +0.0008 |
| alpha 0.5 -> alpha 0.75 | +0.0007 | **+0.0064** |

Validation said the win was adopting prior matching at all, with the damping
value a rounding detail. Public says adopting prior matching is worth almost
nothing and the damping value is the whole effect. The two magnitudes are near
mirror images. So validation got the ordering right and the attribution wrong --
it identified the right family of rules and the wrong lever inside it.

The first step is also inside noise and **changes sign with the probability
model**: on bag20 probabilities Fuel x1.25 beat alpha 0.5 by 0.0008 on public
(0.5247 against 0.5239), and on the lag probabilities alpha 0.5 beats Fuel x1.25
by 0.0008 (0.5289 against 0.5281). A 0.28 class share on ~1,650 public customers
has a standard error of 0.011, so +/-0.0008 is a coin flip either way.

### This corrects the transfer rule stated earlier
The earlier reading was "simple constants and structural changes transferred;
fitted procedures and sub-0.001 refinements did not", resting mainly on
v2_prior -> v3 going from +0.00262 on validation to -0.0004 on public. With three
rules measured on one probability model that reading is too strong. The accurate
version:

- the **machinery** of prior matching buys essentially nothing over a single
  global constant -- +/-0.0008 on public, sign depending on the probability model,
  comfortably inside noise;
- but **one scalar inside that machinery** (alpha) is worth +0.0064 on public,
  the largest single decision-rule effect measured in the project;
- and validation mis-ranked those two by an order of magnitude in both
  directions.

So it is not that fitted procedures fail to transfer. It is that validation was
reliable for choosing *which family* of decision rule to use and unreliable for
tuning *within* it -- which is exactly why the alpha disagreement needed a
pre-registered third reading rather than more validation folds.

### Scoring constants, third confirmation
All three rows carry full-precision components and the formula reproduces each to
about 1e-9: 0.304851105, 0.305152978 and 0.307714694 against reported
0.304851105, 0.305152978 and 0.307714693. 0.74 and 0.816 are settled.

### Seed-count correction (9 Oct)
A frozen `seeds=SEEDS` default argument meant `W.SEEDS = list(CONFIRM)` never
took effect, so every `reg_rmses` call and the CLV columns of
`submission_v4_lags.csv` and `submission_diag_r1_bundle.csv` ran at 5 seeds while
printing and logging 20. Fixed at the root in `src/sweep2.py`.

**No selected pick is affected.** `make_submission.py` takes `seeds` as a
required positional, and both selected files reproduce byte-identically from it,
so their CLV columns are genuine 20-seed. `V3_RF`/`V3_RN` come from
`stability.py`, whose `bag20` entry matches the recorded constants to eight
decimal places. Every classifier number passed seeds explicitly.

Re-measured at matched seed counts, lag-feature regressions score 0.29118 at 5
seeds and 0.29139 at 20 against a 0.29067 base, so the seed effect is **+0.00021**
-- a documentation error rather than a substantive one. No verdict changes.

### Recipe reproduction, verified byte-for-byte
Every candidate is reproducible from the raw CSVs through `make_submission.py`,
with no cached artifacts:

| file | recipe | verification | runtime |
| --- | --- | --- | --- |
| `submission_v4_alpha075.csv` | `--recipe v4_alpha075` | **byte-identical** | 1970 s |
| `submission_v4_hybrid.csv` | `--recipe v4_hybrid` | **byte-identical**, also from a clean GitHub clone | 4785 s full / 1642 s final-fit |
| `submission_v4_simple.csv` | `--recipe v4_simple` | **byte-identical** | 1806 s |
| `submission_v2.csv` | `--recipe v2` | **1 ULP, not byte-identical** -- see below | 328 s |
| `submission_diag_v5_reg.csv` | `--recipe v5_reg` | **byte-identical** across two runs (sha256 `ac71ce631f10c6bc...`) | 4680 s full / 1689 s final-fit |

So both submitted and selected files -- A37bufRz and rjQUYHF9 -- are exactly what
their documented recipes produce. `submission_v4_simple.csv` needed no promotion:
it was built by composition from cached probabilities, and because its CLV came
from the already-promoted v4_hybrid, both lineages are now the loop-summation
form and the recipe matches it exactly.

#### Correction (10 Oct): `submission_v2.csv` is 1 ULP off its recipe
Rebuilding v2 through `make_submission.py` gives a file that differs from the
committed artifact by exactly **4.441e-16** on 1,524 of 5,488 `CLV_fuel` rows and
2,222 `CLV_nonfuel` rows, with the `Opportunity` column identical on all 5,488.
That is the documented `np.mean` pairwise against loop sequential lineage
mismatch: `submission_v2.csv` dates from 2026-10-07 13:09, the same un-promoted
batch as `submission_v3.csv` and `submission_v2_bag20.csv`, and was never
re-derived through the recipe the way `v4_hybrid` was. The earlier
"byte-identical" entry for it was wrong.

**Not promoted.** Rewriting the artifact would change the bytes behind an already
recorded public score (0.2991) for a file that is neither selected nor worth a
submission slot to re-confirm. The promote-then-resubmit route was justified for
`v4_hybrid` because it was a selected pick. The rows that matter for review --
both selected picks -- are unaffected and remain byte-exact.

This surfaced while regression-testing an unrelated refactor, which was itself
verified separately: `stack()` and `feats()` are the only functions that changed,
they are pure, and both produce bit-identical frames on the `extra=None` and
`extra=lags` paths. `v5_reg`'s `Opportunity` column matching the pre-refactor
`v4_alpha075` artifact on all 5,488 rows confirms the classifier path end to end.

### The scoring formula is now confirmed exactly
v4_hybrid is the first public row reported with full-precision components, and it
settles the question:

    0.4*0.528896416 + 0.3*(1 - 0.593163991/0.74) + 0.3*(1 - 0.723339664/0.816)
      = 0.305152978

Residual **0.000000000**. Solving for a common scale factor on both normalisers
gives k = 1.000000000. So **0.74 and 0.816 are exact**, not approximations. The
+/-0.00006 residuals seen on the six earlier rows were entirely an artefact of
those rows' components being published to four decimals.

This also vindicates the inference that identified v4_hybrid as worth submitting:
because its labels were bit-identical to `v4_lags` and its CLV bit-identical to
`v3`, its public score had to be v4_lags' F1 term plus v3's regression terms. The
predicted 0.30515 against a measured 0.305152978 differed only by 4-decimal input
rounding.

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
