# Sasol Customer Value Recruitment Challenge (Zindi)

Solo entry. Per customer, predict for Dec 2025 – Feb 2026: fuel litres
(`CLV_fuel`), non-fuel rands (`CLV_nonfuel`) and an `Opportunity` label
(17 classes). Closes 19 Oct 2026.

## Results

| Stack | Validation | Public | Note |
| --- | --- | --- | --- |
| single-seed baseline (`src/baseline.py`) | 0.28164 | 0.2974 | reference |
| v2 — 5-seed bagging + hurdle regressions | 0.28553 | 0.2991 | |
| v3 — + partial prior matching | 0.28929 | 0.3031 | **selected pick 2** |
| v4_lags — + lag series on all three models | 0.29119 | 0.3045 | submitted, not selected |
| **v4_hybrid — lag classifier, v3 regressions** | 0.29068 | **0.3052** | **selected pick 1**, rank 49 |
| v4_simple — lag classifier, Fuel ×1.25 rule | 0.28796 | not submitted | hedge |

Validation is the mean of two time-based folds; see
[Validation protocol](#validation-protocol). `v4_hybrid` scored **0.305152978**
on the public board (F1 0.528896416, RMSE fuel 0.593163991, RMSE non-fuel
0.723339664) at rank 49, and is one of the two selected picks alongside `v3`.

**What transferred to the public board and what did not.** Six sweeps and about
55 variants produced four kept changes. Comparing each one's validation delta
with its measured public delta gives a sharper rule than "labels matter":

| change | validation | public | public ΔF1 |
| --- | --- | --- | --- |
| bagging + hurdle | +0.00389 | **+0.0017** | +0.0025 |
| Fuel ×1.25 prior | +0.00166 | **+0.0048** | **+0.0120** |
| partial prior matching | +0.00262 | **−0.0004** | −0.0008 |
| lag series, all three models | +0.00190 | **+0.0014** | +0.0050 |
| R1 regression bundle (diagnostic) | +0.00086 | **−0.0009** | 0.0000 |

**Simple constants and structural changes held up; fitted procedures and
sub-0.001 refinements did not.** A single global weight both folds agreed on
(Fuel ×1.25) nearly tripled on public. A structural model change (the hurdle)
transferred. But partial prior matching — an IPF procedure re-estimated per
dataset with its damping chosen cross-fold — went from +0.00262 on validation to
−0.0004 on public, and every regression refinement below 0.001 either vanished or
reversed. That is why the primary pick is `v4_hybrid`: it keeps the lag
classifier's label gain and drops the lag regressors, whose +0.0012/+0.0003 RMSE
damage on public was invisible to validation.

Full evidence for every variant, kept or dropped, is in
[`reports/notes.md`](reports/notes.md); every file and score is in
[`reports/submissions_log.md`](reports/submissions_log.md).

## Reproduce

```bash
python -m venv .venv
source .venv/Scripts/activate        # Windows; use .venv/bin/activate on Linux/macOS
pip install -r requirements-lock.txt
# place train.csv, test.csv and SampleSubmission.csv in data/
python src/make_submission.py --recipe v4_hybrid
```

One entry point, five recipes, each going from `data/train.csv` and
`data/test.csv` to a submission with **no cached artifacts** — all 17 snapshots
and their features are built in memory. Each prints its per-fold and mean
validation score, validates the output before writing, and reports its runtime.

| recipe | seeds | classifier lags | regression lags | decision rule | output |
| --- | --- | --- | --- | --- | --- |
| **v4_hybrid** | 42–61 | yes | no | prior matching, α 0.5 | primary candidate |
| v4_lags | 42–61 | yes | yes | prior matching, α 0.5 | |
| v4_simple | 42–61 | yes | no | Fuel ×1.25 | hedge |
| v3 | 42–61 | no | no | prior matching, α 0.5 | final pick 2 |
| v2 | 42–46 | no | no | argmax | |

The submission path needs only **pandas, numpy, scikit-learn and lightgbm**.
CatBoost, XGBoost and pyarrow appear in `requirements-lock.txt` for the sweep
scripts and are not imported when producing a submission.

### Reproducibility, stated precisely

`--recipe v4_hybrid` is **byte-identical** across runs and reproduces the
committed `submissions/submission_v4_hybrid.csv` exactly (verified by `cmp`).
`--recipe v2` likewise reproduces `submission_v2.csv` byte-for-byte.

One honest caveat. `submission_v3.csv` and `submission_v2_bag20.csv` were
produced by an earlier script that averaged per-seed predictions with
`np.mean([...], axis=0)` (pairwise summation), whereas `make_submission`
accumulates in a loop (sequential). The arithmetic is identical but the last-bit
rounding is not, so **no single implementation can be byte-exact for both
lineages**. Re-deriving v4_hybrid through the recipe changed its CLV columns by
exactly 1 ULP (4.4e-16), which a four-decimal RMSE cannot see; the recipe's
output is what is now committed, so the artifact and its recipe agree.

Note also that CSV floats are written with `float_format='%.17g'` and must be
read with `float_precision='round_trip'` — pandas' default CSV float parser is
not correctly rounded and shifts values by 1 ULP on every read/rewrite cycle.

### Environment, hardware and runtime

| | |
| --- | --- |
| Python | 3.13.13 |
| Packages | pinned in [`requirements-lock.txt`](requirements-lock.txt) |
| OS | Windows 11 (26200) |
| CPU | AMD Zen 3, 12 logical cores |
| `--recipe v4_hybrid` | **4785 s (80 min)** end to end, including both validation folds |
| `--recipe v4_hybrid --skip-validation` | **1642 s (27 min)**, final fit only |
| `--recipe v2` | 646 s (11 min) |

Runtimes are measured with `time.monotonic()` and the v4_hybrid figure comes
from the clean-clone run below. An earlier report of 38,162 s was an artefact of
the system clock being adjusted mid-run while durations were computed from
`time.time()`; the 53 min I first inferred by subtracting the estimated jump was
also wrong, and 4785 s is the measured value.

### Clean-clone test

Verified end to end on 8 Oct 2026: a fresh `git clone` from GitHub, a new venv
installed from `requirements-lock.txt`, the three CSVs copied into `data/`, then
`python src/make_submission.py --recipe v4_hybrid`. The output is **byte-identical**
to the committed `submissions/submission_v4_hybrid.csv` (`cmp`, clean), and the
validation print reproduces exactly: F1 0.5095 / 0.5169, rmse_fuel 0.5969 /
0.5996, rmse_nonfuel 0.7350 / 0.7451, score 0.29068.

## Method

1. **Monthly snapshots** at 16 cutoffs, 2024-06-01 to 2025-09-01. A snapshot
   holds every customer with 3+ distinct baskets strictly before the cutoff,
   with features built **only** from rows strictly before it, and the
   label-rule outcomes for the three months after it.
2. **Features** — the `base` block of [`src/features.py`](src/features.py), 97
   columns:
   recency, tenure, basket-gap mean and standard deviation, per-quarter
   fuel/non-fuel spend and basket counts for q1–q4 plus the last month and all
   time, derived rate ratios (last quarter against the customer's own long-run
   rate), and per-category previous-quarter spend and ever-bought flags.
   plus, for the v4 recipes, the **lag series** of
   [`src/features3.py`](src/features3.py) (62 more): twelve monthly lags of fuel
   litres, fuel rands, non-fuel rands and basket counts, and thirteen weekly fuel
   totals. Lags reaching back before a customer's first transaction are NaN
   rather than 0, so "no history" stays distinct from "history with no spend".
   This is the only feature block in the project that improved the classifier;
   see `reports/notes.md` for the fourteen that did not.
3. **Opportunity** — one 17-class LightGBM, probabilities averaged over the seed
   set, then a **decision rule**. The rule is where most of the F1 came from:
   iterative proportional fitting nudges the predicted class mix toward the last
   fully observed snapshot's true mix over the four large classes, and the
   resulting weights are damped by α = 0.5 before the argmax. α was selected
   cross-fold — each fold's own search chose 0.5 independently — and
   re-validated on the lag classifier's probabilities. Full matching (α = 1)
   gives back the entire gain, so matching the mix is not the objective.
4. **CLV_fuel / CLV_nonfuel** — a hurdle model: `P(y>0)` from a classifier times
   `E[y|y>0]` from a regressor fitted on positive rows only. Each stage is
   seed-averaged and the product is clipped at 0. This is the single largest
   win in the project and the mechanism is confirmed rather than assumed:
   non-fuel is about 65% zeros per quarter against fuel's 29%, and the hurdle
   helps non-fuel more than fuel, ordered by zero-inflation.
5. **Final training** — the submitted models are retrained on **all 16
   snapshots** and predict the test cutoff **2025-12-01**, with features built
   from all of `train.csv`, which ends 2025-11-30. No outcome window is
   available past the cutoff, so the test snapshot carries features only.

### Validation protocol

Two time-based folds, validating on the 2025-06-01 and 2025-09-01 snapshots.
Each fold trains **only** on snapshots whose three-month outcome window ends on
or before its cutoff, so no training outcome can overlap the outcome being
scored:

| Fold | Validate on | Train cutoffs | Snapshots |
| --- | --- | --- | --- |
| 1 | 2025-06-01 | ≤ 2025-03-01 | 10 |
| 2 | 2025-09-01 | ≤ 2025-06-01 | 13 |

Fold 2 is the baseline's original split, so its numbers are directly comparable
with `src/baseline.py`.

### Scoring formula — unpublished constants, verified exactly

```
Score = 0.4*F1 + 0.3*(1 - RMSE_fuel/0.74) + 0.3*(1 - RMSE_nonfuel/0.816)
```

**The mechanism is confirmed.** Zindi staff confirmed that the RMSE components
are normalised, in the competition discussion *"Clarification Request:
Leaderboard Public Score Discrepancy"* (25 Sep 2026), and the Info page now
reads "normalised RMSE … normalised Weighted F1". The weights (0.3 / 0.3 / 0.4)
are official.

**The two constants are unpublished but verified exact.** 0.74 and 0.816 have
never been officially stated. Four things establish them:

- `submission_v4_hybrid.csv` is reported with full-precision components, and the
  formula reproduces its public score with residual **0.000000000**:
  `0.4*0.528896416 + 0.3*(1 - 0.593163991/0.74) + 0.3*(1 - 0.723339664/0.816)
  = 0.305152978`. Solving for a common scale factor on the normalisers gives
  k = 1.000000000;
- the three constant-label probe submissions imply a single shared regression
  term to within 0.00018 of each other, which only happens if both the weights
  and the normalisers are right;
- they reproduce six further public rows to within 0.00006, the residual being
  entirely explained by those rows publishing components to four decimals;
- another participant in the staff thread independently recovered the same two
  values, and under Zindi's multi-metric policy the normalisers are typically
  the starter notebook's scores, consistent with the magnitudes.

So the arithmetic is settled even though the values are not official. The
formula is still used for **local model selection only** and never asserted as
the official metric. Every decision
is additionally reported per component (F1, RMSE fuel, RMSE non-fuel) in
`reports/notes.md`, so no conclusion in this repo depends on the two constants
being exactly right. Blocks H and I are the clearest cases: they hold the
regressions fixed, which makes the combined score an exact affine function of
weighted F1 — measured on block I's 20 variants, `score = 0.4*meanF1 + 0.085393`
to a residual of 1.1e-16 — so the ranking and every verdict there are identical
whether read on raw F1 or on the score, and the two constants cannot influence
them at all.

## Compliance

- **Data**: only the provided `train.csv` / `test.csv`. No external data, no
  scraping, no other competitions' data.
- **Official files unmodified**: [`src/label_rules.py`](src/label_rules.py) and
  [`src/label_config.json`](src/label_config.json) are the official label
  generator and config, byte-identical to the initial commit
  (`git log -- src/label_rules.py src/label_config.json` shows one commit).
- **Open-source packages only**, with versions: pandas 3.0.6, numpy 2.5.3,
  scikit-learn 1.9.1, lightgbm 4.7.0, pyarrow 25.0.1, catboost 1.2.10. Full
  transitive set in `requirements-lock.txt`. Only pandas, numpy, scikit-learn
  and lightgbm are needed by `make_submission.py`; pyarrow is the snapshot cache
  used by the sweep scripts and catboost appears in dropped experiments only.
- **No AutoML and no hyperparameter-tuning libraries.** Every hyperparameter was
  set by hand; the one grid search (18 configs, `reports/notes.md` block A3) was
  written out explicitly and its result was dropped.
- **Seeds, explicitly**: 42–46 for v2 and every reported validation figure;
  42–61 for the 20-seed bag; 42–81 in the seed-stability analysis. Set on every
  LightGBM estimator via `random_state`; no other source of randomness.
- Solo entry, private repo, no code shared.

## Repo layout

**Load-bearing** — needed to produce a submission:

| Path | What |
| --- | --- |
| `src/make_submission.py` | the entry point: raw CSVs → submission, no cache |
| `src/features.py` | feature blocks; `base` (97 columns) is what ships |
| `src/features3.py` | the lag series (62 columns), used by every v4 recipe |
| `src/snapshots.py` | snapshot construction (`load_data`, `build_one`) and the optional parquet cache |
| `src/label_rules.py`, `src/label_config.json` | official label generator and config, unmodified |

**Record** — evidence and reference, not used by the entry point:

| Path | What |
| --- | --- |
| `src/baseline.py` | the original single-seed reference; still pins RMSE fuel 0.6052 / non-fuel 0.7489 / F1 0.5047 |
| `src/validate.py` | fold definitions, per-class F1, confusion matrices, the scoring helper |
| `src/sweep2.py` | sweeps A–G: training recipe, regressions, adoption, cutoff density, rule events, new features, seasonal analogs |
| `src/train_label.py` | sweep 1: label-model variants a–f |
| `src/features2.py` | block F feature groups (customer id, sites, fuel type/price, timing, vouchers) |
| `src/features4.py` | sweep 6 blocks N1–N6 (own label history, per-category history, recency, basket structure, weekly series, longer lookback) |
| `src/stability.py` | seed-stability analysis and the 20-seed bag |
| `src/prior_adjust.py` | 3-parameter class-prior adjustment, cross-fold |
| `src/calibrate.py` | block H: the decision-rule search that produced prior matching |
| `src/project_mix.py` | block I: projected target mixes for prior matching |
| `src/sweep5.py` | sweeps 5–6: regression tracks R1–R5, label tracks L1–L5, blocks N1–N7 |
| `reports/notes.md` | every variant tried, kept or dropped, with the reasoning |
| `reports/submissions_log.md` | every file in `submissions/`, with validation and public scores |
| `docs/` | official label rules and data dictionary |

Nothing in the **Record** group is imported by `make_submission.py`; they are
kept deliberately, because they are the evidence behind roughly fifty dropped
variants across six sweeps, and a reviewer should be able to check the negative
results, not just the final model.

`data/`, `submissions/` and `preds/` are local only and git-ignored. Every file
in `submissions/` is named after the run that produced it and has a row in
`reports/submissions_log.md`.

## Rules to remember

Solo only, no private code sharing. Provided data only. No AutoML. Always set
the seed. 10 submissions a day, 200 total, 2 final picks. Keep this repo private.
