# Sasol Customer Value Recruitment Challenge (Zindi)

Solo entry. Per customer, predict for Dec 2025 – Feb 2026: fuel litres
(`CLV_fuel`), non-fuel rands (`CLV_nonfuel`) and an `Opportunity` label
(17 classes). Closes 19 Oct 2026.

## Results

| Stack | Validation score | Public |
| --- | --- | --- |
| single-seed baseline (`src/baseline.py`) | 0.28164 | — |
| **v2 — 5-seed bagging + hurdle regressions** | **0.28553** | **0.2991** |

Validation is the mean of two time-based folds (see
[Validation protocol](#validation-protocol)). 31 variants were tested across four
sweeps and 2 were kept — seed bagging (+0.00178) and the hurdle regressions
(+0.00212), both variance reduction on the continuous targets. The full evidence,
including every dropped variant and why, is in
[`reports/notes.md`](reports/notes.md).

## Reproduce

```bash
python -m venv .venv
source .venv/Scripts/activate        # Windows; use .venv/bin/activate on Linux/macOS
pip install -r requirements-lock.txt
# place train.csv, test.csv and SampleSubmission.csv in data/
python src/make_submission.py
```

This goes from `data/train.csv` and `data/test.csv` to
`submissions/submission_v2.csv` with **no cached artifacts** — it builds all 17
snapshots in memory — prints the per-fold and mean validation score, validates
the output before writing, and reports its runtime. It reproduces the committed
`submissions/submission_v2.csv` **byte-for-byte**.

```bash
# the 20-seed bag, from the same code path
python src/make_submission.py --seeds 42-61 --out submissions/submission_v2_bag20.csv
```

### Environment, hardware and runtime

| | |
| --- | --- |
| Python | 3.13.13 |
| Packages | pinned in [`requirements-lock.txt`](requirements-lock.txt) |
| OS | Windows 11 (26200) |
| CPU | AMD Zen 3, 12 logical cores |
| Runtime | **646 s (10.8 min)** for `make_submission.py` end to end, 5 seeds, including both validation folds and the final fit |

Scales roughly linearly in the seed count: the 20-seed bag is about 4× that.

## Method

1. **Monthly snapshots** at 16 cutoffs, 2024-06-01 to 2025-09-01. A snapshot
   holds every customer with 3+ distinct baskets strictly before the cutoff,
   with features built **only** from rows strictly before it, and the
   label-rule outcomes for the three months after it.
2. **Features** — 97 columns, the `base` block of [`src/features.py`](src/features.py):
   recency, tenure, basket-gap mean and standard deviation, per-quarter
   fuel/non-fuel spend and basket counts for q1–q4 plus the last month and all
   time, derived rate ratios (last quarter against the customer's own long-run
   rate), and per-category previous-quarter spend and ever-bought flags.
3. **Opportunity** — one 17-class LightGBM; probabilities averaged over the seed
   set, then argmax.
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

### Scoring formula — inferred constants, confirmed mechanism

```
Score = 0.4*F1 + 0.3*(1 - RMSE_fuel/0.74) + 0.3*(1 - RMSE_nonfuel/0.816)
```

**The mechanism is confirmed.** Zindi staff confirmed that the RMSE components
are normalised, in the competition discussion *"Clarification Request:
Leaderboard Public Score Discrepancy"* (25 Sep 2026), and the Info page now
reads "normalised RMSE … normalised Weighted F1". The weights (0.3 / 0.3 / 0.4)
are official.

**The two constants are still inferred.** 0.74 and 0.816 have not been
officially published. Three things support them:

- they were recovered here by fitting four public leaderboard rows, which they
  reproduce to nine decimal places;
- another participant in that same thread independently backed out the same two
  values;
- under Zindi's multi-metric policy the normalisers are typically the starter
  notebook's scores, which is consistent with the magnitudes.

That is strong but not authoritative, so the formula is used for **local model
selection only** and never as a claim about the official metric. Every decision
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
| `src/features.py` | feature blocks; `base` is what ships |
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
| `src/stability.py` | seed-stability analysis and the 20-seed bag |
| `src/prior_adjust.py` | 3-parameter class-prior adjustment, cross-fold |
| `reports/notes.md` | every variant tried, kept or dropped, with the reasoning |
| `reports/submissions_log.md` | every file in `submissions/`, with validation and public scores |
| `docs/` | official label rules and data dictionary |

Nothing in the **Record** group is imported by `make_submission.py`; they are
kept deliberately, because they are the evidence behind 31 dropped variants and
a reviewer should be able to check the negative results, not just the final
model.

`data/`, `submissions/` and `preds/` are local only and git-ignored. Every file
in `submissions/` is named after the run that produced it and has a row in
`reports/submissions_log.md`.

## Rules to remember

Solo only, no private code sharing. Provided data only. No AutoML. Always set
the seed. 10 submissions a day, 200 total, 2 final picks. Keep this repo private.
