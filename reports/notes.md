# Notes

## Data findings (5 Oct)
- 467,880 lines, 6,204 customers, 390 sites, 19 Mar 2024 to 30 Nov 2025.
- Test = 5,488 customers = everyone with 3+ baskets before 1 Dec 2025.
- `label_rules.py` reproduces the published `s` constants exactly, so we can label any past cutoff.
- 73% of rows are fuel. Per quarter about 29% of customers have zero fuel and 65% zero non-fuel.
- Labels (latest snapshot): Fuel growth 27%, Inactivity 27%, Stable 27%; each adoption class about 1% or less.
- Inactivity rises every quarter (7% to 27%); expect about 29% in test.
- Early snapshots have short history, so adoption is inflated there (`hist_days` feature covers this).

## Plan
1. Label model: growth vs stable vs inactive. Key signal: was last quarter unusually low.
2. Derive the label from predicted per-category spend, blend with the direct classifier.
3. Two-stage models for the numbers (buys at all, then how much).
4. Seasonality: only one past Dec to Feb window (cutoff 1 Dec 2024).
5. Ensemble LightGBM, CatBoost and seeds. Test everything on validation before submitting.

## Open questions
- Does higher or lower win on the leaderboard?
- Does public score track validation?
