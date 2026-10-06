# Sasol Customer Value Recruitment Challenge (Zindi)

Solo entry. Predict, per customer, for Dec 2025 to Feb 2026:
fuel litres (`CLV_fuel`), non-fuel rands (`CLV_nonfuel`), and an `Opportunity` label.
Score = 0.3 RMSE fuel + 0.3 RMSE non-fuel + 0.4 weighted F1. Closes 19 Oct 2026.

## Setup
```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```
Copy `train.csv`, `test.csv` and `SampleSubmission.csv` into `data/`.

## Run the baseline
```powershell
python src/baseline.py
```
Prints validation scores (1 Sep 2025 snapshot) and writes
`submissions/submission_baseline_v1.csv`.

Every file in `submissions/` is named after the run that made it and has a row in
`reports/submissions_log.md`; nothing writes an unnamed `submission.csv`.

## Layout
| Path | What |
| --- | --- |
| `src/baseline.py` | Monthly snapshots, customer features, LightGBM |
| `src/label_rules.py`, `src/label_config.json` | Official label generator and config (unchanged) |
| `docs/` | Official label rules and data dictionary |
| `reports/submissions_log.md` | Every submission, with validation and public score |
| `reports/notes.md` | Data findings and plan |
| `data/`, `submissions/`, `preds/` | Local only, ignored by git |

## Rules to remember
Solo only, no private code sharing. Provided data only. No AutoML. Always set the seed (42).
10 submissions a day, 200 total, 2 final picks. Keep this repo private.
