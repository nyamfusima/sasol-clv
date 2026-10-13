"""Check a written submission file, and diff it against a reference file.

`make_submission` asserts its format before writing; this reads the file back
off disk and checks it independently, then reports the comparisons every
submissions_log row quotes -- so those numbers come from a committed script
rather than from an ad-hoc session.

    python src/check_submission.py submissions/submission_v6_reg.csv
    python src/check_submission.py submissions/submission_v6_reg.csv \
        --vs submissions/submission_v5_fuel2.csv
    # expected public score if a validation delta transferred 1:1
    python src/check_submission.py submissions/submission_v6_reg.csv \
        --vs submissions/submission_v5_fuel2.csv \
        --ref-public 0.309785160 --val-delta 0.00289

CSV floats are written with %.17g and must be read with
float_precision='round_trip'; pandas' default parser shifts values by 1 ULP on
every read, which would make a byte-identity check lie.
"""
import argparse, hashlib, json
from pathlib import Path
import numpy as np
import pandas as pd

COLS = ['ID', 'CLV_fuel', 'CLV_nonfuel', 'Opportunity']
CLV = ['CLV_fuel', 'CLV_nonfuel']
N_ROWS = 5488
FUEL_NORM, NONFUEL_NORM = 0.74, 0.816


def read(path):
    return pd.read_csv(path, dtype={'ID': str}, float_precision='round_trip')


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def check_format(df, labels, sample=None):
    """Every assertion make_submission makes, re-run against the file on disk."""
    out = []
    ok = list(df.columns) == COLS
    out.append((ok, f'columns {list(df.columns)}'))
    out.append((len(df) == N_ROWS, f'{len(df)} rows (expected {N_ROWS})'))
    out.append((bool(df.notna().all().all()), 'no missing values'))
    bad = sorted(set(df.Opportunity) - set(labels))
    out.append((not bad, f'labels within the 17 official classes'
                         + (f' -- OUTSIDE: {bad}' if bad else '')))
    out.append((bool((df[CLV] >= 0).all().all()),
                f'CLV >= 0 (min {df[CLV].min().min():.6g})'))
    out.append((df.ID.is_unique, f'{df.ID.nunique()} unique IDs'))
    if sample is not None:
        out.append((df.ID.equals(sample.ID), 'IDs match SampleSubmission, in order'))
        out.append((list(df.columns) == list(sample.columns),
                    'column names match SampleSubmission'))
    return out


def compare(df, ref, name_a, name_b):
    """Label churn, CLV movement, and exact per-column identity."""
    a = df.set_index('ID').reindex(ref.ID)
    b = ref.set_index('ID')
    print(f'\n  --- {name_a} vs {name_b} ---')
    same = (a.Opportunity.to_numpy() == b.Opportunity.to_numpy())
    n = int((~same).sum())
    print(f'  labels changed      {n} of {len(b)} ({100 * n / len(b):.2f}%)'
          f'{"   [IDENTICAL]" if n == 0 else ""}')
    for c in CLV:
        d = a[c].to_numpy() - b[c].to_numpy()
        ident = bool((d == 0).all())
        print(f'  {c:<19} mean |change| {np.abs(d).mean():.6g}   '
              f'max |change| {np.abs(d).max():.6g}'
              f'{"   [BYTE-IDENTICAL]" if ident else ""}')
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('path')
    ap.add_argument('--vs', default=None, help='reference submission to diff against')
    ap.add_argument('--sample', default='data/SampleSubmission.csv')
    ap.add_argument('--config', default='src/label_config.json')
    ap.add_argument('--ref-public', type=float, default=None,
                    help="reference file's measured public score")
    ap.add_argument('--val-delta', type=float, default=None,
                    help='validation gain over the reference, for a 1:1 projection')
    a = ap.parse_args()

    labels = json.loads(Path(a.config).read_text())['opportunity_labels']
    df = read(a.path)
    sample = read(a.sample) if Path(a.sample).exists() else None

    print(f'{a.path}')
    print(f'  sha256 {sha256(a.path)}')
    rows = check_format(df, labels, sample)
    for ok, msg in rows:
        print(f'  [{"ok" if ok else "FAIL"}] {msg}')
    print(f'  label mix: ' + ', '.join(
        f'{k} {v}' for k, v in df.Opportunity.value_counts().head(4).items()))

    if a.vs:
        compare(df, read(a.vs), Path(a.path).name, Path(a.vs).name)

    if a.ref_public is not None and a.val_delta is not None:
        print(f'\n  reference public {a.ref_public:.9f}, validation delta '
              f'{a.val_delta:+.5f}')
        print(f'  expected public if validation transfers 1:1  '
              f'{a.ref_public + a.val_delta:.9f}')

    failed = [m for ok, m in rows if not ok]
    print(f'\n  {"ALL CHECKS PASS" if not failed else "FAILED: " + "; ".join(failed)}')
    return 1 if failed else 0


if __name__ == '__main__':
    raise SystemExit(main())
