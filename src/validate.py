"""Two time-based validation folds, plus the reporting every experiment uses.

Fold k validates on one snapshot and trains only on snapshots whose 3-month
outcome window ends on or before that cutoff, so no training outcome can overlap
the outcome being scored:

    fold 1  validate 2025-06-01  train cutoffs <= 2025-03-01
    fold 2  validate 2025-09-01  train cutoffs <= 2025-06-01

Fold 2 is the baseline's own split, so its numbers are directly comparable.

    python src/validate.py            # baseline model on both folds
"""
import argparse, json
import numpy as np, pandas as pd
from sklearn.metrics import f1_score, confusion_matrix
import snapshots as S

FOLDS = ['2025-06-01', '2025-09-01']
LAST_CUTOFF = '2025-09-01'
FIRST_CUTOFF = '2024-06-01'
GROUPS = ['Fuel growth', 'Inactivity', 'Stable', 'Other']

# Leaderboard formula. Higher is better.
#   Score = 0.4*F1 + 0.3*(1 - RMSE_fuel/0.74) + 0.3*(1 - RMSE_nonfuel/0.816)
# Gradients: +0.4 per unit F1, -0.3/0.74 = -0.4054 per unit RMSE_fuel,
# -0.3/0.816 = -0.3676 per unit RMSE_nonfuel. A 0.005 RMSE drop is therefore
# worth about as much as a 0.005 F1 gain; every decision uses `combined`.
FUEL_NORM = 0.74
NONFUEL_NORM = 0.816
W_F1, W_FUEL, W_NONFUEL = 0.4, 0.3, 0.3


def combined(f1_, rmse_fuel, rmse_nonfuel):
    return (W_F1 * f1_
            + W_FUEL * (1 - rmse_fuel / FUEL_NORM)
            + W_NONFUEL * (1 - rmse_nonfuel / NONFUEL_NORM))


def score_row(name, f1_, rmse_fuel, rmse_nonfuel, ref=None):
    """One line per variant on the single number that decides things."""
    s = combined(f1_, rmse_fuel, rmse_nonfuel)
    d = '' if ref is None else f'  ({s - ref:+.5f} vs ref)'
    return (f'{name:<34} F1 {f1_:.4f} | rmse_f {rmse_fuel:.4f} | '
            f'rmse_nf {rmse_nonfuel:.4f} | score {s:.5f}{d}')


def group(labels):
    """Collapse the 17 classes to the four buckets worth eyeballing."""
    s = pd.Series(labels).astype(str)
    return np.select([s.eq('Existing-category growth: Fuel'), s.eq('Inactivity'), s.eq('Stable')],
                     ['Fuel growth', 'Inactivity', 'Stable'], default='Other')


def folds(blocks=('base',), train_path='data/train.csv', cfg=None, refresh=False, verbose=True):
    """[(val_cutoff, (Xtr, ytr), (Xva, yva)), ...] for both folds."""
    cfg = cfg or json.loads(open('src/label_config.json').read())
    cutoffs = S.monthly(FIRST_CUTOFF, LAST_CUTOFF)
    snaps = S.load(cutoffs, train_path, cfg, blocks, refresh=refresh, verbose=verbose)
    out = []
    for v in FOLDS:
        tr = S.trainable_for(cutoffs, v)
        Xtr = pd.concat([snaps[c][0] for c in tr])
        ytr = pd.concat([snaps[c][1] for c in tr])
        out.append((v, (Xtr, ytr), snaps[v]))
    return out


def f1(y_true, y_pred):
    return f1_score(y_true, y_pred, average='weighted', zero_division=0)


def regression_scores(pred, yva):
    return (np.sqrt(((pred.CLV_fuel - yva.CLV_fuel) ** 2).mean()),
            np.sqrt(((pred.CLV_nonfuel - yva.CLV_nonfuel) ** 2).mean()))


def per_class(y_true, y_pred, labels):
    """Per-class F1 with support, ordered by support."""
    s = f1_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
    n_true = pd.Series(y_true).value_counts().reindex(labels, fill_value=0)
    n_pred = pd.Series(y_pred).value_counts().reindex(labels, fill_value=0)
    return pd.DataFrame({'F1': s, 'support': n_true.to_numpy(), 'predicted': n_pred.to_numpy()},
                        index=labels).sort_values('support', ascending=False)


def grouped_report(y_true, y_pred):
    """Weighted F1 and a confusion matrix over the four buckets."""
    gt, gp = group(y_true), group(y_pred)
    cm = pd.DataFrame(confusion_matrix(gt, gp, labels=GROUPS), index=GROUPS, columns=GROUPS)
    cm.index.name = 'true \\ pred'
    per = f1_score(gt, gp, labels=GROUPS, average=None, zero_division=0)
    return cm, pd.Series(per, index=GROUPS, name='F1')


def report(name, results, labels, show_detail=True):
    """results: {val_cutoff: (y_true, y_pred)}. Returns mean weighted F1."""
    scores = {v: f1(a, b) for v, (a, b) in results.items()}
    mean = float(np.mean(list(scores.values())))
    print(f'\n=== {name} ===')
    for v, s in scores.items():
        print(f'  fold {v}: weighted F1 {s:.4f}')
    print(f'  mean weighted F1: {mean:.4f}')
    if show_detail:
        for v, (a, b) in results.items():
            print(f'\n  -- fold {v} per-class F1 --')
            print(per_class(a, b, labels).to_string(float_format=lambda x: f'{x:.4f}'))
            cm, g = grouped_report(a, b)
            print(f'\n  -- fold {v} grouped confusion matrix --')
            print(cm.to_string())
            print('  grouped F1: ' + ' | '.join(f'{k} {x:.4f}' for k, x in g.items()))
    return mean, scores


def main():
    import lightgbm as lgb
    from baseline import fit_predict
    ap = argparse.ArgumentParser()
    ap.add_argument('--train', default='data/train.csv')
    ap.add_argument('--config', default='src/label_config.json')
    ap.add_argument('--refresh', action='store_true')
    a = ap.parse_args()
    cfg = json.load(open(a.config)); labels = cfg['opportunity_labels']
    results, reg = {}, {}
    for v, (Xtr, ytr), (Xva, yva) in folds(('base',), a.train, cfg, a.refresh):
        p = fit_predict(Xtr, ytr, Xva, labels)
        results[v] = (yva.Opportunity.to_numpy(), p.Opportunity.to_numpy())
        reg[v] = regression_scores(p, yva)
        print(f'fold {v}: {len(Xtr)} train rows ({len(S.trainable_for(S.monthly(FIRST_CUTOFF, LAST_CUTOFF), v))} snapshots), '
              f'{len(Xva)} val rows -> RMSE fuel {reg[v][0]:.4f} | RMSE nonfuel {reg[v][1]:.4f}')
    mean, _ = report('baseline (base features, 17-class LGBM)', results, labels)
    print(f'\nmean RMSE fuel {np.mean([r[0] for r in reg.values()]):.4f} | '
          f'mean RMSE nonfuel {np.mean([r[1] for r in reg.values()]):.4f} | mean F1 {mean:.4f}')


if __name__ == '__main__':
    main()
