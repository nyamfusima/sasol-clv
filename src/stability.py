"""Seed stability of the v2 stack: score spread and label-flip rate by bag size.

Each seed is fitted ONCE and its raw outputs stored, so any bag composes exactly
by averaging. That matters for the hurdle: a true 20-seed hurdle averages the
gate over 20 seeds and the magnitude over 20 seeds and then multiplies, which is
NOT the same as averaging four 5-seed hurdle outputs (product of means vs mean of
products). Composing from per-seed pieces avoids that error.

Seeds 42-66 get the full stack on both folds and the test cutoff. Seeds 67-81 get
the classifier on the test cutoff only, which is all the second 20-seed bag needs
for an honest flip-rate comparison against bag20 (42-61).

    python src/stability.py
"""
import argparse, json, time
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.metrics import f1_score
import make_submission as M
import snapshots as S
import validate as V

FULL = tuple(range(42, 67))        # 25 seeds: folds + test, classifier + hurdle
CLF_ONLY = tuple(range(67, 82))    # 15 seeds: classifier on test only
SETS = [tuple(range(s, s + 5)) for s in range(42, 82, 5)]   # eight disjoint 5-seed sets
OUT = Path('preds/stability.json')


def fit_one_seed(ctx, seed, want_reg=True, want_folds=True):
    """Raw per-seed outputs: classifier probabilities and, optionally, the two
    hurdle components for each target, on both folds and the test cutoff."""
    out = {}
    jobs = ([(v, ctx['tr'][v], ctx['Xva'][v]) for v in V.FOLDS] if want_folds else [])
    jobs.append(('test', ctx['all'], ctx['Xte']))
    for name, (Xtr, ytr), Xva in jobs:
        rec = {'clf': M.bag_clf(Xtr, ytr.Opportunity.map(ctx['code']).to_numpy(),
                                Xva, ctx['labels'], (seed,)).to_numpy()}
        if want_reg:
            for t in M.TARGETS:
                y = ytr[t]
                pos = (y.to_numpy() > 0)
                rec[f'gate_{t}'] = M.bag_binary(Xtr, pos.astype(int), Xva, (seed,))
                rec[f'mag_{t}'] = M.bag_reg(Xtr[pos], y[pos], Xva, (seed,))
        out[name] = rec
    return out


def compose(per_seed, seeds, name, labels, want_reg=True):
    """Average the stored per-seed pieces into one bag of arbitrary size."""
    seeds = [s for s in seeds if s in per_seed and name in per_seed[s]]
    clf = np.mean([per_seed[s][name]['clf'] for s in seeds], axis=0)
    opp = np.array(labels)[clf.argmax(axis=1)]
    reg = {}
    if want_reg:
        for t in M.TARGETS:
            g = np.mean([per_seed[s][name][f'gate_{t}'] for s in seeds], axis=0)
            m = np.mean([per_seed[s][name][f'mag_{t}'] for s in seeds], axis=0)
            reg[t] = np.clip(g * m, 0, None)
    return opp, reg, clf


def score(opp, reg, yva):
    f1 = f1_score(yva.Opportunity, opp, average='weighted', zero_division=0)
    rf = float(np.sqrt(((reg['CLV_fuel'] - yva.CLV_fuel) ** 2).mean()))
    rn = float(np.sqrt(((reg['CLV_nonfuel'] - yva.CLV_nonfuel) ** 2).mean()))
    return f1, rf, rn, M.combined(f1, rf, rn)


def flip_rate(labels_by_build):
    """Mean pairwise share of test customers whose Opportunity differs."""
    keys = list(labels_by_build)
    rates = []
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            a, b = labels_by_build[keys[i]], labels_by_build[keys[j]]
            rates.append(float((a != b).mean()))
    return float(np.mean(rates)), float(np.min(rates)), float(np.max(rates)), len(rates)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--train', default='data/train.csv')
    ap.add_argument('--config', default='src/label_config.json')
    ap.add_argument('--out', default='submissions/submission_v2_bag20.csv')
    a = ap.parse_args()
    t0 = time.time()
    cfg = json.loads(Path(a.config).read_text())
    labels = cfg['opportunity_labels']
    cutoffs = M.monthly()
    snaps = S.load(cutoffs + [M.TEST_CUTOFF], a.train, cfg, ('base',),
                   label_cutoffs=cutoffs, verbose=False)
    import train_label as T
    bc = T.block_columns(list(snaps[cutoffs[0]][0].columns))
    sel = lambda c: T.select(snaps[c][0], ('base',), bc)
    ctx = dict(labels=labels, code={l: i for i, l in enumerate(labels)},
               Xte=sel(M.TEST_CUTOFF),
               all=(pd.concat([sel(c) for c in cutoffs]),
                    pd.concat([snaps[c][1] for c in cutoffs])),
               tr={v: (pd.concat([sel(c) for c in M.trainable_for(cutoffs, v)]),
                       pd.concat([snaps[c][1] for c in M.trainable_for(cutoffs, v)]))
                   for v in V.FOLDS},
               Xva={v: sel(v) for v in V.FOLDS})
    yva = {v: snaps[v][1] for v in V.FOLDS}

    per_seed = {}
    for s in FULL + CLF_ONLY:
        full = s in FULL
        ts = time.time()
        per_seed[s] = fit_one_seed(ctx, s, want_reg=full, want_folds=full)
        print(f'  seed {s} ({"full" if full else "clf/test only"}) '
              f'{time.time() - ts:.0f}s', flush=True)

    # --- five 5-seed builds -------------------------------------------------
    print('\n=== five 5-seed builds (as v2, different seed sets) ===')
    five = SETS[:5]
    rows, test_labels = [], {}
    for st in five:
        per_fold = [score(*compose(per_seed, st, v, labels)[:2], yva[v]) for v in V.FOLDS]
        mean_sc = float(np.mean([p[3] for p in per_fold]))
        opp, _, _ = compose(per_seed, st, 'test', labels, want_reg=False)
        test_labels[f'{st[0]}-{st[-1]}'] = opp
        rows.append((f'{st[0]}-{st[-1]}', per_fold[0], per_fold[1], mean_sc))
        print(f'  seeds {st[0]}-{st[-1]}: fold1 score {per_fold[0][3]:.5f} | '
              f'fold2 {per_fold[1][3]:.5f} | mean {mean_sc:.5f} '
              f'(F1 {per_fold[0][0]:.4f}/{per_fold[1][0]:.4f})')
    means = [r[3] for r in rows]
    print(f'  score spread over 5 builds: min {min(means):.5f} max {max(means):.5f} '
          f'range {max(means) - min(means):.5f} sd {np.std(means):.5f}')
    f5 = flip_rate(test_labels)
    print(f'  Opportunity flip rate between 5-seed builds: mean {f5[0]:.4%} '
          f'(min {f5[1]:.4%}, max {f5[2]:.4%}, {f5[3]} pairs)')

    # --- bag20 --------------------------------------------------------------
    print('\n=== 20-seed bag (42-61) ===')
    bag20 = tuple(range(42, 62))
    b20 = [score(*compose(per_seed, bag20, v, labels)[:2], yva[v]) for v in V.FOLDS]
    b20_mean = float(np.mean([p[3] for p in b20]))
    for i, v in enumerate(V.FOLDS):
        print(f'  fold {v}: F1 {b20[i][0]:.4f} | rmse_fuel {b20[i][1]:.4f} | '
              f'rmse_nonfuel {b20[i][2]:.4f} | score {b20[i][3]:.5f}')
    print(f'  mean score {b20_mean:.5f}   (v2 5-seed 42-46 was {rows[0][3]:.5f})')

    # honest flip-rate comparison: two DISJOINT 20-seed bags
    t20 = {}
    for lo in (42, 62):
        opp, _, _ = compose(per_seed, tuple(range(lo, lo + 20)), 'test', labels,
                            want_reg=False)
        t20[f'{lo}-{lo + 19}'] = opp
    f20 = flip_rate(t20)
    print(f'  flip rate between two disjoint 20-seed bags: {f20[0]:.4%}')
    print(f'  vs 5-seed builds {f5[0]:.4%}  ->  '
          f'{(1 - f20[0] / f5[0]) * 100:.0f}% lower' if f5[0] else '')

    # write bag20
    opp, reg, _ = compose(per_seed, bag20, 'test', labels)
    test_ids = pd.read_csv('data/test.csv', dtype=str).ID
    sub = pd.DataFrame({'CLV_fuel': reg['CLV_fuel'], 'CLV_nonfuel': reg['CLV_nonfuel'],
                        'Opportunity': opp}, index=ctx['Xte'].index).reindex(test_ids)
    assert len(sub) == 5488 and sub.index.equals(pd.Index(test_ids))
    assert sub.notna().all().all() and not set(sub.Opportunity) - set(labels)
    assert (sub[['CLV_fuel', 'CLV_nonfuel']] >= 0).all().all()
    assert list(sub.columns) == ['CLV_fuel', 'CLV_nonfuel', 'Opportunity']
    Path(a.out).parent.mkdir(exist_ok=True)
    sub.rename_axis('ID').reset_index().to_csv(a.out, index=False)
    print(f'  wrote {a.out}: 5488 rows, IDs match, labels in config, CLV >= 0')

    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(dict(
        five=[dict(seeds=r[0], fold1=list(r[1]), fold2=list(r[2]), mean_score=r[3])
              for r in rows],
        spread=dict(min=min(means), max=max(means), range=max(means) - min(means),
                    sd=float(np.std(means))),
        flip_5seed=dict(mean=f5[0], min=f5[1], max=f5[2], pairs=f5[3]),
        bag20=dict(fold1=list(b20[0]), fold2=list(b20[1]), mean_score=b20_mean),
        flip_20seed=dict(mean=f20[0]),
    ), indent=2))
    # bag20 class probabilities: both validation folds (out-of-fold) and the
    # test cutoff, for the prior-adjustment step
    for v in V.FOLDS:
        _, _, clf = compose(per_seed, bag20, v, labels)
        pd.DataFrame(clf, index=ctx['Xva'][v].index, columns=labels).rename_axis('ID') \
          .reset_index().to_parquet(f'preds/oof_bag20_{v}.parquet', index=False)
    _, _, clf_te = compose(per_seed, bag20, 'test', labels)
    pd.DataFrame(clf_te, index=ctx['Xte'].index, columns=labels).rename_axis('ID') \
      .reset_index().to_parquet('preds/oof_bag20_test.parquet', index=False)
    print('  wrote preds/stability.json and bag20 probabilities (both folds + test)')
    print(f'total runtime {time.time() - t0:.0f}s')


if __name__ == '__main__':
    main()
