"""Block I: better targets for partial prior matching.

H3 matched the predicted class mix to the last FULLY OBSERVED snapshot's mix and
damped it by alpha=0.5, scoring 0.28929. That target is stale: the class mix
drifts monotonically (Inactivity rises in 16 of 16 snapshots), so the last
observed quarter understates where the validation quarter actually sits. This
block projects the target forward instead, using only snapshots whose outcomes
are observable before the cutoff.

  I1  projected target mixes, with per-class error against the TRUE fold mix
  I2  alpha grid per target, alpha chosen on one fold and scored on the other
  I3  probability calibration before matching (isotonic; global temperature is
      algebraically identical to alpha and is demonstrated rather than fitted)
  I4  distance to externally supplied reference shares -- REPORTED ONLY, and
      computed after selection is frozen so it cannot influence anything

    python src/project_mix.py
"""
import argparse, json
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.isotonic import IsotonicRegression
import calibrate as C
import make_submission as M
import snapshots as S
import train_label as T
import validate as V

BIG = C.BIG
ALPHAS = (0.25, 0.5, 0.75, 1.0)
METHODS = ('stale', 'lin4', 'lin6', 'qoq', 'damped')
# I4 reference shares, supplied externally. Used for a distance report only.
REFERENCE = {'Stable': 0.276, 'Inactivity': 0.297, 'Existing-category growth: Fuel': 0.252}
# earlier snapshot used to FIT calibration for each target, always fully
# observed before that target's cutoff
CALIB_SOURCE = {V.FOLDS[0]: '2025-03-01', V.FOLDS[1]: V.FOLDS[0],
                M.TEST_CUTOFF: V.FOLDS[1]}


def months(a, b):
    a, b = pd.Timestamp(a), pd.Timestamp(b)
    return (b.year - a.year) * 12 + (b.month - a.month)


def share_matrix(snaps, cutoffs, labels):
    return {c: np.array([snaps[c][1].Opportunity.value_counts(normalize=True).get(l, 0.0)
                         for l in labels]) for c in cutoffs}


def project(shares, avail, for_cutoff, labels, method):
    """Target mix for `for_cutoff` from the snapshots in `avail` (all fully
    observed before it). Only the four large classes are projected; the small
    classes stay at their last-observed shares and the large ones are
    renormalised so the whole vector sums to one."""
    big = [labels.index(l) for l in BIG]
    hist = np.array([shares[c] for c in avail])
    last = hist[-1].copy()
    out = last.copy()
    h = months(avail[-1], for_cutoff)
    if method != 'stale':
        for j in big:
            if method in ('lin4', 'lin6', 'damped'):
                k = min(6 if method == 'lin6' else 4, len(hist))
                y = hist[-k:, j]
                slope = float(np.polyfit(np.arange(k), y, 1)[0])
                step = slope * h
                out[j] = last[j] + (0.5 * step if method == 'damped' else step)
            elif method == 'qoq':
                d = [hist[i, j] - hist[i - 3, j] for i in range(3, len(hist))]
                out[j] = last[j] + (float(np.mean(d)) if d else 0.0)
    out = np.clip(out, 1e-6, 1.0)
    small = float(sum(last[j] for j in range(len(labels)) if j not in big))
    bs = float(sum(out[j] for j in big))
    if bs > 0:
        for j in big:
            out[j] *= (1.0 - small) / bs
    for j in range(len(labels)):
        if j not in big:
            out[j] = last[j]
    return out


def oof_for(cutoff, cfg, labels, seeds=tuple(range(42, 62))):
    """bag20 classifier probabilities for `cutoff`, trained only on snapshots
    whose outcome window ends on or before it. Cached."""
    p = Path(f'preds/oof_bag20_{cutoff}.parquet')
    if p.exists():
        return pd.read_parquet(p).set_index('ID')
    cutoffs = M.monthly()
    tr = M.trainable_for(cutoffs, cutoff)
    snaps = S.load(cutoffs, 'data/train.csv', cfg, ('base',), verbose=False)
    bc = T.block_columns(list(snaps[cutoffs[0]][0].columns))
    sel = lambda c: T.select(snaps[c][0], ('base',), bc)
    Xtr = pd.concat([sel(c) for c in tr])
    ytr = pd.concat([snaps[c][1] for c in tr])
    Xva = sel(cutoff)
    print(f'  building OOF probabilities for {cutoff}: {len(tr)} snapshots, '
          f'{len(Xtr)} rows, {len(seeds)} seeds', flush=True)
    code = {l: i for i, l in enumerate(labels)}
    pr = M.bag_clf(Xtr, ytr.Opportunity.map(code).to_numpy(), Xva, labels, seeds)
    pr.rename_axis('ID').reset_index().to_parquet(p, index=False)
    return pr.rename_axis('ID')


def isotonic_fit(proba_src, y_src, labels):
    """One isotonic map per class, fitted on an earlier snapshot's OOF
    probabilities against its observed one-hot outcomes."""
    models = {}
    for j, l in enumerate(labels):
        yb = (y_src == l).astype(float)
        if yb.sum() < 10 or yb.sum() == len(yb):
            models[j] = None
            continue
        ir = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds='clip')
        ir.fit(proba_src[:, j], yb)
        models[j] = ir
    return models


def isotonic_apply(proba, models):
    out = proba.copy()
    for j, ir in models.items():
        if ir is not None:
            out[:, j] = ir.predict(proba[:, j])
    s = out.sum(axis=1, keepdims=True)
    return np.divide(out, s, out=np.full_like(out, 1.0 / out.shape[1]), where=s > 0)


def _verify_clv_matches(out, bag_path):
    """Re-read both files and confirm the CLV columns are bit-identical. Reading
    with float_precision='round_trip' matters: pandas' default CSV float parser is
    not correctly rounded and shifts values by 1 ULP per read/rewrite cycle."""
    import pandas as pd
    rd = lambda p: pd.read_csv(p, dtype={'ID': str}, float_precision='round_trip')
    a, b = rd(out), rd(bag_path)
    assert a.ID.equals(b.ID), 'ID order differs'
    assert a[['CLV_fuel', 'CLV_nonfuel']].equals(b[['CLV_fuel', 'CLV_nonfuel']]),         'CLV columns are not bit-identical to the source'
    print(f'  verified: {out} CLV columns bit-identical to {bag_path}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', default='src/label_config.json')
    ap.add_argument('--bag20', default='submissions/submission_v2_bag20.csv')
    ap.add_argument('--out', default='submissions/submission_v4.csv')
    a = ap.parse_args()
    cfg = json.loads(Path(a.config).read_text())
    labels = cfg['opportunity_labels']
    big = [labels.index(l) for l in BIG]
    cutoffs = M.monthly()
    snaps = S.load(cutoffs, 'data/train.csv', cfg, ('base',), verbose=False)
    shares = share_matrix(snaps, cutoffs, labels)

    P, Y = {}, {}
    for v in V.FOLDS:
        pr = pd.read_parquet(f'preds/oof_bag20_{v}.parquet').set_index('ID')
        y = snaps[v][1]
        P[v] = pr.reindex(y.index)[labels].to_numpy()
        Y[v] = y.Opportunity.to_numpy()
    true_mix = {v: C.mix(Y[v], labels) for v in V.FOLDS}

    iF = labels.index(C.FUEL)
    base_w = np.ones(len(labels)); base_w[iF] = C.BASE_FUEL
    b = json.loads(Path('preds/stability.json').read_text())['bag20']
    rmses = {V.FOLDS[0]: (b['fold1'][1], b['fold1'][2]),
             V.FOLDS[1]: (b['fold2'][1], b['fold2'][2])}
    sc = lambda f1s: float(np.mean([M.combined(f1s[i], *rmses[v])
                                    for i, v in enumerate(V.FOLDS)]))
    # H3 incumbent: stale target, 4 large classes, alpha 0.5
    inc_f1 = []
    for v in V.FOLDS:
        tgt = project(shares, M.trainable_for(cutoffs, v), v, labels, 'stale')
        w, _ = C.ipf(P[v], tgt, labels, big)
        inc_f1.append(C.f1w(Y[v], C.ap(P[v], w ** 0.5, labels)))
    inc_score = sc(inc_f1)
    print(f'incumbent H3 (stale target, 4 large, alpha 0.5): F1 '
          f'{inc_f1[0]:.4f} / {inc_f1[1]:.4f} | score {inc_score:.5f}')

    # -- I1 -----------------------------------------------------------------
    print(f'\n{"=" * 100}\nI1  PROJECTED TARGET MIXES: absolute error per class against the '
          f'TRUE fold mix\n{"=" * 100}')
    targets = {}
    for v in V.FOLDS:
        avail = M.trainable_for(cutoffs, v)
        print(f'\n  fold {v} (projecting {months(avail[-1], v)} months from {avail[-1]}, '
              f'{len(avail)} snapshots available)')
        print(f'    {"method":<10}' + ''.join(f'{l.split(": ")[-1][:9]:>11}' for l in BIG)
              + f'{"sum|err|":>11}')
        print(f'    {"TRUE":<10}' + ''.join(f'{true_mix[v][j]:>11.3f}' for j in big) + f'{"-":>11}')
        for m in METHODS:
            t = project(shares, avail, v, labels, m)
            targets[(v, m)] = t
            err = float(np.abs(t[big] - true_mix[v][big]).sum())
            print(f'    {m:<10}' + ''.join(f'{t[j]:>11.3f}' for j in big) + f'{err:>11.3f}')

    # -- I2 -----------------------------------------------------------------
    print(f'\n{"=" * 100}\nI2  ALPHA GRID per target, alpha chosen on one fold and scored '
          f'on the other\n{"=" * 100}')
    print(f'{"target":<10}{"alpha":>7}{"fold1":>9}{"fold2":>9}{"mean":>9}{"score":>10}'
          f'{"vs H3":>10}  kept')
    results, best_alpha = [], {}
    W = {}
    for m in METHODS:
        for v in V.FOLDS:
            W[(v, m)], _ = C.ipf(P[v], targets[(v, m)], labels, big)
        for al in ALPHAS:
            f1s = [C.f1w(Y[v], C.ap(P[v], W[(v, m)] ** al, labels)) for v in V.FOLDS]
            s = sc(f1s)
            both = all(f1s[i] > inc_f1[i] for i in (0, 1))
            gain = s - inc_score
            keep = 'KEPT' if (gain >= 0.0015 and both) else 'dropped'
            results.append(dict(method=m, alpha=al, f1=f1s, score=s, gain=gain, keep=keep))
            print(f'{m:<10}{al:>7}{f1s[0]:>9.4f}{f1s[1]:>9.4f}{np.mean(f1s):>9.4f}'
                  f'{s:>10.5f}{gain:>+10.5f}  {keep}')
        picks = {}
        for fit_on in V.FOLDS:
            other = [x for x in V.FOLDS if x != fit_on][0]
            pa = max(ALPHAS, key=lambda al: C.f1w(Y[fit_on],
                                                  C.ap(P[fit_on], W[(fit_on, m)] ** al, labels)))
            picks[other] = (pa, C.f1w(Y[other], C.ap(P[other], W[(other, m)] ** pa, labels)))
        best_alpha[m] = picks
        agree = len({p[0] for p in picks.values()}) == 1
        print(f'  -> cross-fold alpha: ' + ' | '.join(
            f'{p[0]} (held-out {k} F1 {p[1]:.4f})' for k, p in picks.items())
            + ('  AGREE' if agree else '  DISAGREE'))

    # -- I3 -----------------------------------------------------------------
    print(f'\n{"=" * 100}\nI3  CALIBRATION BEFORE MATCHING\n{"=" * 100}')
    print('  Global temperature is algebraically identical to alpha: argmax_c '
          'p_c^(1/T) * w_c = argmax_c p_c * w_c^T,')
    print('  so a temperature T with weights w is exactly weights w**T. '
          'Demonstrated, not fitted:')
    for m in ['lin4']:
        for T_, al in ((2.0, 0.5), (1.0, 1.0)):
            f1s = [C.f1w(Y[v], C.ap(P[v] ** (1 / T_), W[(v, m)] ** al, labels))
                   for v in V.FOLDS]
            print(f'    T={T_} with alpha={al}: F1 {f1s[0]:.4f} / {f1s[1]:.4f}  '
                  f'(effective alpha {al * T_})')
    print('\n  isotonic per class, fitted on an earlier snapshot, then matching:')
    iso_rows = []
    for m in METHODS:
        f1s = []
        for v in V.FOLDS:
            src = CALIB_SOURCE[v]
            ps = oof_for(src, cfg, labels)
            ys = snaps[src][1]
            ps = ps.reindex(ys.index)[labels].to_numpy()
            models = isotonic_fit(ps, ys.Opportunity.to_numpy(), labels)
            pc = isotonic_apply(P[v], models)
            w, _ = C.ipf(pc, targets[(v, m)], labels, big)
            al = best_alpha[m][[x for x in V.FOLDS if x != v][0]][0]
            f1s.append(C.f1w(Y[v], C.ap(pc, w ** al, labels)))
        s = sc(f1s)
        gain = s - inc_score
        both = all(f1s[i] > inc_f1[i] for i in (0, 1))
        keep = 'KEPT' if (gain >= 0.0015 and both) else 'dropped'
        iso_rows.append(dict(method=m, f1=f1s, score=s, gain=gain, keep=keep))
        print(f'    isotonic + {m:<8} F1 {f1s[0]:.4f} / {f1s[1]:.4f} | score {s:.5f} '
              f'({gain:+.5f})  {keep}')

    # -- freeze selection, THEN report I4 ------------------------------------
    kept = [r for r in results if r['keep'] == 'KEPT'] + \
           [r for r in iso_rows if r['keep'] == 'KEPT']
    chosen = None
    if kept:
        # restrict to variants whose cross-fold alpha agreed across folds
        ok = [r for r in kept if 'alpha' in r and
              len({p[0] for p in best_alpha[r['method']].values()}) == 1 and
              r['alpha'] == list(best_alpha[r['method']].values())[0][0]]
        chosen = max(ok or kept, key=lambda r: r['score'])
    print(f'\n{"=" * 100}\nSELECTION FROZEN: '
          + (f'{chosen["method"]} alpha {chosen.get("alpha")} score {chosen["score"]:.5f} '
             f'({chosen["gain"]:+.5f})' if chosen else 'nothing passes')
          + f'\n{"=" * 100}')

    # -- I4 ------------------------------------------------------------------
    print(f'\nI4  DISTANCE TO EXTERNAL REFERENCE SHARES -- report only, not used above')
    print('  These describe the public split (~1,650 customers, sampling error about '
          '+/-0.011 at p=0.28),')
    print('  so a projection inside ~0.01 is indistinguishable from exact.')
    ref = {labels.index(k): v for k, v in REFERENCE.items()}
    avail_te = M.trainable_for(cutoffs, M.TEST_CUTOFF)
    print(f'    {"method":<10}' + ''.join(f'{labels[j].split(": ")[-1][:9]:>11}' for j in ref)
          + f'{"sum|err|":>11}')
    print(f'    {"REFERENCE":<10}' + ''.join(f'{ref[j]:>11.3f}' for j in ref) + f'{"-":>11}')
    for m in METHODS:
        t = project(shares, avail_te, M.TEST_CUTOFF, labels, m)
        err = float(sum(abs(t[j] - ref[j]) for j in ref))
        print(f'    {m:<10}' + ''.join(f'{t[j]:>11.3f}' for j in ref) + f'{err:>11.3f}')

    Path('preds/project_mix.json').write_text(json.dumps(dict(
        incumbent=dict(f1=inc_f1, score=inc_score), results=results, isotonic=iso_rows,
        cross_fold_alpha={m: {k: list(v) for k, v in p.items()}
                          for m, p in best_alpha.items()},
        chosen=chosen), indent=2))
    if chosen is None or 'alpha' not in chosen:
        print('\nnothing deployable passes -> not writing a new submission')
        return

    m, al = chosen['method'], chosen['alpha']
    test_ids = pd.read_csv('data/test.csv', dtype=str).ID
    bag = pd.read_csv(a.bag20, dtype={'ID': str}, float_precision='round_trip').set_index('ID')
    pte = pd.read_parquet('preds/oof_bag20_test.parquet').set_index('ID')
    pte = pte.reindex(test_ids)[labels].to_numpy()
    tgt = project(shares, avail_te, M.TEST_CUTOFF, labels, m)
    w, err = C.ipf(pte, tgt, labels, big)
    print(f'\ntest-time rule: project the mix with "{m}" from {avail_te[-1]}, IPF to it '
          f'(mix-err {err:.3f}), alpha {al}')
    sub = bag[['CLV_fuel', 'CLV_nonfuel']].reindex(test_ids).copy()
    sub['Opportunity'] = C.ap(pte, w ** al, labels)
    assert len(sub) == 5488 and sub.index.equals(pd.Index(test_ids))
    assert sub.notna().all().all() and not set(sub.Opportunity) - set(labels)
    assert (sub[['CLV_fuel', 'CLV_nonfuel']] >= 0).all().all()
    # (checked against a fresh read of both files after writing, below -- comparing
    # `sub` to the frame it was copied from here would be vacuous)
    assert list(sub.columns) == ['CLV_fuel', 'CLV_nonfuel', 'Opportunity']
    sub.rename_axis('ID').reset_index().to_csv(a.out, index=False, float_format='%.17g')
    _verify_clv_matches(a.out, a.bag20)
    print(f'wrote {a.out}: 5488 rows, IDs match, labels in config, CLV >= 0 and '
          f'identical to bag20')
    print('  predicted mix: ' + C.fmt_mix(C.mix(sub.Opportunity.to_numpy(), labels), labels))


if __name__ == '__main__':
    main()
