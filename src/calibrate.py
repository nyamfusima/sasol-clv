"""Block H: decision calibration on top of the bag20 probabilities.

Nothing here refits a model. Every variant is a per-class multiplicative weight
applied to saved class probabilities before the argmax, so the whole block runs
in seconds and the only question is whether a decision rule generalises.

Current best is Fuel growth x1.25 (validation score 0.28667), which is the base
every variant is measured against.

  H1  held-out curve over the Fuel weight, nothing fitted
  H2  prior matching by IPF to a past snapshot's true mix, no fitted parameters
  H3  partial matching, H2 weights raised to a power, alpha chosen cross-fold
  H4  one-at-a-time weights on top of Fuel x1.25, kept only on agreement

    python src/calibrate.py
"""
import argparse, json
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.metrics import f1_score
import make_submission as M
import snapshots as S
import validate as V

FUEL = 'Existing-category growth: Fuel'
BIG = ['Stable', 'Inactivity', FUEL, 'Existing-category growth: Other']
H4_CLASSES = ['Stable', 'Inactivity', 'Existing-category growth: Other']
GRID = (0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.25, 1.4, 1.6, 2.0)
BASE_FUEL = 1.25


def f1w(y, pred):
    return f1_score(y, pred, average='weighted', zero_division=0)


def ap(proba, w, labels):
    return np.array(labels)[(proba * w).argmax(axis=1)]


def mix(pred, labels):
    s = pd.Series(pred).value_counts(normalize=True)
    return np.array([s.get(l, 0.0) for l in labels])


def ipf(proba, target, labels, match_idx, iters=400, damp=0.5):
    """Iterative proportional fitting on the argmax counts: nudge each matched
    class's weight by (target share / predicted share) until the predicted mix
    matches the target. argmax is piecewise constant so exact matching is not
    always reachable; the best-error iterate is returned."""
    n = len(proba)
    w = np.ones(len(labels))
    best, best_w = np.inf, w.copy()
    for _ in range(iters):
        cnt = np.bincount((proba * w).argmax(axis=1), minlength=len(labels)) / n
        err = float(np.abs(cnt[match_idx] - target[match_idx]).sum())
        if err < best:
            best, best_w = err, w.copy()
        upd = np.ones(len(labels))
        for j in match_idx:
            t, c = target[j], cnt[j]
            if t <= 0:
                upd[j] = 0.7 if c > 0 else 1.0
            elif c <= 0:
                upd[j] = 1.5
            else:
                upd[j] = (t / c) ** damp
        w = np.clip(w * upd, 1e-8, 1e8)
        w = w / np.exp(np.log(w).mean())          # argmax is scale-invariant
    return best_w, best


def target_mix(snaps, cutoffs, for_cutoff, labels, n_recent=1):
    """True class mix of the most recent n snapshots whose 3-month outcome
    window is fully observed before `for_cutoff`."""
    avail = M.trainable_for(cutoffs, for_cutoff)[-n_recent:]
    shares = []
    for c in avail:
        s = snaps[c][1].Opportunity.value_counts(normalize=True)
        shares.append(np.array([s.get(l, 0.0) for l in labels]))
    return np.mean(shares, axis=0), avail


def fmt_mix(v, labels):
    return ' '.join(f'{l.split(": ")[-1][:9]} {v[labels.index(l)]:.3f}' for l in BIG)


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
    ap_ = argparse.ArgumentParser()
    ap_.add_argument('--config', default='src/label_config.json')
    ap_.add_argument('--bag20', default='submissions/submission_v2_bag20.csv')
    ap_.add_argument('--out', default='submissions/submission_v3.csv')
    a = ap_.parse_args()
    cfg = json.loads(Path(a.config).read_text())
    labels = cfg['opportunity_labels']
    iF = labels.index(FUEL)
    cutoffs = M.monthly()
    snaps = S.load(cutoffs, 'data/train.csv', cfg, ('base',), verbose=False)

    P, Y = {}, {}
    for v in V.FOLDS:
        p = pd.read_parquet(f'preds/oof_bag20_{v}.parquet').set_index('ID')
        y = snaps[v][1]
        P[v] = p.reindex(y.index)[labels].to_numpy()
        Y[v] = y.Opportunity.to_numpy()
    Pte = pd.read_parquet('preds/oof_bag20_test.parquet').set_index('ID')

    base_w = np.ones(len(labels)); base_w[iF] = BASE_FUEL
    base = {v: f1w(Y[v], ap(P[v], base_w, labels)) for v in V.FOLDS}
    base_mean = float(np.mean(list(base.values())))
    b = json.loads(Path('preds/stability.json').read_text())['bag20']
    rmses = {V.FOLDS[0]: (b['fold1'][1], b['fold1'][2]),
             V.FOLDS[1]: (b['fold2'][1], b['fold2'][2])}
    sc = lambda f1s: float(np.mean([M.combined(f1s[i], *rmses[v])
                                    for i, v in enumerate(V.FOLDS)]))
    base_score = sc([base[v] for v in V.FOLDS])
    print(f'base = bag20 + Fuel x{BASE_FUEL}: F1 ' +
          ' | '.join(f'{v} {base[v]:.4f}' for v in V.FOLDS) +
          f' | mean {base_mean:.4f} | score {base_score:.5f}')
    true_mix = {v: mix(Y[v], labels) for v in V.FOLDS}
    for v in V.FOLDS:
        print(f'  true mix {v}: {fmt_mix(true_mix[v], labels)}')

    results = []

    def add(name, f1s, w=None, note=''):
        m = float(np.mean(f1s))
        s = sc(f1s)
        both = all(f1s[i] > base[v] for i, v in enumerate(V.FOLDS))
        gain = s - base_score
        keep = 'KEPT' if (gain >= 0.0015 and both) else 'dropped'
        why = [] if keep == 'KEPT' else (
            ([f'gain {gain:+.5f}<0.0015'] if gain < 0.0015 else []) +
            ([] if both else ['one fold down']))
        results.append(dict(name=name, f1=f1s, mean=m, score=s, gain=gain,
                            keep=keep, note=note or ' '.join(why),
                            w=(None if w is None else list(w))))
        print(f'{name:<40}{f1s[0]:>9.4f}{f1s[1]:>9.4f}{m:>9.4f}'
              f'{m - base_mean:>+9.4f}{s:>10.5f}  {keep} {note or " ".join(why)}')

    # -- H1 ------------------------------------------------------------------
    print(f'\n{"=" * 104}\nH1  HELD-OUT CURVE over the Fuel-growth weight '
          f'(nothing fitted, just evaluated)\n{"=" * 104}')
    print(f'{"variant":<40}{"fold1":>9}{"fold2":>9}{"mean":>9}{"vs base":>9}{"score":>10}')
    curve = []
    for m_ in np.arange(1.0, 2.001, 0.125):
        w = np.ones(len(labels)); w[iF] = m_
        f1s = [f1w(Y[v], ap(P[v], w, labels)) for v in V.FOLDS]
        curve.append((round(float(m_), 3), f1s[0], f1s[1], float(np.mean(f1s))))
        add(f'h1 fuel x{m_:.3f}', f1s, w)
    pk = lambda i: max(curve, key=lambda r: r[i])[0]
    print(f'  peak on fold 1: x{pk(1)} | peak on fold 2: x{pk(2)} | '
          f'peak on mean: x{pk(3)}')
    f1_at = {r[0]: r for r in curve}
    print(f'  flatness: mean F1 within 0.001 of the peak for weights '
          f'{[k for k, r in f1_at.items() if max(c[3] for c in curve) - r[3] <= 0.001]}')

    # -- H2 ------------------------------------------------------------------
    print(f'\n{"=" * 104}\nH2  PRIOR MATCHING by IPF to a past snapshot\'s true '
          f'mix (no fitted parameters)\n{"=" * 104}')
    print(f'{"variant":<40}{"fold1":>9}{"fold2":>9}{"mean":>9}{"vs base":>9}{"score":>10}')
    h2_w = {}
    specs = [('all 17', list(range(len(labels))), 1),
             ('4 large only', [labels.index(l) for l in BIG], 1),
             ('all 17, mean of last 3', list(range(len(labels))), 3)]
    for tag, idx, nrec in specs:
        f1s, ws, notes = [], {}, []
        for v in V.FOLDS:
            tgt, avail = target_mix(snaps, cutoffs, v, labels, nrec)
            w, err = ipf(P[v], tgt, labels, idx)
            ws[v] = w
            f1s.append(f1w(Y[v], ap(P[v], w, labels)))
            got = mix(ap(P[v], w, labels), labels)
            notes.append(f'{v[:7]} target from {",".join(c[:7] for c in avail)} '
                         f'mix-err {err:.3f}')
            print(f'    {v}: target {fmt_mix(tgt, labels)}')
            print(f'           got    {fmt_mix(got, labels)}')
            print(f'           true   {fmt_mix(true_mix[v], labels)}  '
                  f'(target-vs-true drift '
                  f'{np.abs(tgt[[labels.index(l) for l in BIG]] - true_mix[v][[labels.index(l) for l in BIG]]).sum():.3f})')
        h2_w[tag] = ws
        add(f'h2 {tag}', f1s, note='; '.join(notes))

    # -- H3 ------------------------------------------------------------------
    print(f'\n{"=" * 104}\nH3  PARTIAL MATCHING, H2 weights ** alpha, alpha chosen '
          f'on one fold and scored on the other\n{"=" * 104}')
    print(f'{"variant":<40}{"fold1":>9}{"fold2":>9}{"mean":>9}{"vs base":>9}{"score":>10}')
    h3_alpha = {}
    for tag in h2_w:
        for al in (0.25, 0.5, 0.75, 1.0):
            f1s = [f1w(Y[v], ap(P[v], h2_w[tag][v] ** al, labels)) for v in V.FOLDS]
            add(f'h3 {tag} alpha {al}', f1s)
        # cross-fold alpha selection, both directions
        held = {}
        for fit_on in V.FOLDS:
            other = [v for v in V.FOLDS if v != fit_on][0]
            best_al = max((0.25, 0.5, 0.75, 1.0),
                          key=lambda al: f1w(Y[fit_on], ap(P[fit_on],
                                                           h2_w[tag][fit_on] ** al, labels)))
            held[other] = (best_al, f1w(Y[other], ap(P[other], h2_w[tag][other] ** best_al,
                                                     labels)))
        # `held` is keyed by the HELD-OUT fold; alpha was fitted on the other one
        print(f'  {tag}: cross-fold alpha -> ' + ' | '.join(
            f'alpha {v[0]} fitted on the other fold, held-out F1 on {k} = {v[1]:.4f}'
            for k, v in held.items()))
        h3_alpha[tag] = held

    # -- H4 ------------------------------------------------------------------
    print(f'\n{"=" * 104}\nH4  ONE-AT-A-TIME weights on top of Fuel x{BASE_FUEL}, kept '
          f'only on direction agreement AND both held-out folds up\n{"=" * 104}')
    applied = base_w.copy()
    h4_keep = []
    for cls in H4_CLASSES:
        j = labels.index(cls)
        fits = {}
        for v in V.FOLDS:
            best = max(GRID, key=lambda g: f1w(
                Y[v], ap(P[v], np.where(np.arange(len(labels)) == j, g, base_w), labels)))
            fits[v] = best
        a_, b_ = fits[V.FOLDS[0]], fits[V.FOLDS[1]]
        same_dir = (a_ > 1 and b_ > 1) or (a_ < 1 and b_ < 1) or (a_ == b_ == 1.0)
        # held out: each fold's fitted value scored on the other fold
        held = {}
        for fit_on in V.FOLDS:
            other = [v for v in V.FOLDS if v != fit_on][0]
            w = base_w.copy(); w[j] = fits[fit_on]
            held[other] = f1w(Y[other], ap(P[other], w, labels))
        both_up = all(held[v] > base[v] for v in V.FOLDS)
        ok = same_dir and both_up
        conservative = min((a_, b_), key=lambda x: abs(x - 1.0))
        print(f'  {cls:<34} fits {a_:.2f} / {b_:.2f} | same direction {same_dir} | '
              f'held-out ' + ' '.join(f'{v[:7]} {held[v]:+.4f}' for v in V.FOLDS).replace(
                  '+0', '+0') + f' | both up {both_up} -> {"KEEP" if ok else "drop"}')
        if ok:
            applied[j] = conservative
            h4_keep.append((cls, conservative))
    if h4_keep:
        f1s = [f1w(Y[v], ap(P[v], applied, labels)) for v in V.FOLDS]
        add('h4 ' + '+'.join(f'{c.split(": ")[-1]} x{m}' for c, m in h4_keep), f1s, applied,
            note='conservative value of the two fits')
    else:
        print('  nothing kept in H4')

    # -- decide --------------------------------------------------------------
    kept = [r for r in results if r['keep'] == 'KEPT']
    print(f'\n{"=" * 104}\nSUMMARY: {len(kept)} of {len(results)} variants pass the '
          f'standing rule against bag20 + Fuel x{BASE_FUEL} (score {base_score:.5f})')
    Path('preds/calibrate.json').write_text(json.dumps(
        dict(base=dict(f1=[base[v] for v in V.FOLDS], score=base_score),
             curve=curve, results=results), indent=2))
    if not kept:
        print('nothing passes -> not writing a new submission')
        return
    best = max(kept, key=lambda r: r['score'])
    print(f'best: {best["name"]} score {best["score"]:.5f} ({best["gain"]:+.5f})')
    test_ids = pd.read_csv('data/test.csv', dtype=str).ID
    if best['w'] is None:
        # An H2/H3 variant. Its weights are per-dataset by construction, but the
        # RULE is fixed and needs no labels from the thing being predicted: match
        # the predicted mix to a past snapshot's observed mix, then damp by alpha.
        # Rebuild it on the test probabilities.
        tag = best['name'].split(' alpha ')[0].replace('h3 ', '')
        al = float(best['name'].split(' alpha ')[1])
        idx, nrec = {'all 17': (list(range(len(labels))), 1),
                     '4 large only': ([labels.index(l) for l in BIG], 1),
                     'all 17, mean of last 3': (list(range(len(labels))), 3)}[tag]
        tgt, avail = target_mix(snaps, cutoffs, M.TEST_CUTOFF, labels, nrec)
        pte0 = Pte.reindex(test_ids)[labels].to_numpy()
        w_raw, err = ipf(pte0, tgt, labels, idx)
        w = w_raw ** al
        print(f'  test-time rule: IPF to the {",".join(avail)} mix '
              f'(mix-err {err:.3f}), then alpha {al}')
        print(f'    target {fmt_mix(tgt, labels)}')
        print(f'    full-match got {fmt_mix(mix(ap(pte0, w_raw, labels), labels), labels)}')
    else:
        w = np.array(best['w'])
    bag = pd.read_csv(a.bag20, dtype={'ID': str}, float_precision='round_trip').set_index('ID')
    pte = Pte.reindex(test_ids)[labels].to_numpy()
    sub = bag[['CLV_fuel', 'CLV_nonfuel']].reindex(test_ids).copy()
    sub['Opportunity'] = ap(pte, w, labels)
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
    print('  predicted mix: ' + fmt_mix(mix(sub.Opportunity.to_numpy(), labels), labels))


if __name__ == '__main__':
    main()
