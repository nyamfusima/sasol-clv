"""Sweep 8 model tracks: renewal features for the classifier and the hurdle.

Imported by sweep8.py, which owns the sanity gate and the feature cache. Kept in
its own file only so the gate can be read without scrolling past the fitting
code.

Every classifier number here is reported at BOTH alpha 0.5 and alpha 0.75,
because the two selected picks differ in exactly that parameter and a feature
change has to be judged against both.
"""
import json, time
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.metrics import f1_score
import make_submission as M
import sweep2 as W
import sweep5 as S5
import validate as V

ALPHAS = (0.5, 0.75)
NL = chr(10)
# 5-seed base-feature hurdle RMSEs, measured in sweep 7 and cached in
# preds/sweep7_reg_screen.npz. Held fixed so a classifier-only change moves the
# combined score through F1 alone.
BASE_RF, BASE_RN = [0.5975, 0.5994], [0.7352, 0.7450]


def f1_at_alphas(ctx, proba, cutoff, truth, labels):
    """Weighted F1 under argmax and under prior matching at each alpha. The IPF
    weights are recomputed on the probabilities being scored -- they are a
    property of the matrix, so carrying another model's weights would measure
    this model against a rule calibrated for a different one."""
    P = proba[labels].to_numpy() if hasattr(proba, 'columns') else proba
    tgt, _ = M.target_mix(ctx.snaps, ctx.monthly, cutoff, labels)
    w, err = M.ipf(P, tgt, labels, [labels.index(l) for l in S5.BIG4])
    out = {'argmax': f1_score(truth, np.array(labels)[P.argmax(axis=1)],
                              average='weighted', zero_division=0)}
    for al in ALPHAS:
        pred = np.array(labels)[(P * w ** al).argmax(axis=1)]
        out[al] = f1_score(truth, pred, average='weighted', zero_division=0)
    return out, err


def clf_f1s(ctx, extra, seeds, tag=None):
    """Fit the 17-class classifier on each fold and return F1 at both alphas.
    Caches fold and test-cutoff probabilities whenever `tag` is given, per the
    standing rule that no 20-seed classifier is refit for a submission."""
    labels = ctx.labels
    res = {'argmax': [], 0.5: [], 0.75: []}
    for v in V.FOLDS:
        t0 = time.time()
        # A cached fold matrix is reused rather than refitted. `tag` carries the
        # block name and the seed count, so a hit is the same model spec; this
        # is the same standing rule that caches the test cutoff.
        cached = Path(f'preds/proba_{tag}_{v}.parquet') if tag else None
        if cached is not None and cached.exists():
            pr = pd.read_parquet(cached).set_index('ID')
            yva = ctx.snaps[v][1]
            pr = pr.reindex(yva.index)
            got, err = f1_at_alphas(ctx, pr, v, yva.Opportunity.to_numpy(), labels)
            for k in res:
                res[k].append(got[k])
            print(f'    fold {v}: [reused {cached.name}] argmax '
                  f'{got["argmax"]:.4f}, a0.5 {got[0.5]:.4f}, '
                  f'a0.75 {got[0.75]:.4f}, mix-err {err:.3f}', flush=True)
            continue
        D = ctx.fold(v, extra=extra)
        pr = W.bag_clf(D['Xtr'], D['ycode'], D['Xva'], labels, seeds=seeds)
        got, err = f1_at_alphas(ctx, pr, v, D['yva'].Opportunity.to_numpy(), labels)
        for k in res:
            res[k].append(got[k])
        print(f'    fold {v}: {D["Xtr"].shape[1]} cols, {len(D["Xtr"])} rows, '
              f'argmax {got["argmax"]:.4f}, a0.5 {got[0.5]:.4f}, '
              f'a0.75 {got[0.75]:.4f}, mix-err {err:.3f} '
              f'({time.time() - t0:.0f}s)', flush=True)
        if tag:
            pr.rename_axis('ID').reset_index().to_parquet(
                f'preds/proba_{tag}_{v}.parquet', index=False)
    if tag:
        S5.cache_test_proba(ctx, extra, tag, seeds=seeds)
    return res


def judge_clf(res, ref):
    """Standing keep rule on a classifier-only change. The regressions are held
    at the same base hurdle, so touches=('f1',) and the component slack applies
    to F1 alone."""
    verdicts = {}
    for al in ALPHAS:
        f1s, rf1s = res[al], ref[al]
        per = [M.combined(f1s[i], BASE_RF[i], BASE_RN[i]) for i in (0, 1)]
        rp = [M.combined(rf1s[i], BASE_RF[i], BASE_RN[i]) for i in (0, 1)]
        gain = float(np.mean(per) - np.mean(rp))
        both = all(per[i] > rp[i] for i in (0, 1))
        worst = max(rf1s[i] - f1s[i] for i in (0, 1))
        ok = bool(gain >= W.SCORE_BAR and both and worst <= W.COMPONENT_SLACK)
        why = []
        if gain < W.SCORE_BAR:
            why.append(f'gain {gain:+.5f} < {W.SCORE_BAR}')
        if not both:
            why.append('not better on both folds')
        if worst > W.COMPONENT_SLACK:
            why.append(f'F1 worse by {worst:.4f}')
        print(f'    alpha {al}: F1 {f1s[0]:.4f} / {f1s[1]:.4f} '
              f'(mean {np.mean(f1s):.4f}, ref {np.mean(rf1s):.4f}), score '
              f'{np.mean(per):.5f} ({gain:+.5f})  '
              f'{"KEPT" if ok else "dropped"}  {"; ".join(why)}')
        verdicts[al] = dict(f1=f1s, score=float(np.mean(per)), gain=gain, kept=ok)
    return verdicts


def load_results(path):
    return json.loads(Path(path).read_text()) if Path(path).exists() else {}


def clf_block(ctx, args, blocks, name, renewal_extra, join_extra, save, results):
    seeds = tuple(S5.CONFIRM if args.confirm else S5.SCREEN)
    label = 'confirm' if args.confirm else 'screen'
    cuts = sorted(set(ctx.monthly + list(V.FOLDS) + [W.TEST_CUTOFF]))
    lags = S5.lag_extra(ctx, cuts)
    print(NL + f'--- {name}: lags + {"+".join(blocks) if blocks else "nothing"}, '
          f'{len(seeds)} seeds ---')

    refkey = f'ref_{label}'
    allr = load_results(results)
    if refkey in allr:
        ref = {(k if k == 'argmax' else float(k)): v
               for k, v in allr[refkey].items()}
        print(f'  [lags-only reference from cache: a0.5 mean '
              f'{np.mean(ref[0.5]):.4f}, a0.75 mean {np.mean(ref[0.75]):.4f}]')
    else:
        print('  reference: lags only, no renewal features')
        ref = clf_f1s(ctx, lags, seeds)
        save(refkey, ref)
    if not blocks:
        return ref

    ext = join_extra(lags, renewal_extra(ctx, cuts, blocks))
    ncols = ext[V.FOLDS[0]].shape[1] - lags[V.FOLDS[0]].shape[1]
    print(f'  +{ncols} renewal columns from {"+".join(blocks)}')
    res = clf_f1s(ctx, ext, seeds,
                  tag=f'{name}{len(seeds)}' if args.confirm else None)
    print(NL + '  against the lags-only reference at the same seed count:')
    v = judge_clf(res, ref)
    save(f'{name}_{label}', dict(res={str(k): x for k, x in res.items()},
                                 ref={str(k): x for k, x in ref.items()},
                                 verdicts={str(k): x for k, x in v.items()},
                                 ncols=int(ncols), blocks=list(blocks)))
    return res


def reg_block(ctx, args, renewal_extra, save, results):
    """R1 for the hurdle regressions on top of base features. Projected litres
    = projected fills x median litres per fill is already in the block."""
    seeds = tuple(S5.CONFIRM if args.confirm else S5.SCREEN)
    label = 'confirm' if args.confirm else 'screen'
    cuts = sorted(set(ctx.monthly + list(V.FOLDS) + [W.TEST_CUTOFF]))
    ext = renewal_extra(ctx, cuts, ['r1'])
    W.SEEDS = list(seeds)
    # The reference has to match the seed count. The 5-seed base hurdle is from
    # sweep 7's cache; the 20-seed base is the regression pair the selected
    # picks actually ship, measured in sweep 5.
    ref_rf, ref_rn = (S5.V3_RF, S5.V3_RN) if args.confirm else (BASE_RF, BASE_RN)
    print(NL + f'--- reg: base-feature hurdle + r1, {len(seeds)} seeds ---')
    t0 = time.time()
    # seeds passed explicitly: the module attribute alone used to be ignored
    # by the frozen default argument (fixed in sweep2, see its guard comment)
    rf, rn = W.reg_rmses(ctx, 'hurdle', mode='monthly', extra=ext, seeds=list(seeds))
    print(f'  fitted in {time.time() - t0:.0f}s')
    print(f'  rmse_fuel    {rf[0]:.4f} / {rf[1]:.4f}  against base '
          f'{ref_rf[0]:.4f} / {ref_rf[1]:.4f}')
    print(f'  rmse_nonfuel {rn[0]:.4f} / {rn[1]:.4f}  against base '
          f'{ref_rn[0]:.4f} / {ref_rn[1]:.4f}')
    f1 = S5.V4_F1
    per = [M.combined(f1[i], rf[i], rn[i]) for i in (0, 1)]
    rp = [M.combined(f1[i], ref_rf[i], ref_rn[i]) for i in (0, 1)]
    gain = float(np.mean(per) - np.mean(rp))
    both = all(per[i] > rp[i] for i in (0, 1))
    worst = max(max(rf[i] - ref_rf[i], rn[i] - ref_rn[i]) for i in (0, 1))
    ok = bool(gain >= W.SCORE_BAR and both and worst <= W.COMPONENT_SLACK)
    why = []
    if gain < W.SCORE_BAR:
        why.append(f'gain {gain:+.5f} < {W.SCORE_BAR}')
    if not both:
        why.append('not better on both folds')
    if worst > W.COMPONENT_SLACK:
        why.append(f'an RMSE worse by {worst:.4f}')
    print(f'  score {np.mean(per):.5f} ({gain:+.5f} vs {np.mean(rp):.5f})  '
          f'{"KEPT" if ok else "dropped"}  {"; ".join(why)}')
    save(f'reg_{label}', dict(rf=rf, rn=rn, ref_rf=list(ref_rf),
                              ref_rn=list(ref_rn), gain=gain, kept=ok))
    return rf, rn
