"""Sweep 5: maximise. Tracks R (regressions), L (label model), D (decision rule).

Conventions that make the tracks comparable:
  * Regression variants are measured on top of the hurdle and hold F1 at v3's
    decision-rule values, so touches=('rf','rn').
  * Label variants are measured as probabilities pushed through the SAME v3
    decision rule (partial prior matching, 4 large classes, alpha 0.5), which
    separates model effects from rule effects.
  * Screening runs at 5 seeds (42-46); anything promising is re-measured at 20
    (42-61). Seed sd is 0.00020 against a 0.0015 bar, so screening cannot hide a
    passing effect.

    python src/sweep5.py --track r1
"""
import argparse, json, time
from pathlib import Path
import numpy as np, pandas as pd
import calibrate as C
import make_submission as M
import snapshots as S
import sweep2 as W
import train_label as T
import validate as V

RESULTS = Path('preds/sweep5.json')
SCREEN = tuple(range(42, 47))
CONFIRM = tuple(range(42, 62))
# v3 = hurdle regressions (bag20) + partial prior matching at alpha 0.5
V3_F1 = [0.5022, 0.5173]
V3_RF = [0.5969, 0.5996]
V3_RN = [0.7350, 0.7451]
# the 5-seed hurdle baseline and the pieces already measured in sweep 2
FIVE_SEED = {
    'hurdle (sweep2 b1)':            dict(rf=[0.5975, 0.5994], rn=[0.7352, 0.7450]),
    '+ f3f4 features (freg)':        dict(rf=[0.5957, 0.6005], rn=[0.7336, 0.7448]),
    '+ catboost magnitude (b3)':     dict(rf=[0.5960, 0.5989], rn=[0.7339, 0.7444]),
    '+ lgbm/cat average (b3)':       dict(rf=[0.5965, 0.5989], rn=[0.7344, 0.7446]),
}


def score(f1s, rf, rn):
    return float(np.mean([M.combined(f1s[i], rf[i], rn[i]) for i in (0, 1)]))


def save(key, obj):
    RESULTS.parent.mkdir(exist_ok=True)
    allr = json.loads(RESULTS.read_text()) if RESULTS.exists() else {}
    allr[key] = obj
    RESULTS.write_text(json.dumps(allr, indent=2))


def judge_reg(rf, rn, base_rf, base_rn, f1s=None):
    """Standing rule on a regression-only change: +0.0015 mean score, both folds
    up, and neither touched component worse on either fold by more than 0.001."""
    f1s = f1s or V3_F1
    s, b = score(f1s, rf, rn), score(f1s, base_rf, base_rn)
    gain = s - b
    both = all(M.combined(f1s[i], rf[i], rn[i]) > M.combined(f1s[i], base_rf[i], base_rn[i])
               for i in (0, 1))
    worst = max(max(rf[i] - base_rf[i], rn[i] - base_rn[i]) for i in (0, 1))
    ok = gain >= 0.0015 and both and worst <= 0.001
    why = []
    if gain < 0.0015:
        why.append(f'gain {gain:+.5f}<0.0015')
    if not both:
        why.append('one fold down')
    if worst > 0.001:
        why.append(f'component -{worst:.4f}')
    return dict(score=s, gain=gain, keep='KEPT' if ok else 'dropped', note=' '.join(why))


def row(name, rf, rn, base_rf, base_rn, f1s=None):
    j = judge_reg(rf, rn, base_rf, base_rn, f1s)
    print(f'{name:<38}{rf[0]:>8.4f}{rf[1]:>8.4f}{rn[0]:>8.4f}{rn[1]:>8.4f}'
          f'{j["score"]:>10.5f}{j["gain"]:>+10.5f}  {j["keep"]} {j["note"]}')
    return dict(name=name, rf=rf, rn=rn, **j)


def header():
    print(f'{"variant":<38}{"rf f1":>8}{"rf f2":>8}{"rn f1":>8}{"rn f2":>8}'
          f'{"score":>10}{"gain":>10}  verdict')


# --- R1 ----------------------------------------------------------------------

def track_r1(ctx, args):
    """Bundle the individually sub-bar, same-direction regression changes:
    f3+f4 features for the regressors and a LightGBM/CatBoost magnitude
    average, at monthly spacing. The pieces were measured separately in sweep 2
    at 5 seeds; only the bundle is new."""
    cuts = sorted(set(ctx.monthly + list(V.FOLDS) + [W.TEST_CUTOFF]))
    extra = W.f_snapshots(ctx, ('f3', 'f4'), cuts)
    base = FIVE_SEED['hurdle (sweep2 b1)']

    print('\n--- 5-seed screen: pieces (from sweep 2) and the new bundles ---')
    header()
    rows = [row(n + ' [5s]', v['rf'], v['rn'], base['rf'], base['rn'])
            for n, v in FIVE_SEED.items()]
    bundles = {}
    for name, kind, ex in (('bundle f3f4 + lgbm/cat avg', 'hurdle_avg', extra),
                           ('bundle f3f4 + catboost mag', 'hurdle_cat', extra)):
        t0 = time.time()
        W.SEEDS = list(SCREEN)
        rf, rn = W.reg_rmses(ctx, kind, mode='monthly', extra=ex)
        bundles[name] = (rf, rn)
        rows.append(row(name + ' [5s]', rf, rn, base['rf'], base['rn']))
        print(f'    ({time.time() - t0:.0f}s)')

    # marginal effect of each addition, on the combined score
    print('\n--- marginal effect of each addition (5-seed, score) ---')
    s_base = score(V3_F1, base['rf'], base['rn'])
    for n, v in FIVE_SEED.items():
        if n == 'hurdle (sweep2 b1)':
            continue
        print(f'  {n:<34} alone {score(V3_F1, v["rf"], v["rn"]) - s_base:+.5f}')
    for n, (rf, rn) in bundles.items():
        print(f'  {n:<34} bundle {score(V3_F1, rf, rn) - s_base:+.5f}')
    f34 = FIVE_SEED['+ f3f4 features (freg)']
    for n, (rf, rn) in bundles.items():
        piece = ('+ lgbm/cat average (b3)' if 'avg' in n else '+ catboost magnitude (b3)')
        p = FIVE_SEED[piece]
        print(f'  {n:<34} marginal of f3f4 given {piece.split("(")[0].strip():<24}'
              f'{score(V3_F1, rf, rn) - score(V3_F1, p["rf"], p["rn"]):+.5f}')
        print(f'  {n:<34} marginal of {piece.split("(")[0].strip():<34}'
              f'{score(V3_F1, rf, rn) - score(V3_F1, f34["rf"], f34["rn"]):+.5f}')

    # confirm the best screened bundle at 20 seeds against bag20's hurdle
    best = max(bundles, key=lambda k: score(V3_F1, *bundles[k]))
    print(f'\n--- 20-seed confirmation of "{best}" against v3 regressions ---')
    kind = 'hurdle_avg' if 'avg' in best else 'hurdle_cat'
    t0 = time.time()
    W.SEEDS = list(CONFIRM)
    rf, rn = W.reg_rmses(ctx, kind, mode='monthly', extra=extra)
    header()
    rows.append(row('v3 regressions [20s, incumbent]', V3_RF, V3_RN, V3_RF, V3_RN))
    conf = row(best + ' [20s]', rf, rn, V3_RF, V3_RN)
    rows.append(conf)
    print(f'    ({time.time() - t0:.0f}s)')
    W.SEEDS = list(CONFIRM)
    save('r1', dict(rows=rows, confirmed=conf, bundle=best,
                    bundle_rf=rf, bundle_rn=rn))
    return conf


TRACKS = {'r1': track_r1}


# --- R2 monthly decomposition -------------------------------------------------

# A 1-month outcome window needs only one month of future data, so cutoffs run
# later than the quarterly 2025-09-01 limit: for horizon h, c + h + 1 month must
# land inside the data (which ends 2025-11-30).
R2_LAST_CUTOFF = '2025-11-01'
HORIZONS = (0, 1, 2)


def month_starts(first='2024-06-01', last=R2_LAST_CUTOFF):
    return [str(c.date()) for c in pd.date_range(first, last, freq='MS')]


def monthly_totals(d):
    """Per customer per calendar month: fuel litres and non-fuel rands, on the
    same definitions label_rules uses for the quarterly targets."""
    m = d._time.dt.to_period('M').dt.to_timestamp()
    fu = d[d.is_fuel].groupby([d.ID[d.is_fuel], m[d.is_fuel]]).qty.sum()
    nf = d[d.is_nonfuel].groupby([d.ID[d.is_nonfuel], m[d.is_nonfuel]]).amt.sum()
    fu.index.names = ['ID', 'month']
    nf.index.names = ['ID', 'month']
    return {'CLV_fuel': fu, 'CLV_nonfuel': nf}


def raw_month(tot, tgt, ids, month):
    """Raw total for one calendar month, 0 where the customer bought nothing."""
    s = tot[tgt]
    try:
        v = s.xs(pd.Timestamp(month), level='month')
    except KeyError:
        return pd.Series(0.0, index=ids)
    return v.reindex(ids).fillna(0.0)


def verify_monthly_sum(ctx, tot):
    """Three consecutive monthly totals must reproduce the cached quarterly
    target exactly, otherwise the decomposition is measuring something else."""
    cfg = ctx.cfg
    norm = cfg['normalization']
    worst = {}
    for c in ['2025-03-01', V.FOLDS[1]]:
        X, y = ctx.snaps[c]
        for tgt, col in (('CLV_fuel', 'fuel_litres'), ('CLV_nonfuel', 'nonfuel_rand')):
            tri = sum(raw_month(tot, tgt, X.index,
                                str((pd.Timestamp(c) + pd.DateOffset(months=h)).date()))
                      for h in HORIZONS)
            ref = y[col].reindex(X.index)
            worst[(c, tgt)] = float(np.abs(tri.to_numpy() - ref.to_numpy()).max())
    for k, v in worst.items():
        print(f'    sum of 3 monthly totals vs cached quarterly {k[0]} {k[1]}: '
              f'max abs diff {v:.2e}')
    assert max(worst.values()) < 1e-6, 'monthly decomposition does not reproduce the target'
    print('    monthly decomposition reproduces the quarterly targets exactly')


def r2_training_rows(ctx, tot, val_cutoff, tgt, horizons=HORIZONS):
    """(X, y_raw) over (cutoff, horizon) pairs whose 1-month outcome window ends
    on or before val_cutoff. The horizon is a feature, because at predict time
    all three months must be forecast from history available at the cutoff."""
    v = pd.Timestamp(val_cutoff)
    parts_X, parts_y = [], []
    for h in horizons:
        for c in month_starts():
            end = pd.Timestamp(c) + pd.DateOffset(months=h + 1)
            if end > v:
                continue
            X = T.select(ctx.snaps[c][0], ('base',), ctx.bc).copy()
            X['horizon'] = h
            parts_X.append(X)
            parts_y.append(raw_month(tot, tgt, X.index,
                                     str((pd.Timestamp(c) + pd.DateOffset(months=h)).date())))
    return pd.concat(parts_X), pd.concat(parts_y)


def r2_predict(ctx, tot, val_cutoff, tgt, seeds):
    """Hurdle on raw monthly totals, summed over the three horizons in raw space.
    Returns the RAW quarterly sum, not the transformed value: the transform and
    any shrink are applied by the caller.

    P(y>0) * E[y|y>0] is a conditional MEAN in raw space. The metric is RMSE on
    ln(1+Y)/s, and ln(1+mean) > E[ln(1+actual)] for a skewed Y, so transforming
    a mean prediction directly is biased upward -- the same Jensen gap that
    makes a shrink factor necessary for Tweedie in R3. The caller fits that
    shrink cross-fold; the uncorrected value is reported too, so the size of the
    bias is visible rather than assumed."""
    Xtr, ytr = r2_training_rows(ctx, tot, val_cutoff, tgt)
    Xva = T.select(ctx.snaps[val_cutoff][0], ('base',), ctx.bc)
    pos = (ytr.to_numpy() > 0)
    gate_y = pos.astype(int)
    mag_X = Xtr[pos]
    mag_y = np.log1p(ytr.to_numpy()[pos])
    total = np.zeros(len(Xva))
    for h in HORIZONS:
        Xh = Xva.copy(); Xh['horizon'] = h
        p = M.bag_binary(Xtr, gate_y, Xh, seeds)
        mg = np.expm1(np.clip(M.bag_reg(mag_X, pd.Series(mag_y), Xh, seeds), 0, None))
        total += np.clip(p * mg, 0, None)
    return total, len(Xtr)


def track_r2(ctx, args):
    """Monthly decomposition against the direct quarterly hurdle, plus their
    average. Needs feature snapshots at 2025-10-01 and 2025-11-01, which the
    quarterly pipeline never built because their 3-month outcomes run past the
    data."""
    cuts = month_starts()
    need = [c for c in cuts if c not in ctx.snaps]
    if need:
        print(f'  building feature-only snapshots for {need}')
        ctx.snaps.update(S.load(need, ctx.train, ctx.cfg, ('base',),
                                label_cutoffs=[], verbose=True))
    d = S.load_data(ctx.train, ctx.cfg)
    tot = monthly_totals(d)
    print('  checking the decomposition against the cached quarterly targets:')
    verify_monthly_sum(ctx, tot)

    W.SEEDS = list(SCREEN)
    rmse = lambda z, t: float(np.sqrt(((np.asarray(z) - t) ** 2).mean()))
    raws, cache = {}, Path('preds/r2_raw.json')
    for tgt in M.TARGETS:
        for v in V.FOLDS:
            t0 = time.time()
            raws[(tgt, v)], n = r2_predict(ctx, tot, v, tgt, SCREEN)
            print(f'  {tgt} fold {v}: {n} training rows '
                  f'({time.time() - t0:.0f}s)', flush=True)
    cache.write_text(json.dumps({f'{k[0]}|{k[1]}': list(map(float, v))
                                 for k, v in raws.items()}))

    out, shrunk, ks = {}, {}, []
    for tgt in M.TARGETS:
        s_ = ctx.cfg['normalization'][tgt]
        for v in V.FOLDS:
            truth = ctx.snaps[v][1][tgt].to_numpy()
            z = np.log1p(raws[(tgt, v)]) / s_
            out[(tgt, v)] = (z, rmse(z, truth))
            other = [x for x in V.FOLDS if x != v][0]
            k, _ = fit_shrink(raws[(tgt, other)],
                              ctx.snaps[other][1][tgt].to_numpy(), s_)
            k_own, _ = fit_shrink(raws[(tgt, v)], truth, s_)
            zs = np.log1p(np.clip(k * raws[(tgt, v)], 0, None)) / s_
            shrunk[(tgt, v)] = (zs, rmse(zs, truth))
            ks.append((tgt, v, k, k_own))
            print(f'  {tgt} fold {v}: rmse {out[(tgt, v)][1]:.4f} uncorrected, '
                  f'{shrunk[(tgt, v)][1]:.4f} with shrink k={k:.2f} '
                  f'(own-fold optimum {k_own:.2f})')

    rf = [out[('CLV_fuel', v)][1] for v in V.FOLDS]
    rn = [out[('CLV_nonfuel', v)][1] for v in V.FOLDS]
    srf = [shrunk[('CLV_fuel', v)][1] for v in V.FOLDS]
    srn = [shrunk[('CLV_nonfuel', v)][1] for v in V.FOLDS]
    base = FIVE_SEED['hurdle (sweep2 b1)']
    print('\n--- R2 monthly decomposition (5-seed) ---')
    header()
    rows = [row('quarterly hurdle [5s, base]', base['rf'], base['rn'],
                base['rf'], base['rn']),
            row('monthly, no shrink [5s]', rf, rn, base['rf'], base['rn']),
            row('monthly + cross-fold shrink [5s]', srf, srn, base['rf'], base['rn'])]

    # average of the two, in z space
    W.SEEDS = list(SCREEN)
    qz = {}
    for tgt in M.TARGETS:
        for v in V.FOLDS:
            D = ctx.fold(v, mode='monthly')
            qz[(tgt, v)] = W.hurdle_reg(D['Xtr'], D['ytr'][tgt], D['Xva'])
    for tag, src in (('average(quarterly, monthly) [5s]', out),
                     ('average(quarterly, monthly+shrink) [5s]', shrunk)):
        arf, arn = [], []
        for tgt, acc in (('CLV_fuel', arf), ('CLV_nonfuel', arn)):
            for v in V.FOLDS:
                z = (src[(tgt, v)][0] + qz[(tgt, v)]) / 2
                acc.append(rmse(z, ctx.snaps[v][1][tgt].to_numpy()))
        rows.append(row(tag, arf, arn, base['rf'], base['rn']))
    print('')
    print('  shrink factors, fitted on the other fold:')
    for tgt, v, k, k_own in ks:
        print(f'    {tgt:<12} applied to {v}: k={k:.2f} (own-fold optimum {k_own:.2f})')
    save('r2', dict(rows=rows, monthly=dict(rf=rf, rn=rn),
                    monthly_shrunk=dict(rf=srf, rn=srn),
                    shrink=[list(x) for x in ks]))
    return rows


TRACKS['r2'] = track_r2


# --- R3 loss and space --------------------------------------------------------

RAW_COL = {'CLV_fuel': 'fuel_litres', 'CLV_nonfuel': 'nonfuel_rand'}
SHRINK_GRID = tuple(np.round(np.arange(0.40, 1.51, 0.05), 2))


def bag_reg_obj(Xtr, y, Xva, seeds, **params):
    """Seed-averaged LightGBM regressor with an explicit objective."""
    import lightgbm as lgb
    p = dict(M.REG); p.update(params)
    out = np.zeros(len(Xva))
    for s in seeds:
        out += lgb.LGBMRegressor(random_state=s, verbose=-1, **p).fit(Xtr, y).predict(Xva)
    return out / len(seeds)


def hurdle_obj(Xtr, y, Xva, seeds, **params):
    """The current hurdle with a different stage-2 objective. Stage 1 and the
    clipping are unchanged, so any difference is attributable to the loss."""
    pos = (y.to_numpy() > 0)
    if pos.all() or not pos.any():
        return np.clip(bag_reg_obj(Xtr, y, Xva, seeds, **params), 0, None)
    p = M.bag_binary(Xtr, pos.astype(int), Xva, seeds)
    mag = np.clip(bag_reg_obj(Xtr[pos], y[pos], Xva, seeds, **params), 0, None)
    return np.clip(p * mag, 0, None)


def fit_shrink(pred_raw, truth_z, s):
    """k minimising RMSE of ln(1+k*yhat)/s against the transformed truth.
    E[ln(1+Y)] < ln(1+E[Y]), so a point prediction of the mean overshoots in
    log space and k < 1 is expected."""
    best, bk = np.inf, 1.0
    for k in SHRINK_GRID:
        z = np.log1p(np.clip(k * pred_raw, 0, None)) / s
        e = float(np.sqrt(((z - truth_z) ** 2).mean()))
        if e < best:
            best, bk = e, float(k)
    return bk, best


def track_r3(ctx, args):
    base = FIVE_SEED['hurdle (sweep2 b1)']
    W.SEEDS = list(SCREEN)
    folds = {v: ctx.fold(v, mode='monthly') for v in V.FOLDS}
    truth = {(t, v): ctx.snaps[v][1][t].to_numpy() for t in M.TARGETS for v in V.FOLDS}
    rmse = lambda z, t: float(np.sqrt(((np.asarray(z) - t) ** 2).mean()))

    preds = {}
    # the incumbent stage 2, recomputed here so every comparison is like for like
    for t in M.TARGETS:
        for v in V.FOLDS:
            preds[('current', t, v)] = W.hurdle_reg(folds[v]['Xtr'], folds[v]['ytr'][t],
                                                   folds[v]['Xva'])
    variants = {'huber': dict(objective='huber'),
                'quantile median': dict(objective='quantile', alpha=0.5)}
    for name, kw in variants.items():
        for t in M.TARGETS:
            for v in V.FOLDS:
                t0 = time.time()
                preds[(name, t, v)] = hurdle_obj(folds[v]['Xtr'], folds[v]['ytr'][t],
                                                 folds[v]['Xva'], SCREEN, **kw)
                print(f'  {name} {t} fold {v} ({time.time() - t0:.0f}s)', flush=True)

    # Tweedie on raw totals, no hurdle (Tweedie models the zero mass itself),
    # with the shrink factor fitted on one fold and applied to the other
    shrink_report = []
    for t in M.TARGETS:
        s = ctx.cfg['normalization'][t]
        raw_pred = {}
        for v in V.FOLDS:
            t0 = time.time()
            yraw = folds[v]['ytr'][RAW_COL[t]]
            raw_pred[v] = np.clip(bag_reg_obj(folds[v]['Xtr'], yraw, folds[v]['Xva'],
                                              SCREEN, objective='tweedie',
                                              tweedie_variance_power=1.3), 0, None)
            print(f'  tweedie {t} fold {v} ({time.time() - t0:.0f}s)', flush=True)
        for v in V.FOLDS:
            other = [x for x in V.FOLDS if x != v][0]
            k, _ = fit_shrink(raw_pred[other], truth[(t, other)], s)
            preds[('tweedie', t, v)] = np.log1p(k * raw_pred[v]) / s
            k_own, _ = fit_shrink(raw_pred[v], truth[(t, v)], s)
            shrink_report.append((t, v, k, k_own))

    print('\n  shrink factors (fitted on the OTHER fold, own-fold value shown for '
          'reference):')
    for t, v, k, k_own in shrink_report:
        print(f'    {t:<12} applied to {v}: k={k:.2f} (own-fold optimum {k_own:.2f})')

    print('\n--- R3 loss and space (5-seed) ---')
    header()
    rows = [row('hurdle, current stage 2 [5s]',
                [rmse(preds[('current', 'CLV_fuel', v)], truth[('CLV_fuel', v)]) for v in V.FOLDS],
                [rmse(preds[('current', 'CLV_nonfuel', v)], truth[('CLV_nonfuel', v)]) for v in V.FOLDS],
                base['rf'], base['rn'])]
    for name in ['huber', 'quantile median', 'tweedie']:
        rf = [rmse(preds[(name, 'CLV_fuel', v)], truth[('CLV_fuel', v)]) for v in V.FOLDS]
        rn = [rmse(preds[(name, 'CLV_nonfuel', v)], truth[('CLV_nonfuel', v)]) for v in V.FOLDS]
        rows.append(row(f'{name} [5s]', rf, rn, base['rf'], base['rn']))

    # average each variant with the current stage 2
    print('\n--- averaged with the current stage 2 ---')
    header()
    for name in ['huber', 'quantile median', 'tweedie']:
        rf, rn = [], []
        for t, acc in (('CLV_fuel', rf), ('CLV_nonfuel', rn)):
            for v in V.FOLDS:
                z = (preds[(name, t, v)] + preds[('current', t, v)]) / 2
                acc.append(rmse(z, truth[(t, v)]))
        rows.append(row(f'avg(current, {name}) [5s]', rf, rn, base['rf'], base['rn']))
    save('r3', dict(rows=rows, shrink=[list(x) for x in shrink_report]))
    return rows


TRACKS['r3'] = track_r3


# --- R4 model diversity for the hurdle stages ---------------------------------

BLEND_W = (0.1, 0.2, 0.3, 0.5)


def signed_log(X):
    """sign(x)*log1p(|x|), which keeps the sign of slope and ratio features
    while compressing the heavy-tailed spend columns. NaNs become 0 after
    scaling, which is what a linear model needs."""
    A = X.to_numpy(dtype=float)
    A = np.sign(A) * np.log1p(np.abs(A))
    mu = np.nanmean(A, axis=0)
    sd = np.nanstd(A, axis=0)
    sd[~np.isfinite(sd) | (sd == 0)] = 1.0
    A = (A - mu) / sd
    return np.nan_to_num(A, nan=0.0, posinf=0.0, neginf=0.0)


def _family_hurdle(family, Xtr, y, Xva, seeds):
    """Hurdle with both stages from one model family."""
    from sklearn.ensemble import ExtraTreesRegressor, ExtraTreesClassifier
    from sklearn.linear_model import Ridge, LogisticRegression
    pos = (y.to_numpy() > 0)
    if family == 'xgboost':
        import xgboost as xgb
        g = np.zeros(len(Xva)); m = np.zeros(len(Xva))
        for s in seeds:
            c = xgb.XGBClassifier(n_estimators=400, learning_rate=0.03, max_depth=5,
                                  subsample=0.8, colsample_bytree=0.7, random_state=s,
                                  tree_method='hist', verbosity=0, n_jobs=-1)
            g += c.fit(Xtr, pos.astype(int)).predict_proba(Xva)[:, 1]
            r = xgb.XGBRegressor(n_estimators=500, learning_rate=0.03, max_depth=6,
                                 subsample=0.8, colsample_bytree=0.7, random_state=s,
                                 tree_method='hist', verbosity=0, n_jobs=-1)
            m += r.fit(Xtr[pos], y[pos]).predict(Xva)
        return np.clip((g / len(seeds)) * np.clip(m / len(seeds), 0, None), 0, None)
    if family == 'extratrees':
        g = np.zeros(len(Xva)); m = np.zeros(len(Xva))
        Xt = Xtr.fillna(-999); Xv = Xva.fillna(-999)
        for s in seeds:
            c = ExtraTreesClassifier(n_estimators=300, min_samples_leaf=10,
                                     random_state=s, n_jobs=-1)
            g += c.fit(Xt, pos.astype(int)).predict_proba(Xv)[:, 1]
            r = ExtraTreesRegressor(n_estimators=300, min_samples_leaf=10,
                                    random_state=s, n_jobs=-1)
            m += r.fit(Xt[pos], y[pos]).predict(Xv)
        return np.clip((g / len(seeds)) * np.clip(m / len(seeds), 0, None), 0, None)
    if family == 'ridge':
        # deterministic, so seeds are irrelevant and one fit is enough
        At, Av = signed_log(Xtr), signed_log(Xva)
        c = LogisticRegression(max_iter=2000, C=1.0)
        g = c.fit(At, pos.astype(int)).predict_proba(Av)[:, 1]
        r = Ridge(alpha=10.0).fit(At[pos], y[pos])
        return np.clip(g * np.clip(r.predict(Av), 0, None), 0, None)
    raise ValueError(family)


def track_r4(ctx, args):
    base = FIVE_SEED['hurdle (sweep2 b1)']
    W.SEEDS = list(SCREEN)
    folds = {v: ctx.fold(v, mode='monthly') for v in V.FOLDS}
    truth = {(t, v): ctx.snaps[v][1][t].to_numpy() for t in M.TARGETS for v in V.FOLDS}
    rmse = lambda z, t: float(np.sqrt(((np.asarray(z) - t) ** 2).mean()))
    preds = {}
    for t in M.TARGETS:
        for v in V.FOLDS:
            preds[('lightgbm', t, v)] = W.hurdle_reg(folds[v]['Xtr'], folds[v]['ytr'][t],
                                                     folds[v]['Xva'])
    fams = ['xgboost', 'extratrees', 'ridge']
    for fam in fams:
        for t in M.TARGETS:
            for v in V.FOLDS:
                t0 = time.time()
                try:
                    preds[(fam, t, v)] = _family_hurdle(fam, folds[v]['Xtr'],
                                                        folds[v]['ytr'][t],
                                                        folds[v]['Xva'], SCREEN)
                    print(f'  {fam} {t} fold {v} ({time.time() - t0:.0f}s)', flush=True)
                except Exception as e:
                    print(f'  {fam} {t} fold {v} FAILED: {type(e).__name__}: {e}')
                    preds[(fam, t, v)] = None

    print('\n--- R4 families alone (5-seed) ---')
    header()
    rows = []
    for fam in ['lightgbm'] + fams:
        if any(preds.get((fam, t, v)) is None for t in M.TARGETS for v in V.FOLDS):
            continue
        rf = [rmse(preds[(fam, 'CLV_fuel', v)], truth[('CLV_fuel', v)]) for v in V.FOLDS]
        rn = [rmse(preds[(fam, 'CLV_nonfuel', v)], truth[('CLV_nonfuel', v)]) for v in V.FOLDS]
        rows.append(row(f'{fam} alone [5s]', rf, rn, base['rf'], base['rn']))

    print('\n--- blended with LightGBM, weight fitted on one fold, scored on the '
          'other ---')
    header()
    for fam in fams:
        if any(preds.get((fam, t, v)) is None for t in M.TARGETS for v in V.FOLDS):
            continue
        picks = {}
        for fit_on in V.FOLDS:
            other = [x for x in V.FOLDS if x != fit_on][0]
            best_w, best_e = None, np.inf
            for w in BLEND_W:
                e = float(np.mean([
                    rmse((1 - w) * preds[('lightgbm', t, fit_on)] + w * preds[(fam, t, fit_on)],
                         truth[(t, fit_on)]) for t in M.TARGETS]))
                if e < best_e:
                    best_w, best_e = w, e
            picks[other] = best_w
        agree = len(set(picks.values())) == 1
        rf, rn = [], []
        for t, acc in (('CLV_fuel', rf), ('CLV_nonfuel', rn)):
            for v in V.FOLDS:
                w = picks[v]
                acc.append(rmse((1 - w) * preds[('lightgbm', t, v)] + w * preds[(fam, t, v)],
                                truth[(t, v)]))
        rows.append(row(f'lgbm + {fam} blend [5s]', rf, rn, base['rf'], base['rn']))
        print(f'    cross-fold weights: ' + ' | '.join(f'{k} w={v}' for k, v in picks.items())
              + ('  AGREE' if agree else '  DISAGREE'))
    save('r4', dict(rows=rows))
    return rows


# --- R5 monthly lag series for the regressors ---------------------------------

def lag_extra(ctx, cuts):
    """Cached lag features for the given cutoffs."""
    import features3 as F3
    p = Path('preds/lags.parquet')
    have = {}
    if p.exists():
        t = pd.read_parquet(p)
        for c, g in t.groupby('_cutoff', observed=True):
            have[str(c)] = g.drop(columns=['_cutoff']).set_index('ID')
    need = [c for c in cuts if c not in have]
    if need:
        d = S.load_data(ctx.train, ctx.cfg)
        for c in need:
            t0 = time.time()
            have[c] = F3.build(d, c, ctx.snaps[c][0].index)
            print(f'  lags {c}: {have[c].shape[1]} cols ({time.time() - t0:.0f}s)',
                  flush=True)
        pd.concat([have[c].assign(_cutoff=c) for c in sorted(have)])           .rename_axis('ID').reset_index().to_parquet(p, index=False)
    return {c: have[c] for c in cuts}


def track_r5(ctx, args):
    base = FIVE_SEED['hurdle (sweep2 b1)']
    cuts = sorted(set(ctx.monthly + list(V.FOLDS) + [W.TEST_CUTOFF]))
    extra = lag_extra(ctx, cuts)
    W.SEEDS = list(SCREEN)
    t0 = time.time()
    rf, rn = W.reg_rmses(ctx, 'hurdle', mode='monthly', extra=extra)
    print(f'  hurdle + lag series ({time.time() - t0:.0f}s)')
    print('\n--- R5 monthly lag series, regressors (5-seed) ---')
    header()
    rows = [row('hurdle [5s, base]', base['rf'], base['rn'], base['rf'], base['rn']),
            row('hurdle + lag series [5s]', rf, rn, base['rf'], base['rn'])]
    save('r5', dict(rows=rows, rf=rf, rn=rn))
    return rows


TRACKS['r4'] = track_r4
TRACKS['r5'] = track_r5


# --- shared track-L harness ---------------------------------------------------

BIG4 = C.BIG
V3_ALPHA = 0.5


def v3_rule(ctx, proba, cutoff, labels):
    """The v3 decision rule, recomputed on the probabilities it is applied to:
    IPF the predicted mix to the last fully observed snapshot's true mix over the
    four large classes, then damp the weights by alpha=0.5.

    Recomputing matters. The IPF weights are a property of the probability
    matrix, so carrying bag20's weights onto a different model would measure the
    new model against a rule calibrated for the old one. The PROCEDURE is held
    fixed (stale target, 4 large classes, alpha 0.5); only the weights move.
    """
    import project_mix as PM
    big = [labels.index(l) for l in BIG4]
    shares = PM.share_matrix(ctx.snaps, ctx.monthly, labels)
    tgt = PM.project(shares, M.trainable_for(ctx.monthly, cutoff), cutoff, labels, 'stale')
    w, err = C.ipf(proba, tgt, labels, big)
    return C.ap(proba, w ** V3_ALPHA, labels), w, err


def l_eval(name, ctx, proba_by_fold, labels, extra_note=''):
    """F1 of a probability model under the v3 rule, with v3's regressions held
    fixed so touches=('f1',)."""
    f1s = []
    for v in V.FOLDS:
        pred, _, err = v3_rule(ctx, proba_by_fold[v], v, labels)
        f1s.append(C.f1w(ctx.snaps[v][1].Opportunity.to_numpy(), pred))
    s = score(f1s, V3_RF, V3_RN)
    base = score(V3_F1, V3_RF, V3_RN)
    gain = s - base
    both = all(f1s[i] > V3_F1[i] for i in (0, 1))
    worst = max(V3_F1[i] - f1s[i] for i in (0, 1))
    ok = gain >= 0.0015 and both and worst <= 0.001
    why = []
    if gain < 0.0015:
        why.append(f'gain {gain:+.5f}<0.0015')
    if not both:
        why.append('one fold down')
    if worst > 0.001:
        why.append(f'f1 -{worst:.4f}')
    verdict = 'KEPT' if ok else 'dropped'
    print(f'{name:<40}{f1s[0]:>9.4f}{f1s[1]:>9.4f}{np.mean(f1s):>9.4f}{s:>10.5f}'
          f'{gain:>+10.5f}  {verdict} {" ".join(why)} {extra_note}')
    return dict(name=name, f1=f1s, score=s, gain=gain, keep=verdict,
                note=' '.join(why))


def l_header():
    print(f'{"variant":<40}{"F1 f1":>9}{"F1 f2":>9}{"mean":>9}{"score":>10}{"gain":>10}'
          f'  verdict')


def bag20_proba(labels):
    out = {}
    for v in V.FOLDS:
        p = pd.read_parquet(f'preds/oof_bag20_{v}.parquet').set_index('ID')
        out[v] = p[labels].to_numpy()
    return out


def clf_proba(ctx, seeds, extra=None, mode='monthly'):
    """LightGBM 17-class probabilities per fold, optionally with extra features."""
    out = {}
    for v in V.FOLDS:
        D = ctx.fold(v, mode=mode, extra=extra)
        out[v] = W.bag_clf(D['Xtr'], D['ycode'], D['Xva'], ctx.labels,
                           seeds=seeds).to_numpy()
    return out


# --- L1 lag series as classifier features -------------------------------------

def track_l1(ctx, args):
    labels = ctx.labels
    cuts = sorted(set(ctx.monthly + list(V.FOLDS) + [W.TEST_CUTOFF]))
    extra = lag_extra(ctx, cuts)
    print('\n--- L1 lag series as classifier features ---')
    l_header()
    rows = []
    # reference: bag20 probabilities under the same rule (must reproduce v3)
    rows.append(l_eval('bag20 [20s, incumbent]', ctx, bag20_proba(labels), labels))
    for tag, seeds in (('[5s]', SCREEN),):
        rows.append(l_eval(f'base features {tag}', ctx, clf_proba(ctx, seeds), labels))
        t0 = time.time()
        rows.append(l_eval(f'+ lag series {tag}', ctx,
                           clf_proba(ctx, seeds, extra=extra), labels,
                           extra_note=f'({time.time() - t0:.0f}s)'))
    best = max(rows[1:], key=lambda r: r['score'])
    if best['name'].startswith('+ lag') and best['score'] > rows[1]['score']:
        print('\n  lag series beat base features at 5 seeds; confirming at 20')
        l_header()
        rows.append(l_eval('+ lag series [20s]', ctx,
                           clf_proba(ctx, CONFIRM, extra=extra), labels))
    save('l1', dict(rows=rows))
    return rows


TRACKS['l1'] = track_l1


# --- lag series as ONE change across all three models -------------------------

def cached_clf_proba(ctx, seeds, extra, tag, cutoffs=None):
    """LightGBM 17-class probabilities per fold, cached so track D does not pay
    for a refit."""
    out = {}
    for v in V.FOLDS:
        p = Path(f'preds/proba_{tag}_{v}.parquet')
        if p.exists():
            out[v] = pd.read_parquet(p).set_index('ID')[ctx.labels].to_numpy()
            continue
        t0 = time.time()
        D = ctx.fold(v, mode='monthly', extra=extra)
        pr = W.bag_clf(D['Xtr'], D['ycode'], D['Xva'], ctx.labels, seeds=seeds)
        pr.rename_axis('ID').reset_index().to_parquet(p, index=False)
        out[v] = pr[ctx.labels].to_numpy()
        print(f'  classifier probabilities {v} ({time.time() - t0:.0f}s)', flush=True)
    return out


def combined_rule(f1s, rf, rn):
    """Standing rule on a change that touches all three components."""
    s = score(f1s, rf, rn)
    b = score(V3_F1, V3_RF, V3_RN)
    per = [M.combined(f1s[i], rf[i], rn[i]) for i in (0, 1)]
    ref = [M.combined(V3_F1[i], V3_RF[i], V3_RN[i]) for i in (0, 1)]
    both = all(per[i] > ref[i] for i in (0, 1))
    worst, wname = 0.0, ''
    for i in (0, 1):
        for nm, dv in (('f1', V3_F1[i] - f1s[i]), ('rmse_fuel', rf[i] - V3_RF[i]),
                       ('rmse_nonfuel', rn[i] - V3_RN[i])):
            if dv > worst:
                worst, wname = dv, nm
    gain = s - b
    ok = gain >= 0.0015 and both and worst <= 0.001
    why = []
    if gain < 0.0015:
        why.append(f'gain {gain:+.5f}<0.0015')
    if not both:
        why.append('one fold down')
    if worst > 0.001:
        why.append(f'{wname} -{worst:.4f}')
    return dict(score=s, gain=gain, per_fold=per, ref_per_fold=ref,
                keep='KEPT' if ok else 'dropped', note=' '.join(why),
                worst=worst, worst_name=wname)


def track_v4lags(ctx, args):
    labels = ctx.labels
    big = [labels.index(l) for l in BIG4]
    cuts = sorted(set(ctx.monthly + list(V.FOLDS) + [W.TEST_CUTOFF]))
    extra = lag_extra(ctx, cuts)

    print('  classifier: lag features, 20 seeds')
    pr = cached_clf_proba(ctx, CONFIRM, extra, 'lags20')
    f1s = []
    for v in V.FOLDS:
        pred, _, _ = v3_rule(ctx, pr[v], v, labels)
        f1s.append(C.f1w(ctx.snaps[v][1].Opportunity.to_numpy(), pred))

    print('  regressors: hurdle + lag features, 20 seeds')
    W.SEEDS = list(CONFIRM)
    t0 = time.time()
    rf, rn = W.reg_rmses(ctx, 'hurdle', mode='monthly', extra=extra)
    print(f'    ({time.time() - t0:.0f}s)')

    j = combined_rule(f1s, rf, rn)
    print('\n--- lag series as ONE change across all three models, 20 seeds ---')
    print(f'{"":<22}{"F1 f1":>9}{"F1 f2":>9}{"rmse_f f1":>11}{"rmse_f f2":>11}'
          f'{"rmse_n f1":>11}{"rmse_n f2":>11}')
    print(f'{"v3 (incumbent)":<22}{V3_F1[0]:>9.4f}{V3_F1[1]:>9.4f}{V3_RF[0]:>11.4f}'
          f'{V3_RF[1]:>11.4f}{V3_RN[0]:>11.4f}{V3_RN[1]:>11.4f}')
    print(f'{"v4 lags":<22}{f1s[0]:>9.4f}{f1s[1]:>9.4f}{rf[0]:>11.4f}{rf[1]:>11.4f}'
          f'{rn[0]:>11.4f}{rn[1]:>11.4f}')
    print(f'{"delta":<22}{f1s[0]-V3_F1[0]:>+9.4f}{f1s[1]-V3_F1[1]:>+9.4f}'
          f'{rf[0]-V3_RF[0]:>+11.4f}{rf[1]-V3_RF[1]:>+11.4f}'
          f'{rn[0]-V3_RN[0]:>+11.4f}{rn[1]-V3_RN[1]:>+11.4f}')
    print(f'\n  per-fold combined score: v3 {j["ref_per_fold"][0]:.5f} / '
          f'{j["ref_per_fold"][1]:.5f}   v4 {j["per_fold"][0]:.5f} / {j["per_fold"][1]:.5f}')
    print(f'  mean {j["score"]:.5f} vs 0.28929, gain {j["gain"]:+.5f}')
    print(f'  worst component move: {j["worst_name"] or "none"} '
          f'-{j["worst"]:.4f} (slack 0.001)')
    print(f'  -> {j["keep"]} {j["note"]}')

    # --- track D on the cached lag probabilities --------------------------
    print('\n--- track D: alpha grid on the lag classifier probabilities ---')
    import project_mix as PM
    shares = PM.share_matrix(ctx.snaps, ctx.monthly, labels)
    W_ipf, truth = {}, {}
    for v in V.FOLDS:
        tgt = PM.project(shares, M.trainable_for(ctx.monthly, v), v, labels, 'stale')
        W_ipf[v], _ = C.ipf(pr[v], tgt, labels, big)
        truth[v] = ctx.snaps[v][1].Opportunity.to_numpy()
    for al in (0.25, 0.5, 0.75, 1.0):
        fs = [C.f1w(truth[v], C.ap(pr[v], W_ipf[v] ** al, labels)) for v in V.FOLDS]
        print(f'    alpha {al}: F1 {fs[0]:.4f} / {fs[1]:.4f}  mean {np.mean(fs):.4f}')
    picks = {}
    for fit_on in V.FOLDS:
        other = [x for x in V.FOLDS if x != fit_on][0]
        pa = max((0.25, 0.5, 0.75, 1.0),
                 key=lambda al: C.f1w(truth[fit_on],
                                      C.ap(pr[fit_on], W_ipf[fit_on] ** al, labels)))
        picks[other] = (pa, C.f1w(truth[other], C.ap(pr[other], W_ipf[other] ** pa, labels)))
    agree = len({p[0] for p in picks.values()}) == 1
    print('    cross-fold: ' + ' | '.join(
        f'alpha {p[0]} from the other fold, held-out F1 on {k} = {p[1]:.4f}'
        for k, p in picks.items()) + ('  AGREE' if agree else '  DISAGREE'))
    alpha = list(picks.values())[0][0] if agree else 0.5
    print(f'    -> using alpha {alpha}' + ('' if agree else ' (folds disagreed, '
                                           'keeping 0.5)'))
    if alpha != V3_ALPHA:
        f1s = [C.f1w(truth[v], C.ap(pr[v], W_ipf[v] ** alpha, labels)) for v in V.FOLDS]
        j = combined_rule(f1s, rf, rn)
        print(f'    with alpha {alpha}: F1 {f1s[0]:.4f} / {f1s[1]:.4f}, score '
              f'{j["score"]:.5f}, gain {j["gain"]:+.5f} -> {j["keep"]} {j["note"]}')

    save('v4lags', dict(f1=f1s, rf=rf, rn=rn, alpha=alpha, agree=bool(agree),
                        **{k: v for k, v in j.items() if k != 'ref_per_fold'}))
    if j['keep'] != 'KEPT':
        print('\nthe combined change does not pass -> not writing submission_v4_lags.csv')
        return j
    write_v4_lags(ctx, extra, alpha)
    return j


def write_v4_lags(ctx, extra, alpha, out='submissions/submission_v4_lags.csv'):
    labels = ctx.labels
    big = [labels.index(l) for l in BIG4]
    import project_mix as PM
    cutoffs = W.schedule(W.TEST_CUTOFF, 'monthly')
    ctx.pool(cutoffs)
    Xtr, ytr, _ = W.assemble(ctx.snaps, cutoffs, ('base',), ctx.bc, extra=extra)
    Xte = T.select(ctx.snaps[W.TEST_CUTOFF][0], ('base',), ctx.bc).join(extra[W.TEST_CUTOFF])
    print(f'  final fit: {len(Xtr)} rows, {Xtr.shape[1]} features, {len(CONFIRM)} seeds')
    W.SEEDS = list(CONFIRM)
    code = {l: i for i, l in enumerate(labels)}
    t0 = time.time()
    pte = W.bag_clf(Xtr, ytr.Opportunity.map(code).to_numpy(), Xte, labels,
                    seeds=CONFIRM).to_numpy()
    print(f'    classifier ({time.time() - t0:.0f}s)')
    shares = PM.share_matrix(ctx.snaps, ctx.monthly, labels)
    tgt = PM.project(shares, M.trainable_for(ctx.monthly, W.TEST_CUTOFF),
                     W.TEST_CUTOFF, labels, 'stale')
    w, err = C.ipf(pte, tgt, labels, big)
    opp = C.ap(pte, w ** alpha, labels)
    preds = {}
    for t in M.TARGETS:
        t0 = time.time()
        preds[t] = W.hurdle_reg(Xtr, ytr[t], Xte)
        print(f'    {t} ({time.time() - t0:.0f}s)')
    test_ids = pd.read_csv('data/test.csv', dtype=str).ID
    sub = pd.DataFrame({'CLV_fuel': preds['CLV_fuel'], 'CLV_nonfuel': preds['CLV_nonfuel'],
                        'Opportunity': opp}, index=Xte.index).reindex(test_ids)
    assert len(sub) == 5488, len(sub)
    assert sub.index.equals(pd.Index(test_ids)), 'IDs do not match data/test.csv'
    assert sub.notna().all().all(), 'missing predictions'
    assert not set(sub.Opportunity) - set(labels), 'label outside the config'
    assert (sub[['CLV_fuel', 'CLV_nonfuel']] >= 0).all().all(), 'negative CLV'
    assert list(sub.columns) == ['CLV_fuel', 'CLV_nonfuel', 'Opportunity'], sub.columns
    sub.rename_axis('ID').reset_index().to_csv(out, index=False, float_format='%.17g')
    print(f'  wrote {out}: 5488 rows, IDs match, labels in config, CLV >= 0 '
          f'(min {sub[["CLV_fuel", "CLV_nonfuel"]].min().min():.4f}), mix-err {err:.3f}')
    print('  mix: ' + C.fmt_mix(C.mix(sub.Opportunity.to_numpy(), labels), labels))
    return sub


TRACKS['v4lags'] = track_v4lags



# --- diagnostics --------------------------------------------------------------

def write_diag_r(ctx, out='submissions/submission_diag_r1_bundle.csv'):
    """Track R's best non-passing variant: v3's labels with the R1 bundle
    regressions (f3+f4 features, CatBoost magnitude), 20 seeds, all snapshots."""
    labels = ctx.labels
    cutoffs = W.schedule(W.TEST_CUTOFF, 'monthly')
    ctx.pool(cutoffs)
    extra = W.f_snapshots(ctx, ('f3', 'f4'), sorted(set(cutoffs + [W.TEST_CUTOFF])))
    Xte = T.select(ctx.snaps[W.TEST_CUTOFF][0], ('base',), ctx.bc).join(extra[W.TEST_CUTOFF])
    Xtr, ytr, _ = W.assemble(ctx.snaps, cutoffs, ('base',), ctx.bc, extra=extra)
    W.SEEDS = list(CONFIRM)
    print(f'  fitting the R1 bundle on {len(Xtr)} rows, {Xtr.shape[1]} features, '
          f'{len(CONFIRM)} seeds')
    preds = {}
    for t in M.TARGETS:
        t0 = time.time()
        preds[t] = W.hurdle_reg(Xtr, ytr[t], Xte, cat=True)
        print(f'    {t} ({time.time() - t0:.0f}s)', flush=True)
    v3 = pd.read_csv('submissions/submission_v3.csv', dtype={'ID': str},
                     float_precision='round_trip').set_index('ID')
    test_ids = pd.read_csv('data/test.csv', dtype=str).ID
    sub = pd.DataFrame({'CLV_fuel': preds['CLV_fuel'], 'CLV_nonfuel': preds['CLV_nonfuel']},
                       index=Xte.index).reindex(test_ids)
    sub['Opportunity'] = v3.Opportunity.reindex(test_ids).to_numpy()
    assert len(sub) == 5488 and sub.index.equals(pd.Index(test_ids))
    assert sub.notna().all().all() and not set(sub.Opportunity) - set(labels)
    assert (sub[['CLV_fuel', 'CLV_nonfuel']] >= 0).all().all()
    assert list(sub.columns) == ['CLV_fuel', 'CLV_nonfuel', 'Opportunity']
    sub.rename_axis('ID').reset_index().to_csv(out, index=False, float_format='%.17g')
    print(f'  wrote {out}: 5488 rows, labels identical to v3, CLV from the R1 bundle')
    return sub


TRACKS['diag_r'] = lambda ctx, args: write_diag_r(ctx)






def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--track', nargs='+', required=True)
    ap.add_argument('--train', default='data/train.csv')
    a = ap.parse_args()
    ctx = W.Ctx(a.train)
    for t in a.track:
        print(f'\n########## track {t} ##########')
        t0 = time.time()
        TRACKS[t](ctx, a)
        print(f'[track {t} took {time.time() - t0:.0f}s]')


if __name__ == '__main__':
    main()
