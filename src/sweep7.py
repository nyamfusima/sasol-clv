"""Sweep 7: expected-F1 decoding (E1) and two regression ideas (E2).

E1 is parameter-free by construction. It picks the assignment that maximises the
model's OWN expected weighted F1, using probabilities only -- no true labels
enter the rule -- so there is nothing to fit and nothing to overfit. That matters
here because the project's record is unambiguous: derived, low-dimensional
adjustments transferred to the public board and freely fitted ones did not.

    Expected weighted F1 = sum_c (n_c/N) * 2*TP_c / (n_c + m_c)

with m_c the rows assigned to c and TP_c the sum of P[i,c] over them. Moving row
i into class c changes the objective by a_c*P[i,c] - b_c to first order, where
a_c = 2*(n_c/N)/(n_c+m_c) and b_c = a_c*TP_c/(n_c+m_c), which gives the fixed
point iterated below.

    python src/sweep7.py --block e1
    python src/sweep7.py --block e2            # 5-seed screen
    python src/sweep7.py --block e2 --confirm  # 20-seed confirmation
"""
import argparse, json, time
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.metrics import f1_score
from sklearn.isotonic import IsotonicRegression
import make_submission as M
import sweep2 as W
import sweep5 as S5
import validate as V

RESULTS = Path('preds/sweep7.json')
LAMBDAS = (0.0, 0.25, 0.5, 0.75, 1.0)
NL = chr(10)


def save(key, obj):
    RESULTS.parent.mkdir(exist_ok=True)
    allr = json.loads(RESULTS.read_text()) if RESULTS.exists() else {}
    allr[key] = obj
    RESULTS.write_text(json.dumps(allr, indent=2, default=float))


# --- E1 expected-F1 decoding -------------------------------------------------

def expected_wf1(P, assign, n_c):
    """The objective itself, computed from probabilities only."""
    K = P.shape[1]
    N = float(n_c.sum())
    m = np.bincount(assign, minlength=K).astype(float)
    tp = np.zeros(K)
    np.add.at(tp, assign, P[np.arange(len(assign)), assign])
    denom = np.where(n_c + m > 0, n_c + m, 1.0)
    return float(np.sum((n_c / N) * 2.0 * tp / denom))


def decode(P, n_c, iters=200):
    """Fixed-point iteration on the first-order gain. Returns the best
    assignment seen, so an oscillation cannot make the result worse."""
    K = P.shape[1]
    N = float(n_c.sum())
    assign = P.argmax(axis=1)
    best_obj, best_assign = expected_wf1(P, assign, n_c), assign.copy()
    seen, damp, sweeps = {}, 1.0, 0
    for it in range(iters):
        sweeps = it + 1
        m = np.bincount(assign, minlength=K).astype(float)
        tp = np.zeros(K)
        np.add.at(tp, assign, P[np.arange(len(assign)), assign])
        denom = np.where(n_c + m > 0, n_c + m, 1.0)
        a = 2.0 * (n_c / N) / denom
        b = a * tp / denom
        new = (P * a - b).argmax(axis=1)
        if damp < 1.0:                      # move only a deterministic subset
            keep = (np.arange(len(new)) % int(round(1 / damp))) != 0
            new = np.where(keep, assign, new)
        obj = expected_wf1(P, new, n_c)
        if obj > best_obj:
            best_obj, best_assign = obj, new.copy()
        key = new.tobytes()
        if key in seen:                     # cycle: damp and keep going
            damp = max(damp / 2.0, 0.125)
            seen.clear()
        seen[key] = it
        if np.array_equal(new, assign):
            break
        assign = new
    return best_assign, best_obj, sweeps


def coordinate_ascent(P, assign, n_c):
    """Exact one-row-at-a-time ascent, used only as a check that the fixed point
    is not leaving obvious gains unclaimed and never reduces the objective."""
    K = P.shape[1]
    N = float(n_c.sum())
    a = assign.copy()
    m = np.bincount(a, minlength=K).astype(float)
    tp = np.zeros(K)
    np.add.at(tp, a, P[np.arange(len(a)), a])
    w = n_c / N
    moved = 0
    for i in range(len(a)):
        u = a[i]
        cur_u = w[u] * 2.0 * tp[u] / (n_c[u] + m[u])
        new_u = w[u] * 2.0 * (tp[u] - P[i, u]) / max(n_c[u] + m[u] - 1, 1e-9)
        gain = np.full(K, -np.inf)
        for v in range(K):
            if v == u:
                continue
            cur_v = w[v] * 2.0 * tp[v] / (n_c[v] + m[v])
            new_v = w[v] * 2.0 * (tp[v] + P[i, v]) / (n_c[v] + m[v] + 1)
            gain[v] = (new_u - cur_u) + (new_v - cur_v)
        v = int(gain.argmax())
        if gain[v] > 1e-15:
            tp[u] -= P[i, u]
            m[u] -= 1
            tp[v] += P[i, v]
            m[v] += 1
            a[i] = v
            moved += 1
    return a, moved


def support_choices(ctx, P, cutoff, labels):
    """n_c under the three definitions. Keys must not embed the source cutoff:
    it differs by fold (each fold's stale source is its own last fully observed
    snapshot) and the three choices are compared across folds by name."""
    n = len(P)
    model = P.sum(axis=0)
    tgt, src = M.target_mix(ctx.snaps, ctx.monthly, cutoff, labels)
    stale = n * tgt
    return {'model-implied': model, 'stale mix': stale,
            'average of the two': 0.5 * (model + stale)}, src


def track_e1(ctx, args):
    labels = ctx.labels
    big = [labels.index(l) for l in S5.BIG4]
    f1w = lambda y, p: f1_score(y, p, average='weighted', zero_division=0)
    P, truth = {}, {}
    for v in V.FOLDS:
        y = ctx.snaps[v][1]
        P[v] = (pd.read_parquet(f'preds/proba_lags20_{v}.parquet').set_index('ID')
                .reindex(y.index)[labels].to_numpy())
        truth[v] = y.Opportunity.to_numpy()

    ref = {'argmax': {v: np.array(labels)[P[v].argmax(axis=1)] for v in V.FOLDS}}
    ipf_w = {}
    for v in V.FOLDS:
        tgt, _ = M.target_mix(ctx.snaps, ctx.monthly, v, labels)
        ipf_w[v], _ = M.ipf(P[v], tgt, labels, big)
    for al in (0.5, 0.75):
        ref[f'alpha {al}'] = {v: np.array(labels)[(P[v] * ipf_w[v] ** al).argmax(axis=1)]
                              for v in V.FOLDS}

    print(NL + '--- E1 expected-F1 decoding, parameter-free ---')
    print(f'{"rule":<32}{"fold 2025-06":>14}{"fold 2025-09":>14}{"mean":>9}')
    rows = []
    for name, pred in ref.items():
        fs = [f1w(truth[v], pred[v]) for v in V.FOLDS]
        rows.append(dict(name=name, f1=fs, kind='reference'))
        print(f'{name:<32}{fs[0]:>14.4f}{fs[1]:>14.4f}{np.mean(fs):>9.4f}')

    eff = {}
    for v in V.FOLDS:
        choices, src = support_choices(ctx, P[v], v, labels)
        eff[v] = {}
        for cname, n_c in choices.items():
            a, obj, sweeps = decode(P[v], n_c)
            ca, moved = coordinate_ascent(P[v], a, n_c)
            eff[v][cname] = dict(assign=a, obj=obj, sweeps=sweeps, moved=moved,
                                 obj_ca=expected_wf1(P[v], ca, n_c), n_c=n_c, src=src)
    for cname in list(eff[V.FOLDS[0]].keys()):
        fs = [f1w(truth[v], np.array(labels)[eff[v][cname]['assign']]) for v in V.FOLDS]
        rows.append(dict(name=f'expected-F1, {cname}', f1=fs, kind='e1'))
        print(f'{"expected-F1, " + cname:<32}{fs[0]:>14.4f}{fs[1]:>14.4f}'
              f'{np.mean(fs):>9.4f}')
        for v in V.FOLDS:
            e = eff[v][cname]
            flag = 'OK' if e['obj_ca'] >= e['obj'] - 1e-12 else 'WARNING: ascent LOWERED it'
            print(f'    {v}: {e["sweeps"]} sweeps to objective {e["obj"]:.6f}; exact '
                  f'coordinate ascent moved {e["moved"]} rows to {e["obj_ca"]:.6f} [{flag}]')

    print(NL + '  predicted class mix on fold 2025-09 (4 large classes):')
    vv = V.FOLDS[1]
    hdr = ''.join(f'{l.split(": ")[-1][:9]:>11}' for l in S5.BIG4)
    print('    ' + f'{"rule":<32}' + hdr)
    print('    ' + f'{"TRUE":<32}'
          + ''.join(f'{(truth[vv] == l).mean():>11.3f}' for l in S5.BIG4))
    for name, pred in ref.items():
        print('    ' + f'{name:<32}'
              + ''.join(f'{(pred[vv] == l).mean():>11.3f}' for l in S5.BIG4))
    for cname in eff[vv]:
        pl = np.array(labels)[eff[vv][cname]['assign']]
        print('    ' + f'{"expected-F1, " + cname:<32}'
              + ''.join(f'{(pl == l).mean():>11.3f}' for l in S5.BIG4))

    print(NL + '  effective per-class multiplier at convergence (fold 2025-09, n_c'
          + NL + '  averaged) against the IPF weights. The decoding rule is AFFINE'
          + NL + '  (a_c*P - b_c), so a_c is only its multiplicative part.')
    e = eff[vv]['average of the two']
    K = len(labels)
    m = np.bincount(e['assign'], minlength=K).astype(float)
    tp = np.zeros(K)
    np.add.at(tp, e['assign'], P[vv][np.arange(len(e['assign'])), e['assign']])
    n_c = e['n_c']
    denom = np.where(n_c + m > 0, n_c + m, 1.0)
    a_c = 2.0 * (n_c / n_c.sum()) / denom
    b_c = a_c * tp / denom
    a_n = a_c / np.exp(np.log(np.clip(a_c, 1e-12, None)).mean())
    print(f'      {"class":<40}{"a_c norm":>10}{"b_c":>9}{"IPF^0.5":>10}{"IPF^0.75":>10}')
    for j, l in enumerate(labels):
        mark = '*' if l in S5.BIG4 else ' '
        print(f'    {mark} {l:<40}{a_n[j]:>10.3f}{b_c[j]:>9.4f}'
              f'{ipf_w[vv][j] ** 0.5:>10.3f}{ipf_w[vv][j] ** 0.75:>10.3f}')
    bi = [labels.index(l) for l in S5.BIG4]
    spread = lambda z: float(np.ptp(z[bi]) / np.mean(z[bi]))
    print(f'    (* = a matched class.) Relative spread over those four: a_c '
          f'{spread(a_n):.3f},' + NL + f'    IPF^0.5 {spread(ipf_w[vv] ** 0.5):.3f}, '
          f'IPF^0.75 {spread(ipf_w[vv] ** 0.75):.3f}; |b_c| at most {np.abs(b_c).max():.5f}.'
          + NL + '    So the rule is effectively multiplicative with NEARLY FLAT '
          'weights on the' + NL + '    large classes, where IPF is strongly '
          'class-specific.')

    best = {V.FOLDS[i]: max(rows[1]['f1'][i], rows[2]['f1'][i]) for i in (0, 1)}
    bar = float(np.mean(rows[2]['f1']))
    print(NL + f'  keep rule: mean F1 >= {bar:.4f} (alpha 0.75) and neither fold more '
          f'than 0.001' + NL + '  below the better of alpha 0.5 / 0.75 on that fold '
          f'({best[V.FOLDS[0]]:.4f} / {best[V.FOLDS[1]]:.4f})')
    passed = []
    for r in rows:
        if r['kind'] != 'e1':
            continue
        ok_mean = np.mean(r['f1']) >= bar - 1e-12
        ok_f = [r['f1'][i] >= best[v] - 0.001 for i, v in enumerate(V.FOLDS)]
        why = []
        if not ok_mean:
            why.append(f'mean {np.mean(r["f1"]):.4f} < {bar:.4f}')
        for i, v in enumerate(V.FOLDS):
            if not ok_f[i]:
                why.append(f'{v} {best[v] - r["f1"][i]:.4f} below best')
        r['keep'] = 'KEPT' if (ok_mean and all(ok_f)) else 'dropped'
        print(f'    {r["name"]:<36}{r["keep"]}  {"; ".join(why)}')
        if r['keep'] == 'KEPT':
            passed.append(r)

    amax = float(np.mean(rows[0]['f1']))
    bestef1 = max(float(np.mean(r['f1'])) for r in rows if r['kind'] == 'e1')
    print(NL + '  share of the argmax -> alpha 0.75 gain recovered with no fitted'
          + NL + f'  parameter: {(bestef1 - amax) / (bar - amax):.0%} '
          f'({bestef1 - amax:+.4f} of {bar - amax:+.4f})')
    save('e1', dict(rows=rows, recovered=(bestef1 - amax) / (bar - amax)))
    return rows, passed


# --- E2 regressions ----------------------------------------------------------
# One fitting pass produces everything both sub-blocks need, so a 20-seed
# confirmation is paid for once rather than twice.

def reg_cache(ctx, seeds, tag):
    """Gate, magnitude and direct prediction at each fold cutoff, plus the gate
    and outcome at the latest snapshot fully observed BEFORE that fold -- the
    only data E2b's calibrator is allowed to see."""
    path = Path(f'preds/sweep7_reg_{tag}.npz')
    if path.exists():
        z = np.load(path)
        print(f'  [reusing {path}]')
        return {k: z[k] for k in z.files}
    out = {}
    for v in V.FOLDS:
        src = M.trainable_for(ctx.monthly, v)[-1]
        Dv, Ds = ctx.fold(v), ctx.fold(src)
        print(f'  fold {v}: calibration source {src} ({len(Ds["Xtr"])} train rows, '
              f'{len(Ds["Xva"])} source rows); fold has {len(Dv["Xva"])} rows')
        for t in W.TARGETS:
            t0 = time.time()
            yv = Dv['ytr'][t]
            pos = (yv.to_numpy() > 0)
            out[f'{v}|{t}|gate'] = W.bag_binary(Dv['Xtr'], pos.astype(int),
                                                Dv['Xva'], seeds=seeds)
            out[f'{v}|{t}|mag'] = W.bag_reg(Dv['Xtr'][pos], yv[pos], Dv['Xva'],
                                            seeds=seeds)
            out[f'{v}|{t}|direct'] = W.bag_reg(Dv['Xtr'], yv, Dv['Xva'], seeds=seeds)
            out[f'{v}|{t}|truth'] = Dv['yva'][t].to_numpy()
            ys = Ds['ytr'][t]
            ps = (ys.to_numpy() > 0)
            out[f'{v}|{t}|gate_src'] = W.bag_binary(Ds['Xtr'], ps.astype(int),
                                                    Ds['Xva'], seeds=seeds)
            out[f'{v}|{t}|pos_src'] = (Ds['yva'][t].to_numpy() > 0).astype(float)
            print(f'    {t}: {time.time() - t0:.0f}s')
    path.parent.mkdir(exist_ok=True)
    np.savez(path, **out)
    return out


def rmse(pred, truth):
    """W.rmse expects a pandas Series; the cache holds plain arrays."""
    return float(np.sqrt(((np.asarray(pred) - np.asarray(truth)) ** 2).mean()))


def hurdle_of(d, v, t):
    return np.clip(d[f'{v}|{t}|gate'] * d[f'{v}|{t}|mag'], 0, None)


def score_fixed_f1(rf, rn):
    """Combined score holding F1 at the v4_hybrid classifier's fold values, so
    every delta reported below is the regression change alone."""
    return [M.combined(S5.V4_F1[i], rf[i], rn[i]) for i in (0, 1)]


def judge_reg(name, rf, rn, ref_rf, ref_rn, ref_name):
    """Standing keep rule, applied only to the components a regression change
    touches: mean score +0.0015, better on both folds, neither RMSE worse by
    more than 0.001 on either fold."""
    per, ref = score_fixed_f1(rf, rn), score_fixed_f1(ref_rf, ref_rn)
    gain = float(np.mean(per) - np.mean(ref))
    both = all(per[i] > ref[i] for i in (0, 1))
    worst, wname = 0.0, ''
    for i in (0, 1):
        for nm, dv in (('rmse_fuel', rf[i] - ref_rf[i]),
                       ('rmse_nonfuel', rn[i] - ref_rn[i])):
            if dv > worst:
                worst, wname = dv, nm
    ok = gain >= W.SCORE_BAR and both and worst <= W.COMPONENT_SLACK
    why = []
    if gain < W.SCORE_BAR:
        why.append(f'gain {gain:+.5f} < {W.SCORE_BAR}')
    if not both:
        why.append('not better on both folds')
    if worst > W.COMPONENT_SLACK:
        why.append(f'{wname} worse by {worst:.4f}')
    print(f'    {name:<38}score {np.mean(per):.5f} ({gain:+.5f} vs {ref_name})  '
          f'{"KEPT" if ok else "dropped"}  {"; ".join(why)}')
    return ok, gain


def track_e2a(ctx, args, d=None, seeds=None, tag='screen'):
    """Convex blend of the hurdle and the direct regressor on the transformed
    target. lambda is chosen on one fold and scored on the other, in both
    directions, so no number below is fitted on the fold it is reported on."""
    seeds = tuple(seeds or W.SEEDS)
    d = d if d is not None else reg_cache(ctx, seeds, tag)
    print(NL + f'--- E2a hurdle/direct blend, {len(seeds)} seeds: '
          'pred = (1-lam)*hurdle + lam*direct ---')
    curves, xfold = {}, {}
    for t in W.TARGETS:
        print(NL + f'  {t}')
        print(f'    {"lambda":<9}{"fold 2025-06":>14}{"fold 2025-09":>14}{"mean":>9}')
        curves[t] = {}
        for lam in LAMBDAS:
            r = []
            for v in V.FOLDS:
                p = ((1 - lam) * hurdle_of(d, v, t)
                     + lam * np.clip(d[f'{v}|{t}|direct'], 0, None))
                r.append(rmse(p, d[f'{v}|{t}|truth']))
            curves[t][lam] = r
            tail = '  <- hurdle' if lam == 0 else ('  <- direct' if lam == 1 else '')
            print(f'    {lam:<9.2f}{r[0]:>14.4f}{r[1]:>14.4f}{np.mean(r):>9.4f}{tail}')
        held = {}
        for i, j in ((0, 1), (1, 0)):
            lam = min(LAMBDAS, key=lambda L: curves[t][L][i])
            held[j] = curves[t][lam][j]
            print(f'    chosen on fold {V.FOLDS[i]} -> lambda {lam:.2f}; held-out fold '
                  f'{V.FOLDS[j]} RMSE {curves[t][lam][j]:.4f} against hurdle '
                  f'{curves[t][0.0][j]:.4f} ({curves[t][lam][j] - curves[t][0.0][j]:+.4f})')
        xfold[t] = [held[0], held[1]]
    print(NL + '  cross-fold blend against the pure hurdle, same seeds:')
    ok, gain = judge_reg('blend, cross-fold lambda', xfold['CLV_fuel'],
                         xfold['CLV_nonfuel'],
                         [curves['CLV_fuel'][0.0][i] for i in (0, 1)],
                         [curves['CLV_nonfuel'][0.0][i] for i in (0, 1)],
                         f'{len(seeds)}-seed hurdle')
    save(f'e2a_{tag}', dict(curves={t: {str(k): val for k, val in c.items()}
                                    for t, c in curves.items()},
                            cross_fold=xfold, gain=gain, kept=ok))
    return curves, xfold


def track_e2b(ctx, args, d=None, seeds=None, tag='screen'):
    """Isotonic calibration of the hurdle gate, fitted strictly forward in time
    on the last snapshot whose outcome window closes before the fold cutoff."""
    seeds = tuple(seeds or W.SEEDS)
    d = d if d is not None else reg_cache(ctx, seeds, tag)
    print(NL + f'--- E2b forward-in-time isotonic gate calibration, '
          f'{len(seeds)} seeds ---')
    cal, raw = {}, {}
    for t in W.TARGETS:
        print(NL + f'  {t}')
        for v in V.FOLDS:
            g = d[f'{v}|{t}|gate']
            iso = IsotonicRegression(out_of_bounds='clip', y_min=0.0, y_max=1.0)
            iso.fit(d[f'{v}|{t}|gate_src'], d[f'{v}|{t}|pos_src'])
            gc = iso.predict(g)
            tp = (d[f'{v}|{t}|truth'] > 0).astype(float)
            print(f'    fold {v}: P(y>0) base rate {d[f"{v}|{t}|pos_src"].mean():.3f} '
                  f'at the calibration source, {tp.mean():.3f} at the fold')
            q = pd.qcut(g, 10, labels=False, duplicates='drop')
            print(f'      {"decile":<7}{"n":>6}{"raw p":>9}{"calib p":>9}{"actual":>9}'
                  f'{"raw err":>9}{"cal err":>9}')
            for k in sorted(pd.unique(q)):
                msk = q == k
                a = tp[msk].mean()
                print(f'      {k + 1:<7}{msk.sum():>6}{g[msk].mean():>9.3f}'
                      f'{gc[msk].mean():>9.3f}{a:>9.3f}{g[msk].mean() - a:>+9.3f}'
                      f'{gc[msk].mean() - a:>+9.3f}')
            act = pd.Series(tp).groupby(q).mean()
            ar = float(np.abs(pd.Series(g).groupby(q).mean() - act).mean())
            ac = float(np.abs(pd.Series(gc).groupby(q).mean() - act).mean())
            print(f'      mean |decile gap|: raw {ar:.4f} -> calibrated {ac:.4f}')
            p = np.clip(gc * d[f'{v}|{t}|mag'], 0, None)
            cal.setdefault(t, {})[v] = rmse(p, d[f'{v}|{t}|truth'])
            raw.setdefault(t, {})[v] = rmse(hurdle_of(d, v, t), d[f'{v}|{t}|truth'])
            print(f'      RMSE: hurdle {raw[t][v]:.4f} -> calibrated gate '
                  f'{cal[t][v]:.4f} ({cal[t][v] - raw[t][v]:+.4f})')
    RF = [cal['CLV_fuel'][v] for v in V.FOLDS]
    RN = [cal['CLV_nonfuel'][v] for v in V.FOLDS]
    rrf = [raw['CLV_fuel'][v] for v in V.FOLDS]
    rrn = [raw['CLV_nonfuel'][v] for v in V.FOLDS]
    print(NL + '  calibrated gate against the pure hurdle, same seeds:')
    ok, gain = judge_reg('isotonic gate calibration', RF, RN, rrf, rrn,
                         f'{len(seeds)}-seed hurdle')
    save(f'e2b_{tag}', dict(rf=RF, rn=RN, raw_rf=rrf, raw_rn=rrn, gain=gain, kept=ok))
    return RF, RN


def track_e2(ctx, args):
    seeds = tuple(S5.CONFIRM if args.confirm else W.SEEDS)
    tag = 'confirm' if args.confirm else 'screen'
    d = reg_cache(ctx, seeds, tag)
    track_e2a(ctx, args, d, seeds, tag)
    track_e2b(ctx, args, d, seeds, tag)
    print(NL + '  (a 20-seed result is comparable with the shipped v3 regressions: '
          f'rmse_f {S5.V3_RF}, rmse_nf {S5.V3_RN})')


TRACKS = {'e1': track_e1, 'e2': track_e2, 'e2a': track_e2a, 'e2b': track_e2b}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--block', nargs='+', required=True)
    ap.add_argument('--train', default='data/train.csv')
    ap.add_argument('--confirm', action='store_true',
                    help='20 seeds (42-61) instead of the 5-seed screen')
    a = ap.parse_args()
    ctx = W.Ctx(a.train)
    for b in a.block:
        print(NL + f'########## block {b} ##########')
        t0 = time.time()
        TRACKS[b](ctx, a)
        print(f'[block {b} took {time.time() - t0:.0f}s]')


if __name__ == '__main__':
    main()
