"""Second sweep: blocks A-E, all 5-seed bagged, all judged on the combined score.

Every variant is bagged over seeds 42-46 (predictions averaged) so the reference
and the variants are compared like for like. Both folds train only on snapshots
whose 3-month outcome window ends on or before the validation cutoff.

    python src/sweep2.py --block ref        # the new bagged reference
    python src/sweep2.py --block a1         # recency weighting
    python src/sweep2.py --block a2 a3 a4   # several in one run
"""
import argparse, json, time
from pathlib import Path
import numpy as np, pandas as pd, lightgbm as lgb
from sklearn.metrics import roc_auc_score
import features as F
import snapshots as S
import validate as V
import train_label as T

SEEDS = [42, 43, 44, 45, 46]
TEST_CUTOFF = '2025-12-01'
RESULTS = Path('preds/sweep2_results.json')
BASE_CLF = dict(n_estimators=400, learning_rate=0.03, num_leaves=15, min_child_samples=30,
                subsample=0.8, subsample_freq=1, colsample_bytree=0.7)
BASE_REG = dict(n_estimators=500, learning_rate=0.03, num_leaves=31, min_child_samples=30,
                subsample=0.8, subsample_freq=1, colsample_bytree=0.7)
TARGETS = ['CLV_fuel', 'CLV_nonfuel']


# --- cutoff schedules --------------------------------------------------------

def weekly(first=V.FIRST_CUTOFF, last='2025-09-01', step_days=7):
    return [str(c.date()) for c in pd.date_range(first, last, freq=f'{step_days}D')]


def schedule(val_cutoff, mode, first=V.FIRST_CUTOFF):
    """Training cutoffs for one fold under a spacing mode. Anchored at the most
    recent usable cutoff and stepped backwards, so the freshest data is always
    in and only the older snapshots get thinned."""
    v = pd.Timestamp(val_cutoff)
    if mode in ('weekly', 'fortnightly'):
        pool = weekly(first, '2025-09-01', 7 if mode == 'weekly' else 14)
    else:
        pool = S.monthly(first, '2025-09-01')
    ok = [c for c in pool if pd.Timestamp(c) + pd.DateOffset(months=3) <= v]
    step = {'monthly': 1, '2month': 2, 'quarterly': 3, 'weekly': 1, 'fortnightly': 1}[mode]
    return sorted(ok[::-1][::step])


def age_months(cutoffs, ref):
    r = pd.Timestamp(ref)
    # str(c): cutoffs arrive as numpy.str_ from np.concatenate, which pandas
    # 3.0's Timestamp refuses
    return np.array([(r.year - pd.Timestamp(str(c)).year) * 12
                     + (r.month - pd.Timestamp(str(c)).month)
                     for c in cutoffs], dtype=float)


# --- fold assembly -----------------------------------------------------------

def assemble(snaps, cutoffs, blocks, bc, extra=None):
    X = pd.concat([T.select(snaps[c][0], blocks, bc) for c in cutoffs])
    y = pd.concat([snaps[c][1] for c in cutoffs])
    cut = np.concatenate([[c] * len(snaps[c][0]) for c in cutoffs])
    if extra is not None:
        X = X.join(pd.concat([extra[c] for c in cutoffs]))
    return X, y, cut


def weights_for(cut, ref, half_life):
    if half_life is None:
        return None
    return 0.5 ** (age_months(cut, ref) / half_life)


# --- bagged fitting ----------------------------------------------------------

def bag_clf(Xtr, ycode, Xva, labels, params=None, w=None, seeds=SEEDS, n_est=None, cat=False):
    """Average predict_proba over seeds; returns an (n x 17) frame."""
    p = np.zeros((len(Xva), len(labels)))
    params = dict(params or BASE_CLF)
    if n_est:
        params['n_estimators'] = n_est
    for s in seeds:
        if cat:
            from catboost import CatBoostClassifier
            m = CatBoostClassifier(iterations=params.get('n_estimators', 400),
                                   learning_rate=params.get('learning_rate', 0.03),
                                   depth=params.get('depth', 6), random_seed=s,
                                   loss_function='MultiClass', verbose=0,
                                   thread_count=-1)
            m.fit(Xtr, ycode, sample_weight=w)
            cls = [int(c) for c in m.classes_]
        else:
            m = lgb.LGBMClassifier(random_state=s, verbose=-1, **params)
            m.fit(Xtr, ycode, sample_weight=w)
            cls = list(m.classes_)
        p += T._wide(m.predict_proba(Xva), cls, labels, Xva.index).to_numpy()
    return pd.DataFrame(p / len(seeds), index=Xva.index, columns=labels)


def bag_reg(Xtr, y, Xva, params=None, w=None, seeds=SEEDS, cat=False, clip=True):
    out = np.zeros(len(Xva))
    params = dict(params or BASE_REG)
    for s in seeds:
        if cat:
            from catboost import CatBoostRegressor
            m = CatBoostRegressor(iterations=params.get('n_estimators', 500),
                                  learning_rate=params.get('learning_rate', 0.03),
                                  depth=params.get('depth', 6), random_seed=s,
                                  verbose=0, thread_count=-1)
            m.fit(Xtr, y, sample_weight=w)
        else:
            m = lgb.LGBMRegressor(random_state=s, verbose=-1, **params)
            m.fit(Xtr, y, sample_weight=w)
        out += m.predict(Xva)
    out /= len(seeds)
    return np.clip(out, 0, None) if clip else out


def bag_binary(Xtr, yb, Xva, params=None, w=None, seeds=SEEDS):
    """Average P(class 1) over seeds."""
    out = np.zeros(len(Xva))
    params = dict(params or BASE_CLF)
    for s in seeds:
        m = lgb.LGBMClassifier(random_state=s, verbose=-1, **params)
        m.fit(Xtr, yb, sample_weight=w)
        out += m.predict_proba(Xva)[:, list(m.classes_).index(1)]
    return out / len(seeds)


def hurdle_reg(Xtr, y, Xva, params=None, w=None, seeds=SEEDS):
    """(B1) P(y>0) * E[y | y>0], the second stage fitted on positive rows only."""
    pos = (y.to_numpy() > 0)
    if pos.all() or not pos.any():
        return bag_reg(Xtr, y, Xva, params, w, seeds)
    p = bag_binary(Xtr, pos.astype(int), Xva, BASE_CLF, w, seeds)
    wp = None if w is None else w[pos]
    mag = bag_reg(Xtr[pos], y[pos], Xva, params, wp, seeds)
    return np.clip(p * mag, 0, None)


# --- scoring -----------------------------------------------------------------

def rmse(pred, truth):
    return float(np.sqrt(((np.asarray(pred) - truth.to_numpy()) ** 2).mean()))


def save(block, rows):
    RESULTS.parent.mkdir(exist_ok=True)
    allr = json.loads(RESULTS.read_text()) if RESULTS.exists() else {}
    allr[block] = rows
    RESULTS.write_text(json.dumps(allr, indent=2))


def table(title, rows, ref=None, kind='score'):
    """rows: [{'name','f1','rf','rn'} or {'name','f1'} or {'name','rf','rn'}]"""
    print(f'\n{"=" * 92}\n{title}\n{"=" * 92}')
    hdr = f'{"variant":<34}{"fold 2025-06":>14}{"fold 2025-09":>14}{"mean":>11}{"delta":>11}  kept'
    print(hdr)
    for r in rows:
        a, b = r['f1'][0], r['f1'][1]
        m = (a + b) / 2
        d = '' if ref is None else f'{m - ref:+.5f}'
        print(f'{r["name"]:<34}{a:>14.4f}{b:>14.4f}{m:>11.4f}{d:>11}  {r.get("kept","")}')
    return rows


# --- one result row = per-fold (f1, rmse_fuel, rmse_nonfuel) -----------------

class Row:
    """A variant's two folds on all three targets, so the combined score is
    always computable. Components a variant does not touch are filled from the
    reference, which makes `score` directly comparable across blocks."""

    def __init__(self, name, f1=None, rf=None, rn=None, ref=None, note=''):
        g = lambda v, k: v if v is not None else [ref[k][i] for i in (0, 1)]
        self.name, self.note = name, note
        self.f1, self.rf, self.rn = g(f1, 'f1'), g(rf, 'rf'), g(rn, 'rn')
        self.kept = ''

    def score(self, i):
        return V.combined(self.f1[i], self.rf[i], self.rn[i])

    @property
    def mean_score(self):
        return (self.score(0) + self.score(1)) / 2

    def as_dict(self):
        return dict(name=self.name, f1=self.f1, rf=self.rf, rn=self.rn,
                    score=[self.score(0), self.score(1)], mean_score=self.mean_score,
                    kept=self.kept, note=self.note)


def report(title, rows, ref_row=None, metric='score'):
    """metric: 'score' (default), 'f1', 'rf', 'rn' -- picks which column drives
    the kept/dropped call and the delta shown."""
    get = {'score': lambda r: (r.score(0), r.score(1)),
           'f1': lambda r: (r.f1[0], r.f1[1]),
           'rf': lambda r: (r.rf[0], r.rf[1]),
           'rn': lambda r: (r.rn[0], r.rn[1])}[metric]
    lower_better = metric in ('rf', 'rn')
    print(f'\n{"=" * 104}\n{title}   [decided on {metric}, '
          f'{"lower" if lower_better else "higher"} better]\n{"=" * 104}')
    print(f'{"variant":<36}{"fold 2025-06":>13}{"fold 2025-09":>13}{"mean":>10}'
          f'{"delta":>10}{"score":>10}  kept')
    rr = get(ref_row) if ref_row else None
    refm = (rr[0] + rr[1]) / 2 if rr else None
    for r in rows:
        a, b = get(r)
        m = (a + b) / 2
        d = '' if refm is None else f'{m - refm:+.5f}'
        print(f'{r.name:<36}{a:>13.4f}{b:>13.4f}{m:>10.4f}{d:>10}'
              f'{r.mean_score:>10.5f}  {r.kept}{("  " + r.note) if r.note else ""}')
    return rows


def judge(rows, ref_row, metric, bar):
    """Apply the GO bar: mean gain >= bar AND better on BOTH folds."""
    get = {'score': lambda r: (r.score(0), r.score(1)),
           'f1': lambda r: (r.f1[0], r.f1[1]),
           'rf': lambda r: (r.rf[0], r.rf[1]),
           'rn': lambda r: (r.rn[0], r.rn[1])}[metric]
    sign = -1 if metric in ('rf', 'rn') else 1
    ra, rb = get(ref_row)
    for r in rows:
        if r is ref_row:
            r.kept = 'reference'
            continue
        a, b = get(r)
        gain = sign * (((a + b) / 2) - ((ra + rb) / 2))
        both = sign * (a - ra) > 0 and sign * (b - rb) > 0
        r.kept = 'KEPT' if (gain >= bar and both) else 'dropped'
        if gain >= bar and not both:
            r.note = 'mean bar met but one fold worse'
    return rows


# --- block context -----------------------------------------------------------

class Ctx:
    def __init__(self, train='data/train.csv', cfg=None):
        self.cfg = cfg or json.loads(Path('src/label_config.json').read_text())
        self.labels = self.cfg['opportunity_labels']
        self.cats = S.categories(self.cfg)
        self.code = {l: i for i, l in enumerate(self.labels)}
        self.train = train
        self.monthly = S.monthly(V.FIRST_CUTOFF, V.LAST_CUTOFF)
        self.snaps = S.load(self.monthly + [TEST_CUTOFF], train, self.cfg, ('base',),
                            label_cutoffs=self.monthly, verbose=False)
        self.bc = T.block_columns(list(self.snaps[self.monthly[0]][0].columns))

    def pool(self, cutoffs):
        """Load any cutoffs not already in hand (weekly/fortnightly schedules)."""
        need = [c for c in cutoffs if c not in self.snaps]
        if need:
            self.snaps.update(S.load(need, self.train, self.cfg, ('base',), verbose=True))
        return self.snaps

    def fold(self, val_cutoff, mode='monthly', half_life=None, extra=None):
        cuts = schedule(val_cutoff, mode)
        self.pool(cuts)
        Xtr, ytr, cut = assemble(self.snaps, cuts, ('base',), self.bc, extra)
        Xva, yva = self.snaps[val_cutoff]
        Xva = T.select(Xva, ('base',), self.bc)
        if extra is not None:
            Xva = Xva.join(extra[val_cutoff])
        return dict(cuts=cuts, Xtr=Xtr, ytr=ytr, cut=cut, Xva=Xva, yva=yva,
                    w=weights_for(cut, val_cutoff, half_life),
                    ycode=ytr.Opportunity.map(self.code).to_numpy())


def eval_clf(ctx, proba, yva):
    return V.f1(yva.Opportunity.to_numpy(), T.argmax_labels(proba, ctx.labels))


# --- reference ---------------------------------------------------------------

def block_ref(ctx):
    f1s, rfs, rns = [], [], []
    for v in V.FOLDS:
        D = ctx.fold(v)
        t0 = time.time()
        p = bag_clf(D['Xtr'], D['ycode'], D['Xva'], ctx.labels)
        f1s.append(eval_clf(ctx, p, D['yva']))
        preds = {t: bag_reg(D['Xtr'], D['ytr'][t], D['Xva']) for t in TARGETS}
        rfs.append(rmse(preds['CLV_fuel'], D['yva'].CLV_fuel))
        rns.append(rmse(preds['CLV_nonfuel'], D['yva'].CLV_nonfuel))
        print(f'  fold {v}: {len(D["cuts"])} snapshots, {len(D["Xtr"])} rows, '
              f'{time.time() - t0:.0f}s')
    ref = Row('reference (5-seed bag, monthly)', f1s, rfs, rns)
    ref.kept = 'reference'
    report('BAGGED REFERENCE (seeds 42-46)', [ref])
    print(f'\n  single-seed baseline for comparison: F1 0.4949 / 0.5047, '
          f'rmse_f 0.5998 / 0.6052, rmse_nf 0.7418 / 0.7489, score '
          f'{(V.combined(0.4949, 0.5998, 0.7418) + V.combined(0.5047, 0.6052, 0.7489)) / 2:.5f}')
    save('ref', [ref.as_dict()])
    return ref


def load_ref():
    d = json.loads(RESULTS.read_text())['ref'][0]
    r = Row(d['name'], d['f1'], d['rf'], d['rn'])
    r.kept = 'reference'
    return r


def ref_dict(ref):
    return dict(f1=ref.f1, rf=ref.rf, rn=ref.rn)


# --- A) training recipe for the classifier -----------------------------------

def block_a1(ctx, ref):
    rows = [ref]
    for hl in (3, 6, 12):
        f1s = []
        for v in V.FOLDS:
            D = ctx.fold(v, half_life=hl)
            f1s.append(eval_clf(ctx, bag_clf(D['Xtr'], D['ycode'], D['Xva'], ctx.labels,
                                             w=D['w']), D['yva']))
        rows.append(Row(f'a1 recency half-life {hl}mo', f1s, ref=ref_dict(ref)))
    judge(rows, ref, 'f1', 0.005)
    report('A1  RECENCY WEIGHTING (classifier)', rows, ref, 'f1')
    save('a1', [r.as_dict() for r in rows])
    return rows


def block_a2(ctx, ref):
    rows = [ref]
    for mode in ('2month', 'quarterly'):
        f1s = []
        for v in V.FOLDS:
            D = ctx.fold(v, mode=mode)
            f1s.append(eval_clf(ctx, bag_clf(D['Xtr'], D['ycode'], D['Xva'], ctx.labels),
                                D['yva']))
        rows.append(Row(f'a2 spacing {mode}', f1s, ref=ref_dict(ref),
                        note=f'{len(schedule(V.FOLDS[1], mode))} snaps on fold 2'))
    judge(rows, ref, 'f1', 0.005)
    report('A2  SNAPSHOT SPACING (classifier)', rows, ref, 'f1')
    save('a2', [r.as_dict() for r in rows])
    return rows


def block_a3(ctx, ref):
    """Capacity grid. n_estimators is chosen by early stopping on fold 1, then
    FIXED and checked on fold 2. Fold 1's score is therefore optimistic -- only
    fold 2 is an honest read, which is why the bar is applied to fold 2."""
    D1 = ctx.fold(V.FOLDS[0])
    D2 = ctx.fold(V.FOLDS[1])
    grid = [(lv, mc, cs) for lv in (7, 15, 31) for mc in (30, 100, 300) for cs in (0.4, 0.7)]
    out = []
    print(f'grid of {len(grid)} configs; early stopping on fold 1 (seed 42), '
          f'then fixed n_estimators checked on fold 2')
    for lv, mc, cs in grid:
        p = dict(BASE_CLF, num_leaves=lv, min_child_samples=mc, colsample_bytree=cs,
                 n_estimators=3000)
        m = lgb.LGBMClassifier(random_state=42, verbose=-1, **p)
        m.fit(D1['Xtr'], D1['ycode'], eval_set=[(D1['Xva'], D1['yva'].Opportunity.map(ctx.code))],
              callbacks=[lgb.early_stopping(50, verbose=False)])
        n = int(m.best_iteration_ or 400)
        f1a = eval_clf(ctx, T._wide(m.predict_proba(D1['Xva']), list(m.classes_), ctx.labels,
                                   D1['Xva'].index), D1['yva'])
        pf = dict(BASE_CLF, num_leaves=lv, min_child_samples=mc, colsample_bytree=cs)
        # single seed while ranking 18 configs; the winner is re-bagged below
        f1b = eval_clf(ctx, bag_clf(D2['Xtr'], D2['ycode'], D2['Xva'], ctx.labels,
                                    params=pf, n_est=n, seeds=(42,)), D2['yva'])
        out.append((lv, mc, cs, n, f1a, f1b))
        print(f'  leaves {lv:>2} mcs {mc:>3} cols {cs} -> n_est {n:>4} | '
              f'fold1 {f1a:.4f} (optimistic) | fold2 {f1b:.4f}')
    out.sort(key=lambda r: -r[5])                      # rank on the honest fold
    lv, mc, cs, n, f1a, f1b = out[0]
    print(f'\n  best on fold 2: leaves {lv}, min_child_samples {mc}, colsample {cs}, n_est {n}')
    rows = [ref]
    # re-bag the winner on BOTH folds with n_estimators fixed, so fold 1 is clean too
    pf = dict(BASE_CLF, num_leaves=lv, min_child_samples=mc, colsample_bytree=cs)
    f1s = []
    for v, D in ((V.FOLDS[0], D1), (V.FOLDS[1], D2)):
        f1s.append(eval_clf(ctx, bag_clf(D['Xtr'], D['ycode'], D['Xva'], ctx.labels,
                                         params=pf, n_est=n), D['yva']))
    rows.append(Row(f'a3 leaves{lv} mcs{mc} cols{cs} n{n}', f1s, ref=ref_dict(ref)))
    judge(rows, ref, 'f1', 0.005)
    report('A3  CAPACITY GRID, winner re-bagged on both folds', rows, ref, 'f1')
    save('a3', dict(grid=[dict(leaves=r[0], mcs=r[1], cols=r[2], n_est=r[3],
                              fold1_optimistic=r[4], fold2=r[5]) for r in out],
                    rows=[r.as_dict() for r in rows],
                    best=dict(num_leaves=lv, min_child_samples=mc, colsample_bytree=cs, n_estimators=n)))
    return rows


def block_a4(ctx, ref):
    rows = [ref]
    lgb_p, cat_p = {}, {}
    f1c = []
    for v in V.FOLDS:
        D = ctx.fold(v)
        t0 = time.time()
        cat_p[v] = bag_clf(D['Xtr'], D['ycode'], D['Xva'], ctx.labels,
                           params=dict(n_estimators=400, learning_rate=0.03, depth=6), cat=True)
        lgb_p[v] = bag_clf(D['Xtr'], D['ycode'], D['Xva'], ctx.labels)
        f1c.append(eval_clf(ctx, cat_p[v], D['yva']))
        print(f'  fold {v}: catboost bag in {time.time() - t0:.0f}s')
    rows.append(Row('a4 catboost', f1c, ref=ref_dict(ref)))
    gap = abs(np.mean(f1c) - np.mean(ref.f1))
    print(f'\n  |catboost - lgbm| mean F1 gap = {gap:.4f} '
          f'({"within" if gap <= 0.005 else "outside"} 0.005 -> '
          f'{"averaging" if gap <= 0.005 else "not averaging"})')
    if gap <= 0.005:
        f1m = [eval_clf(ctx, (lgb_p[v] + cat_p[v]) / 2, ctx.snaps[v][1]) for v in V.FOLDS]
        rows.append(Row('a4 lgbm+catboost average', f1m, ref=ref_dict(ref)))
    judge(rows, ref, 'f1', 0.005)
    report('A4  CATBOOST (classifier)', rows, ref, 'f1')
    save('a4', [r.as_dict() for r in rows])
    return rows


# --- B) regressions ----------------------------------------------------------

def _reg_rows(ctx, ref, variants, title, block):
    """variants: {name: (mode, half_life, kind, cat)} -> one Row per variant with
    both RMSEs; F1 is held at the reference so the score stays comparable."""
    rows = [ref]
    for name, (mode, hl, kind, cat) in variants.items():
        rf, rn = [], []
        for v in V.FOLDS:
            D = ctx.fold(v, mode=mode, half_life=hl)
            got = {}
            for t in TARGETS:
                if kind == 'hurdle':
                    got[t] = hurdle_reg(D['Xtr'], D['ytr'][t], D['Xva'], w=D['w'])
                elif kind == 'avg':
                    got[t] = (bag_reg(D['Xtr'], D['ytr'][t], D['Xva'], w=D['w'])
                              + bag_reg(D['Xtr'], D['ytr'][t], D['Xva'],
                                        params=dict(n_estimators=500, learning_rate=0.03, depth=6),
                                        w=D['w'], cat=True)) / 2
                else:
                    got[t] = bag_reg(D['Xtr'], D['ytr'][t], D['Xva'], w=D['w'], cat=cat)
            rf.append(rmse(got['CLV_fuel'], D['yva'].CLV_fuel))
            rn.append(rmse(got['CLV_nonfuel'], D['yva'].CLV_nonfuel))
        rows.append(Row(name, rf=rf, rn=rn, ref=ref_dict(ref)))
    for metric, label in (('rf', 'CLV_fuel'), ('rn', 'CLV_nonfuel')):
        judge(rows, ref, metric, 0.005)
        report(f'{title} -- {label}', rows, ref, metric)
    save(block, [r.as_dict() for r in rows])
    return rows


def block_b1(ctx, ref):
    return _reg_rows(ctx, ref, {'b1 hurdle (P(y>0) x E[y|y>0])': ('monthly', None, 'hurdle', False)},
                     'B1  HURDLE vs DIRECT', 'b1')


def block_b2(ctx, ref):
    vs = {f'b2 recency half-life {hl}mo': ('monthly', hl, 'direct', False) for hl in (3, 6, 12)}
    vs.update({f'b2 spacing {m}': (m, None, 'direct', False) for m in ('2month', 'quarterly')})
    return _reg_rows(ctx, ref, vs, 'B2  RECENCY / SPACING (regressions)', 'b2')


def block_b3(ctx, ref):
    return _reg_rows(ctx, ref, {'b3 catboost': ('monthly', None, 'direct', True),
                                'b3 lgbm+catboost average': ('monthly', None, 'avg', False)},
                     'B3  CATBOOST (regressions)', 'b3')


# --- C) adoption, one cheap test --------------------------------------------

def block_c(ctx, ref):
    """Binary 'any New category adoption' model. Report AUC, then the BEST
    achievable weighted-F1 gain from relabelling its top-k customers to their
    most likely adoption class -- an oracle sweep over k, so it is an upper
    bound, not an achievable score."""
    rows, aucs = [ref], []
    best_f1 = []
    for v in V.FOLDS:
        D = ctx.fold(v)
        isad = np.array([l.startswith('New category adoption') for l in ctx.labels])
        ytr_b = isad[D['ycode']].astype(int)
        truth = D['yva'].Opportunity.to_numpy()
        yva_b = np.array([l.startswith('New category adoption') for l in truth]).astype(int)
        p = bag_binary(D['Xtr'], ytr_b, D['Xva'])
        auc = roc_auc_score(yva_b, p)
        aucs.append(auc)
        # base prediction from the reference classifier, then relabel top-k
        proba = bag_clf(D['Xtr'], D['ycode'], D['Xva'], ctx.labels)
        base_pred = T.argmax_labels(proba, ctx.labels)
        f0 = V.f1(truth, base_pred)
        ad_cols = [c for c in ctx.labels if c.startswith('New category adoption')]
        most_likely = proba[ad_cols].to_numpy().argmax(axis=1)
        order = np.argsort(-p)
        best, bestk = f0, 0
        for k in range(0, min(1200, len(order)) + 1, 25):
            pred = base_pred.copy()
            idx = order[:k]
            pred[idx] = np.array(ad_cols)[most_likely[idx]]
            s = V.f1(truth, pred)
            if s > best:
                best, bestk = s, k
        print(f'  fold {v}: AUC {auc:.4f} | base F1 {f0:.4f} -> best oracle-k F1 '
              f'{best:.4f} at k={bestk} ({best - f0:+.4f})')
        best_f1.append(best)
    rows.append(Row(f'c adoption relabel (ORACLE k, AUC {np.mean(aucs):.3f})',
                    best_f1, ref=ref_dict(ref)))
    gains = [best_f1[i] - ref.f1[i] for i in (0, 1)]
    judge(rows, ref, 'f1', 0.003)
    report('C  ADOPTION BINARY + ORACLE RELABEL (upper bound)', rows, ref, 'f1')
    verdict = ('DROP the idea: oracle gain under 0.003 on at least one fold'
               if min(gains) < 0.003 else
               'worth pursuing: oracle gain >= 0.003 on both folds')
    print(f'\n  per-fold oracle gain: {gains[0]:+.4f}, {gains[1]:+.4f}  ->  {verdict}')
    save('c', dict(auc=aucs, rows=[r.as_dict() for r in rows], gains=gains, verdict=verdict))
    return rows


# --- D) more cutoffs ---------------------------------------------------------

def block_d(ctx, ref):
    """monthly vs fortnightly vs weekly training cutoffs, classifier and both
    regressions. Snapshots are cached, but the first weekly run builds ~65 of
    them, so this block is the slow one."""
    rows = [ref]
    for mode in ('fortnightly', 'weekly'):
        f1s, rf, rn = [], [], []
        for v in V.FOLDS:
            t0 = time.time()
            D = ctx.fold(v, mode=mode)
            p = bag_clf(D['Xtr'], D['ycode'], D['Xva'], ctx.labels)
            f1s.append(eval_clf(ctx, p, D['yva']))
            got = {t: bag_reg(D['Xtr'], D['ytr'][t], D['Xva']) for t in TARGETS}
            rf.append(rmse(got['CLV_fuel'], D['yva'].CLV_fuel))
            rn.append(rmse(got['CLV_nonfuel'], D['yva'].CLV_nonfuel))
            print(f'  {mode} fold {v}: {len(D["cuts"])} snapshots, {len(D["Xtr"])} rows, '
                  f'{time.time() - t0:.0f}s')
        rows.append(Row(f'd {mode}', f1s, rf, rn,
                        note=f'{len(schedule(V.FOLDS[1], mode))} snaps on fold 2'))
    for metric, label, bar in (('f1', 'Opportunity', 0.005),
                               ('rf', 'CLV_fuel', 0.005), ('rn', 'CLV_nonfuel', 0.005)):
        judge(rows, ref, metric, bar)
        report(f'D  CUTOFF DENSITY -- {label}', rows, ref, metric)
    save('d', [r.as_dict() for r in rows])
    return rows


# --- E) rule-component classifiers ------------------------------------------

EVENTS = ['ev_inactive', 'ev_fuel_growth', 'ev_nonfuel_growth', 'ev_adopt_any']


def rule_events(X, tgt, cats, cfg, y):
    """The four rule EVENTS as binary targets, from label_rules outputs and the
    future per-category matrices the rules compare. Targets only -- never fed
    as features at the same cutoff."""
    clean = {c: F._clean([c])[0] for c in cats}
    prev = pd.DataFrame({c: X[f'q1sp_{clean[c]}'] for c in cats}, index=X.index)
    ever = pd.DataFrame({c: X[f'ever_{clean[c]}'] for c in cats}, index=X.index)
    fsp = pd.DataFrame({c: tgt[f'fsp_{clean[c]}'] for c in cats}, index=X.index)
    fn = pd.DataFrame({c: tgt[f'fn_{clean[c]}'] for c in cats}, index=X.index)
    grow = ((ever > 0) & (fn > 0) & ((fsp - prev) >= cfg['growth_min_rand'])
            & (fsp > prev.clip(lower=0) * (1 + cfg['growth_min_relative'])))
    adopt = (ever == 0) & (fn > 0) & (fsp > 0)
    nonfuel = [c for c in cats if c != 'Fuel']
    return pd.DataFrame({
        'ev_inactive': y.Opportunity.eq('Inactivity').astype(int),
        'ev_fuel_growth': grow['Fuel'].astype(int),
        'ev_nonfuel_growth': grow[nonfuel].any(axis=1).astype(int),
        'ev_adopt_any': adopt.any(axis=1).astype(int)}, index=X.index)


def build_oof_events(ctx, tgts, seeds=(42,)):
    """Out-of-fold event probabilities built strictly forward in time: the
    features at cutoff c come from models trained only on snapshots whose
    outcome window ends on or before c. The earliest cutoffs have no usable
    training data, so they are NaN and LightGBM treats them as missing."""
    cuts = ctx.monthly
    ev = {c: rule_events(T.select(ctx.snaps[c][0], ('base',), ctx.bc), tgts[c],
                         ctx.cats, ctx.cfg, ctx.snaps[c][1]) for c in cuts}
    out = {}
    for c in cuts + [TEST_CUTOFF]:
        past = [k for k in cuts if pd.Timestamp(k) + pd.DateOffset(months=3) <= pd.Timestamp(c)]
        X = T.select(ctx.snaps[c][0], ('base',), ctx.bc)
        if not past:
            out[c] = pd.DataFrame(np.nan, index=X.index, columns=EVENTS)
            continue
        Xp = pd.concat([T.select(ctx.snaps[k][0], ('base',), ctx.bc) for k in past])
        yp = pd.concat([ev[k] for k in past])
        cols = {}
        for e in EVENTS:
            yb = yp[e].to_numpy()
            cols[e] = (np.full(len(X), float(yb[0])) if len(np.unique(yb)) < 2
                       else bag_binary(Xp, yb, X, seeds=seeds))
        out[c] = pd.DataFrame(cols, index=X.index)
        print(f'  oof events at {c}: trained on {len(past)} earlier snapshots '
              f'({len(Xp)} rows)', flush=True)
    return out, ev


def block_e(ctx, ref):
    tgts = S.load_category_targets(ctx.monthly, ctx.train, ctx.cfg, verbose=False)
    print('building out-of-fold rule-event features (forward in time, seed 42)')
    oof, ev = build_oof_events(ctx, tgts)
    rows = [ref]
    # E1: stack the four probabilities into the 17-class model
    f1s = []
    for v in V.FOLDS:
        D = ctx.fold(v, extra=oof)
        keep = [c for c in D['Xtr'].columns if c not in EVENTS or D['Xtr'][c].notna().any()]
        f1s.append(eval_clf(ctx, bag_clf(D['Xtr'][keep], D['ycode'], D['Xva'][keep],
                                         ctx.labels), D['yva']))
        print(f'  e1 fold {v}: event features non-null on '
              f'{D["Xtr"][EVENTS].notna().all(axis=1).mean():.0%} of training rows')
    rows.append(Row('e1 stacked event probabilities', f1s, ref=ref_dict(ref)))
    # E2: the four probabilities through a simple rule, no 17-class model
    f1s2 = []
    for v in V.FOLDS:
        D = ctx.fold(v)
        p = {}
        for e in EVENTS:
            yb = pd.concat([ev[k] for k in D['cuts']])[e].to_numpy()
            p[e] = (np.full(len(D['Xva']), float(yb[0])) if len(np.unique(yb)) < 2
                    else bag_binary(D['Xtr'], yb, D['Xva']))
        pred = np.where(
            p['ev_inactive'] >= 0.5, 'Inactivity',
            np.where(p['ev_adopt_any'] >= 0.5, 'New category adoption: Other',
                     np.where(p['ev_fuel_growth'] >= 0.5, 'Existing-category growth: Fuel',
                              np.where(p['ev_nonfuel_growth'] >= 0.5,
                                       'Existing-category growth: Other', 'Stable'))))
        f1s2.append(V.f1(D['yva'].Opportunity.to_numpy(), pred))
    rows.append(Row('e2 events through a simple rule', f1s2, ref=ref_dict(ref)))
    judge(rows, ref, 'f1', 0.005)
    report('E  RULE-COMPONENT CLASSIFIERS', rows, ref, 'f1')
    save('e', [r.as_dict() for r in rows])
    return rows


BLOCKS = {'ref': block_ref, 'a1': block_a1, 'a2': block_a2, 'a3': block_a3, 'a4': block_a4,
          'b1': block_b1, 'b2': block_b2, 'b3': block_b3, 'c': block_c, 'd': block_d,
          'e': block_e}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--block', nargs='+', required=True)
    ap.add_argument('--train', default='data/train.csv')
    a = ap.parse_args()
    ctx = Ctx(a.train)
    ref = None
    for b in a.block:
        t0 = time.time()
        print(f'\n########## block {b} ##########', flush=True)
        if b == 'ref':
            ref = BLOCKS[b](ctx)
        else:
            ref = ref or load_ref()
            BLOCKS[b](ctx, ref)
        print(f'\n[block {b} took {time.time() - t0:.0f}s]', flush=True)


if __name__ == '__main__':
    main()
