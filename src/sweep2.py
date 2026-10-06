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
    # Join the extra columns PER CUTOFF, before concatenating. IDs repeat across
    # snapshots, so joining after the concat is a join on a non-unique index:
    # pandas expands it many-to-many and the row count no longer matches y.
    parts = []
    for c in cutoffs:
        x = T.select(snaps[c][0], blocks, bc)
        if extra is not None:
            x = x.join(extra[c])
        parts.append(x)
    X = pd.concat(parts)
    y = pd.concat([snaps[c][1] for c in cutoffs])
    cut = np.concatenate([[c] * len(snaps[c][0]) for c in cutoffs])
    assert len(X) == len(y) == len(cut), (len(X), len(y), len(cut))
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


def hurdle_reg(Xtr, y, Xva, params=None, w=None, seeds=SEEDS, cat=False):
    """(B1) P(y>0) * E[y | y>0], the second stage fitted on positive rows only.
    `cat` switches the magnitude stage to CatBoost; the gate stays LightGBM."""
    pos = (y.to_numpy() > 0)
    if pos.all() or not pos.any():
        return bag_reg(Xtr, y, Xva, params, w, seeds, cat=cat)
    p = bag_binary(Xtr, pos.astype(int), Xva, BASE_CLF, w, seeds)
    wp = None if w is None else w[pos]
    cp = dict(n_estimators=500, learning_rate=0.03, depth=6) if cat else params
    mag = bag_reg(Xtr[pos], y[pos], Xva, cp, wp, seeds, cat=cat)
    return np.clip(p * mag, 0, None)


def reg_predict(D, t, kind):
    """One regression prediction under a named recipe."""
    X, y, Xv, w = D['Xtr'], D['ytr'][t], D['Xva'], D['w']
    if kind == 'direct':
        return bag_reg(X, y, Xv, w=w)
    if kind == 'direct_cat':
        return bag_reg(X, y, Xv, params=dict(n_estimators=500, learning_rate=0.03, depth=6),
                       w=w, cat=True)
    if kind == 'hurdle':
        return hurdle_reg(X, y, Xv, w=w)
    if kind == 'hurdle_cat':
        return hurdle_reg(X, y, Xv, w=w, cat=True)
    if kind == 'hurdle_avg':
        return (hurdle_reg(X, y, Xv, w=w) + hurdle_reg(X, y, Xv, w=w, cat=True)) / 2
    raise ValueError(kind)


def reg_rmses(ctx, kind, mode='monthly', extra=None):
    rf, rn = [], []
    for v in V.FOLDS:
        D = ctx.fold(v, mode=mode, extra=extra)
        rf.append(rmse(reg_predict(D, 'CLV_fuel', kind), D['yva'].CLV_fuel))
        rn.append(rmse(reg_predict(D, 'CLV_nonfuel', kind), D['yva'].CLV_nonfuel))
    return rf, rn


def clf_f1s(ctx, mode='monthly', params=None, extra=None, n_est=None):
    out = []
    for v in V.FOLDS:
        D = ctx.fold(v, mode=mode, extra=extra)
        out.append(eval_clf(ctx, bag_clf(D['Xtr'], D['ycode'], D['Xva'], ctx.labels,
                                         params=params, n_est=n_est), D['yva']))
    return out


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

    def __init__(self, name, f1=None, rf=None, rn=None, ref=None, note='', touches=None):
        g = lambda v, k: v if v is not None else [ref[k][i] for i in (0, 1)]
        self.name, self.note = name, note
        self.f1, self.rf, self.rn = g(f1, 'f1'), g(rf, 'rf'), g(rn, 'rn')
        # which components this change actually alters; condition (3) guards
        # only these, so a regression-only change cannot be rejected for an F1
        # effect it does not have. Defaults to whatever was passed explicitly.
        self.touches = tuple(touches) if touches else tuple(
            k for k, v in (('f1', f1), ('rf', rf), ('rn', rn)) if v is not None) or ('f1', 'rf', 'rn')
        self.kept = ''

    def score(self, i):
        return V.combined(self.f1[i], self.rf[i], self.rn[i])

    @property
    def mean_score(self):
        return (self.score(0) + self.score(1)) / 2

    def as_dict(self):
        return dict(name=self.name, f1=self.f1, rf=self.rf, rn=self.rn,
                    score=[self.score(0), self.score(1)], mean_score=self.mean_score,
                    kept=self.kept, note=self.note, touches=list(self.touches))


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


SCORE_BAR = 0.0015
COMPONENT_SLACK = 0.001


def judge(rows, ref_row, metric=None, bar=None):
    """The single decision rule (replaces the original per-target -0.005 bars,
    which were set before the score formula was known):

      KEEP if (1) mean combined score improves by >= 0.0015 over the current
      best, (2) the combined score improves on BOTH folds, and (3) no component
      the change TOUCHES gets worse on either fold by more than 0.001.

    Condition (3) is scoped to `row.touches`, so a regression-only change is
    never rejected for an F1 effect it does not have. For a single-component
    change the 0.0015 score bar is equivalent to these component gains:
      F1 >= +0.00375 | rmse_fuel <= -0.00370 | rmse_nonfuel <= -0.00408

    `metric`/`bar` are accepted and ignored so per-target report() calls work.
    """
    for r in rows:
        if r is ref_row:
            r.kept = 'reference'
            continue
        gain = r.mean_score - ref_row.mean_score
        both = all(r.score(i) > ref_row.score(i) for i in (0, 1))
        worst, worst_name = 0.0, ''
        for i in (0, 1):
            for k, delta in (('f1', ref_row.f1[i] - r.f1[i]),
                             ('rf', r.rf[i] - ref_row.rf[i]),
                             ('rn', r.rn[i] - ref_row.rn[i])):
                if k in r.touches and delta > worst:
                    worst, worst_name = delta, k
        ok = gain >= SCORE_BAR and both and worst <= COMPONENT_SLACK
        r.kept = 'KEPT' if ok else 'dropped'
        why = []
        if gain < SCORE_BAR:
            why.append(f'gain {gain:+.5f}<{SCORE_BAR}')
        if not both:
            why.append('one fold down')
        if worst > COMPONENT_SLACK:
            why.append(f'{worst_name} -{worst:.4f}')
        r.note = ('' if ok else ' '.join(why))
    return rows


def component_bar(component):
    """The component change equivalent to the 0.0015 combined-score bar."""
    return {'f1': SCORE_BAR / V.W_F1,
            'rf': -SCORE_BAR * V.FUEL_NORM / V.W_FUEL,
            'rn': -SCORE_BAR * V.NONFUEL_NORM / V.W_NONFUEL}[component]


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
    # Two selection routes, both reported, because neither is free:
    #  - by fold 1: fold 1 is already spent on early stopping, so selecting there
    #    leaves fold 2 a genuine check. This is the honest number.
    #  - by fold 2: higher apparent score, but taking the max of 18 configs on
    #    fold 2 makes fold 2 a selection set rather than a check, and with a
    #    ~0.01 spread across the grid that optimism is worth several thousandths.
    by1 = max(out, key=lambda r: r[4])
    by2 = max(out, key=lambda r: r[5])
    print(f'\n  selected on fold 1 (honest): leaves {by1[0]}, mcs {by1[1]}, cols {by1[2]}, '
          f'n_est {by1[3]} -> fold 2 CHECK {by1[5]:.4f}')
    print(f'  selected on fold 2 (biased): leaves {by2[0]}, mcs {by2[1]}, cols {by2[2]}, '
          f'n_est {by2[3]} -> fold 2 {by2[5]:.4f} (max of {len(out)}, not a check)')
    print(f'  grid spread on fold 2: {min(r[5] for r in out):.4f} .. '
          f'{max(r[5] for r in out):.4f}')
    out.sort(key=lambda r: -r[5])
    lv, mc, cs, n, f1a, f1b = by2
    print(f'  re-bagging the fold-2 winner on both folds, n_estimators fixed at {n}')
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
                    best=dict(num_leaves=lv, min_child_samples=mc, colsample_bytree=cs, n_estimators=n),
                    selected_on_fold1=dict(leaves=by1[0], mcs=by1[1], cols=by1[2],
                                           n_est=by1[3], fold2_check=by1[5]),
                    grid_spread_fold2=[min(r[5] for r in out), max(r[5] for r in out)]))
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


def best_reg_row(ref):
    """The current best regression stack: B1's hurdle if it was kept, else the
    bagged reference. Regression changes are measured on top of this so we can
    see whether gains stack."""
    saved = json.loads(RESULTS.read_text()) if RESULTS.exists() else {}
    for d in saved.get('b1', []):
        if 'hurdle' in d['name']:
            r = Row(d['name'] + ' [current best]', d['f1'], d['rf'], d['rn'])
            r.kept = 'reference'
            return r, 'hurdle'
    return ref, 'direct'


def block_b3(ctx, ref):
    base, base_kind = best_reg_row(ref)
    rows = [base]
    for name, kind in (('b3 hurdle + catboost magnitude', 'hurdle_cat'),
                       ('b3 hurdle lgbm/catboost average', 'hurdle_avg')):
        rf, rn = reg_rmses(ctx, kind)
        rows.append(Row(name, rf=rf, rn=rn, ref=ref_dict(base)))
        print(f'  {name}: rmse_f {rf[0]:.4f}/{rf[1]:.4f}, rmse_nf {rn[0]:.4f}/{rn[1]:.4f}')
    judge(rows, base)
    for metric, label in (('score', 'combined score'), ('rf', 'CLV_fuel'), ('rn', 'CLV_nonfuel')):
        report(f'B3  CATBOOST ON TOP OF THE HURDLE -- {label}', rows, base, metric)
    save('b3', [r.as_dict() for r in rows])
    return rows


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
    """Cutoff density, decided PER TARGET. B2 showed the regressions want more
    snapshots while the classifier is indifferent, so the three targets are
    allowed to choose different spacings and each is judged only on its own
    component. monthly is the incumbent for all three: the bagged reference for
    the classifier, the kept hurdle for the regressions.
    """
    base, base_kind = best_reg_row(ref)
    saved = json.loads(RESULTS.read_text()) if RESULTS.exists() else {}
    for d in saved.get('b3', []):
        if d.get('kept') == 'KEPT':
            base_kind = 'hurdle_avg' if 'average' in d['name'] else 'hurdle_cat'
            print(f'  B3 kept "{d["name"]}" -> regression recipe here is {base_kind}')
    modes = ('fortnightly', 'weekly')
    f1_by, rf_by, rn_by = {}, {}, {}
    for mode in modes:
        t0 = time.time()
        f1_by[mode] = clf_f1s(ctx, mode=mode)
        rf_by[mode], rn_by[mode] = reg_rmses(ctx, base_kind, mode=mode)
        print(f'  {mode}: {len(schedule(V.FOLDS[1], mode))} snaps on fold 2, '
              f'{time.time() - t0:.0f}s', flush=True)

    out, winners = {}, {}
    specs = [('f1', 'Opportunity (vs bagged reference classifier)', ref,
              lambda m: dict(f1=f1_by[m])),
             ('rf', 'CLV_fuel (vs kept hurdle)', base, lambda m: dict(rf=rf_by[m])),
             ('rn', 'CLV_nonfuel (vs kept hurdle)', base, lambda m: dict(rn=rn_by[m]))]
    for comp, label, incumbent, pick in specs:
        inc = Row('monthly [incumbent]', f1=incumbent.f1, rf=incumbent.rf,
                  rn=incumbent.rn, touches=(comp,))
        inc.kept = 'reference'
        rows = [inc]
        for mode in modes:
            rows.append(Row(f'd {mode}', ref=ref_dict(incumbent), touches=(comp,),
                            note=f'{len(schedule(V.FOLDS[1], mode))} snaps', **pick(mode)))
        judge(rows, inc)
        report(f'D  CUTOFF DENSITY -- {label}', rows, inc, comp)
        print(f'  0.0015 score bar on this component = {component_bar(comp):+.5f}')
        kept = [r for r in rows[1:] if r.kept == 'KEPT']
        best = max(kept, key=lambda r: r.mean_score) if kept else inc
        winners[comp] = 'monthly' if best is inc else best.name.split()[1]
        print(f'  -> best spacing for this target: {winners[comp]}')
        out[comp] = [r.as_dict() for r in rows]
    print('')
    print(f'  per-target spacing chosen: {winners}')
    save('d', dict(per_target=out, winners=winners, reg_kind=base_kind))
    return out


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




# --- F) new information ------------------------------------------------------

F_CACHE = Path('preds/fblocks.parquet')
_RICH = {}


def rich_data(ctx):
    """load_rich memoised: block F calls this once per group and the Decimal
    rescaling in label_rules.prepare makes a reload expensive."""
    import features2 as F2
    if 'd' not in _RICH:
        t0 = time.time()
        _RICH['d'], _RICH['dall'] = F2.load_rich(ctx.train, ctx.cfg)
        print(f'  loaded rich transactions in {time.time() - t0:.0f}s '
              f'(memoised for the remaining groups)', flush=True)
    return _RICH['d'], _RICH['dall']


def f_snapshots(ctx, groups, cutoffs):
    """Extra F columns per cutoff, cached. Site rates are built forward in time:
    the table used at cutoff c aggregates outcomes only from snapshots whose
    outcome window ends on or before c."""
    import features2 as F2
    tag = '_'.join(g for g in F2.GROUPS if g in groups)
    key = f'{tag}'
    if F_CACHE.exists():
        t = pd.read_parquet(F_CACHE)
        have = t[t._key == key] if '_key' in t.columns else t.iloc[:0]
        if len(have):
            out = {}
            for c, g in have.groupby('_cutoff', observed=True):
                out[str(c)] = g.drop(columns=['_cutoff', '_key']).set_index('ID')
            if all(c in out for c in cutoffs):
                return {c: out[c] for c in cutoffs}
    d, dall = rich_data(ctx)
    mono = ctx.monthly
    # only F2 needs the home-site history, and it is the same for every group
    if 'f2' in groups and 'home' not in _RICH:
        t0 = time.time()
        _RICH['home'] = {c: F2.home_site(d[d._time < pd.Timestamp(c)],
                                         ctx.snaps[c][0].index) for c in mono}
        print(f'  built home-site history in {time.time() - t0:.0f}s (memoised)', flush=True)
    home = _RICH.get('home', {})
    labs = {c: ctx.snaps[c][1] for c in mono}
    out = {}
    for c in cutoffs:
        earlier = [k for k in mono
                   if pd.Timestamp(k) + pd.DateOffset(months=3) <= pd.Timestamp(c)]
        tbl = F2.site_rate_table(home, labs, earlier) if 'f2' in groups else None
        out[c] = F2.build(d, dall, c, ctx.snaps[c][0].index, groups, tbl)
        print(f'  F[{tag}] {c}: {out[c].shape[1]} cols'
              + (f', site table from {len(earlier)} earlier snapshots' if 'f2' in groups else ''),
              flush=True)
    prev = pd.read_parquet(F_CACHE) if F_CACHE.exists() else None
    new = pd.concat([out[c].assign(_cutoff=c, _key=key) for c in out]).rename_axis('ID').reset_index()
    F_CACHE.parent.mkdir(exist_ok=True)
    keep = pd.concat([prev[prev._key != key], new]) if prev is not None else new
    keep.to_parquet(F_CACHE, index=False)
    return out


def _f_eval(ctx, ref, name, groups, mode='quarterly'):
    extra = f_snapshots(ctx, groups, sorted(set(
        schedule(V.FOLDS[0], mode) + schedule(V.FOLDS[1], mode) + list(V.FOLDS))))
    f1s, rf, rn = [], [], []
    for v in V.FOLDS:
        D = ctx.fold(v, mode=mode, extra=extra)
        p = bag_clf(D['Xtr'], D['ycode'], D['Xva'], ctx.labels)
        f1s.append(eval_clf(ctx, p, D['yva']))
        got = {t: bag_reg(D['Xtr'], D['ytr'][t], D['Xva']) for t in TARGETS}
        rf.append(rmse(got['CLV_fuel'], D['yva'].CLV_fuel))
        rn.append(rmse(got['CLV_nonfuel'], D['yva'].CLV_nonfuel))
        print(f'  {name} fold {v}: {D["Xtr"].shape[1]} features, {len(D["Xtr"])} rows')
    return Row(name, f1s, rf, rn)


def block_fref(ctx, ref):
    """Quarterly reference on all three targets. Block F runs quarterly for
    speed, so its deltas must be measured against quarterly, not monthly --
    otherwise the spacing effect contaminates every F result. A2 only checked
    spacing on the classifier, never the regressions."""
    f1s, rf, rn = [], [], []
    for v in V.FOLDS:
        D = ctx.fold(v, mode='quarterly')
        f1s.append(eval_clf(ctx, bag_clf(D['Xtr'], D['ycode'], D['Xva'], ctx.labels), D['yva']))
        got = {t: bag_reg(D['Xtr'], D['ytr'][t], D['Xva']) for t in TARGETS}
        rf.append(rmse(got['CLV_fuel'], D['yva'].CLV_fuel))
        rn.append(rmse(got['CLV_nonfuel'], D['yva'].CLV_nonfuel))
        print(f'  fold {v}: {len(D["cuts"])} quarterly snapshots, {len(D["Xtr"])} rows')
    q = Row('reference quarterly (block F base)', f1s, rf, rn)
    q.kept = 'reference'
    report('F0  QUARTERLY REFERENCE vs MONTHLY', [ref, q])
    save('fref', [q.as_dict()])
    return q


def block_f(ctx, ref):
    import features2 as F2
    saved = json.loads(RESULTS.read_text()) if RESULTS.exists() else {}
    qref = None
    if 'fref' in saved:
        d = saved['fref'][0]
        qref = Row(d['name'], d['f1'], d['rf'], d['rn']); qref.kept = 'reference'
    else:
        qref = block_fref(ctx, ref)
    rows = [qref]
    singles = {}
    for g in F2.GROUPS:
        r = _f_eval(ctx, qref, f'{g} {FG_NAMES[g]}', (g,))
        singles[g] = r
        rows.append(r)
    # F6: everything that individually improved the combined score
    good = [g for g in F2.GROUPS if singles[g].mean_score > qref.mean_score]
    print(f'\n  groups that individually improved the score: {good or "none"}')
    if good:
        rows.append(_f_eval(ctx, qref, f'f6 combined ({"+".join(good)})', tuple(good)))
    for metric, label, bar in (('score', 'combined score', 0.0),
                               ('f1', 'Opportunity', 0.005),
                               ('rf', 'CLV_fuel', 0.005), ('rn', 'CLV_nonfuel', 0.005)):
        judge(rows, qref, metric, bar)
        report(f'F  NEW INFORMATION -- {label}', rows, qref, metric)
    save('f', [r.as_dict() for r in rows])
    return rows


FG_NAMES = {'f1': 'customer id', 'f2': 'sites', 'f3': 'fuel type/price',
            'f4': 'timing', 'f5': 'vouchers/discounts'}


def write_probes(ctx):
    """Three constant-label diagnostic files with the bagged reference's
    regressions. Written to submissions/ but NOT submitted."""
    cuts = ctx.monthly
    D = dict(Xtr=pd.concat([T.select(ctx.snaps[c][0], ('base',), ctx.bc) for c in cuts]),
             ytr=pd.concat([ctx.snaps[c][1] for c in cuts]))
    Xte = T.select(ctx.snaps[TEST_CUTOFF][0], ('base',), ctx.bc)
    preds = {t: bag_reg(D['Xtr'], D['ytr'][t], Xte) for t in TARGETS}
    test_ids = pd.read_csv('data/test.csv', dtype=str).ID
    base = pd.DataFrame({'CLV_fuel': preds['CLV_fuel'],
                         'CLV_nonfuel': preds['CLV_nonfuel']}, index=Xte.index)
    base = base.reindex(test_ids)
    assert base.notna().all().all(), 'probe regressions missing test IDs'
    for fname, lab in (('probe_stable.csv', 'Stable'),
                       ('probe_inactivity.csv', 'Inactivity'),
                       ('probe_fuelgrowth.csv', 'Existing-category growth: Fuel')):
        assert lab in ctx.labels, lab
        sub = base.copy()
        sub['Opportunity'] = lab
        assert len(sub) == 5488 and sub.index.equals(pd.Index(test_ids))
        Path('submissions').mkdir(exist_ok=True)
        sub.rename_axis('ID').reset_index().to_csv(f'submissions/{fname}', index=False)
        print(f'  wrote submissions/{fname}: {len(sub)} rows, all Opportunity="{lab}"')
    print('  (diagnostic only -- not submitted)')


BLOCKS['fref'] = block_fref
BLOCKS['f'] = block_f
BLOCKS['probes'] = lambda ctx, ref: write_probes(ctx)


# --- submission v2 from the kept stack ---------------------------------------

def kept_stack():
    """What sweep 2 actually kept: per-target spacing from D, regression recipe
    from B1/B3, classifier from C/E. Defaults to the bagged reference +
    monthly + hurdle if a block has not run."""
    saved = json.loads(RESULTS.read_text()) if RESULTS.exists() else {}
    spacing = {'f1': 'monthly', 'rf': 'monthly', 'rn': 'monthly'}
    spacing.update((saved.get('d') or {}).get('winners', {}) or {})
    reg_kind = 'direct'
    for d in saved.get('b1', []):
        if 'hurdle' in d['name'] and d.get('kept') == 'KEPT':
            reg_kind = 'hurdle'
    for d in saved.get('b3', []):
        if d.get('kept') == 'KEPT':
            reg_kind = 'hurdle_avg' if 'average' in d['name'] else 'hurdle_cat'
    clf = 'reference'
    for blk in ('c', 'e'):
        rows = saved.get(blk)
        rows = rows.get('rows', []) if isinstance(rows, dict) else (rows or [])
        for d in rows:
            if d.get('kept') == 'KEPT':
                clf = d['name']
    return dict(spacing=spacing, reg_kind=reg_kind, classifier=clf)


def stack_validation_score(stack):
    """Composite of the per-target validation measurements actually taken: F1
    from the classifier's winning spacing, each RMSE from its own winner."""
    saved = json.loads(RESULTS.read_text())
    ref = saved['ref'][0]
    got = {'f1': ref['f1'], 'rf': ref['rf'], 'rn': ref['rn']}
    for d in saved.get('b1', []):
        if 'hurdle' in d['name'] and d.get('kept') == 'KEPT':
            got['rf'], got['rn'] = d['rf'], d['rn']
    per = (saved.get('d') or {}).get('per_target', {})
    for comp in ('f1', 'rf', 'rn'):
        want = stack['spacing'][comp]
        if want == 'monthly':
            continue
        for d in per.get(comp, []):
            if d['name'] == f'd {want}':
                got[comp] = d[comp]
    r = Row('submission_v2 stack', got['f1'], got['rf'], got['rn'])
    return r


def build_v2(ctx, out='submissions/submission_v2.csv'):
    stack = kept_stack()
    print(f'kept stack: {json.dumps(stack)}')
    assert stack['classifier'] == 'reference', \
        f'classifier variant {stack["classifier"]} passed; wire it in before building'
    Xte = T.select(ctx.snaps[TEST_CUTOFF][0], ('base',), ctx.bc)

    # classifier at its own winning spacing
    cuts = schedule(TEST_CUTOFF, stack['spacing']['f1'])
    ctx.pool(cuts)
    Xtr, ytr, _ = assemble(ctx.snaps, cuts, ('base',), ctx.bc)
    print(f'  classifier: {stack["spacing"]["f1"]}, {len(cuts)} snapshots, {len(Xtr)} rows')
    proba = bag_clf(Xtr, ytr.Opportunity.map(ctx.code).to_numpy(), Xte, ctx.labels)
    opp = T.argmax_labels(proba, ctx.labels)

    # each regression at its own winning spacing
    preds = {}
    for comp, tgt in (('rf', 'CLV_fuel'), ('rn', 'CLV_nonfuel')):
        cuts = schedule(TEST_CUTOFF, stack['spacing'][comp])
        ctx.pool(cuts)
        X, y, _ = assemble(ctx.snaps, cuts, ('base',), ctx.bc)
        print(f'  {tgt}: {stack["spacing"][comp]}, {len(cuts)} snapshots, {len(X)} rows, '
              f'recipe {stack["reg_kind"]}')
        preds[tgt] = reg_predict(dict(Xtr=X, ytr=y, Xva=Xte, w=None), tgt, stack['reg_kind'])

    test_ids = pd.read_csv('data/test.csv', dtype=str).ID
    sub = pd.DataFrame({'CLV_fuel': preds['CLV_fuel'], 'CLV_nonfuel': preds['CLV_nonfuel'],
                        'Opportunity': opp}, index=Xte.index).reindex(test_ids)
    # validate before writing, never after
    assert len(sub) == 5488, f'expected 5488 rows, got {len(sub)}'
    assert sub.index.equals(pd.Index(test_ids)), 'IDs do not match data/test.csv'
    assert sub.notna().all().all(), 'missing predictions'
    bad = set(sub.Opportunity) - set(ctx.labels)
    assert not bad, f'labels outside the config: {bad}'
    neg = (sub[['CLV_fuel', 'CLV_nonfuel']] < 0).sum().sum()
    assert neg == 0, f'{neg} negative CLV values'
    assert list(sub.columns) == ['CLV_fuel', 'CLV_nonfuel', 'Opportunity'], sub.columns
    Path(out).parent.mkdir(exist_ok=True)
    sub.rename_axis('ID').reset_index().to_csv(out, index=False)

    r = stack_validation_score(stack)
    print(f'\nWrote {out}: {len(sub)} rows | IDs match data/test.csv | '
          f'{sub.Opportunity.nunique()} distinct labels, all in config | '
          f'CLV min {sub[["CLV_fuel", "CLV_nonfuel"]].min().min():.4f} (>= 0)')
    print('\nvalidation score of this stack vs the references:')
    print(f'  single-seed baseline          0.28164')
    print(f'  5-seed bagged reference       0.28342')
    print(f'  bagged + hurdle (B1)          0.28553')
    print(f'  submission_v2 stack           {r.mean_score:.5f}   '
          f'(folds {r.score(0):.5f} / {r.score(1):.5f})')
    print(f'  components: F1 {r.f1[0]:.4f}/{r.f1[1]:.4f} | '
          f'rmse_f {r.rf[0]:.4f}/{r.rf[1]:.4f} | rmse_nf {r.rn[0]:.4f}/{r.rn[1]:.4f}')
    print(f'\n  label mix:')
    print(sub.Opportunity.value_counts(normalize=True).round(4).head(8).to_string())
    save('v2', dict(stack=stack, score=r.as_dict()))
    return sub


BLOCKS['v2'] = lambda ctx, ref: build_v2(ctx)


# --- re-judge saved results under the current rule ---------------------------

# which components each block's variants actually change, and what they are
# measured against ('ref' = bagged reference, 'b1' = the kept hurdle)
BLOCK_SCOPE = {'a1': (('f1',), 'ref'), 'a2': (('f1',), 'ref'), 'a3': (('f1',), 'ref'),
               'a4': (('f1',), 'ref'), 'b1': (('rf', 'rn'), 'ref'),
               'b2': (('rf', 'rn'), 'ref'), 'b3': (('rf', 'rn'), 'b1'),
               'c': (('f1',), 'ref'), 'e': (('f1',), 'ref')}


def rejudge():
    """Re-apply the current rule to every saved block and persist the verdicts.

    Needed because blocks run before the rule change stored verdicts from the
    old per-target bars; kept_stack() reads those verdicts, so a stale 'dropped'
    on the hurdle would silently drop it from the submission.
    """
    saved = json.loads(RESULTS.read_text())
    def mk(d, touches=None):
        r = Row(d['name'], d['f1'], d['rf'], d['rn'], touches=touches or d.get('touches'))
        return r
    ref = mk(saved['ref'][0], ('f1', 'rf', 'rn'))
    bases = {'ref': ref}
    for d in saved.get('b1', []):
        if 'hurdle' in d['name']:
            bases['b1'] = mk(d, ('rf', 'rn'))
    changed = []
    for blk, (touches, base_key) in BLOCK_SCOPE.items():
        rows = saved.get(blk)
        if rows is None:
            continue
        is_dict = isinstance(rows, dict)
        lst = rows.get('rows', []) if is_dict else rows
        base = bases.get(base_key, ref)
        built = [mk(d, touches) for d in lst]
        variants = [r for r in built if r.name != base.name and 'current best' not in r.name]
        judge([base] + variants, base)
        for d, r in zip(lst, built):
            if r in variants:
                was = d.get('kept', '')
                d['kept'], d['note'] = r.kept, r.note
                d['touches'] = list(r.touches)
                if was != r.kept:
                    changed.append(f'{blk}: "{d["name"]}" {was or "?"} -> {r.kept}')
            else:
                d['kept'] = 'reference'
        saved[blk] = rows
    RESULTS.write_text(json.dumps(saved, indent=2))
    print('re-judged under the current rule:')
    for c in changed:
        print(f'  {c}')
    if not changed:
        print('  (no verdict changed)')
    return saved


BLOCKS['rejudge'] = lambda ctx, ref: rejudge()


if __name__ == '__main__':
    main()
