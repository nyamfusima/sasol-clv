"""Opportunity-label model v2: variants a-f, each scored on both validation folds.

The regressions are already level with 1st place; the whole gap is the label, so
this file only touches Opportunity. Every variant returns a (n x 17) probability
frame over cfg['opportunity_labels'] so variants can be blended and reweighted
the same way, and the predicted label is always the argmax.

    python src/train_label.py                 # run the full variant sweep
    python src/train_label.py --only base,growth
"""
import argparse, json
from pathlib import Path
import numpy as np, pandas as pd, lightgbm as lgb
import features as F
import snapshots as S
import validate as V

SEED = 42
TEST_CUTOFF = '2025-12-01'
ALL_BLOCKS = ('base', 'growth', 'recency', 'season')
PRED_DIR = Path('preds')


# --- feature-block column selection -----------------------------------------

def block_columns(cols):
    """Which block emitted each column. growth only emits g_*, season only sy_*;
    the rest of the non-base names come from the recency block."""
    base = [c for c in cols if not (c.startswith('g_') or c.startswith('sy_')
                                    or c.startswith(('d7_', 'd14_', 'd30_', 'd60_', 'since_',
                                                     'slope6_', 'last6_'))
                                    or c in ('gap_max', 'gap_last_vs_max', 'gap_last_vs_mean'))]
    return {'base': base,
            'growth': [c for c in cols if c.startswith('g_')],
            'season': [c for c in cols if c.startswith('sy_')],
            'recency': [c for c in cols if c not in base and not c.startswith(('g_', 'sy_'))]}


def select(X, blocks, bc):
    keep = [c for b in ALL_BLOCKS if b in blocks for c in bc[b]]
    return X[keep]


# --- models ------------------------------------------------------------------

def _clf(n=400, leaves=15, **kw):
    return lgb.LGBMClassifier(n_estimators=n, learning_rate=0.03, num_leaves=leaves,
                              min_child_samples=30, subsample=0.8, subsample_freq=1,
                              colsample_bytree=0.7, random_state=SEED, verbose=-1, **kw)


def _reg(n=300, leaves=31, **kw):
    return lgb.LGBMRegressor(n_estimators=n, learning_rate=0.05, num_leaves=leaves,
                             min_child_samples=30, subsample=0.8, subsample_freq=1,
                             colsample_bytree=0.7, random_state=SEED, verbose=-1, **kw)


def _wide(proba, classes, labels, index):
    """Spread a model's proba over all 17 labels (training folds can miss some)."""
    out = pd.DataFrame(0.0, index=index, columns=labels)
    out.loc[:, [labels[i] for i in classes]] = proba
    return out


def single(Xtr, ytr, Xva, labels, **kw):
    """The baseline shape: one 17-class classifier."""
    code = {l: i for i, l in enumerate(labels)}
    m = _clf(**kw).fit(Xtr, ytr.Opportunity.map(code))
    return _wide(m.predict_proba(Xva), m.classes_, labels, Xva.index)


def two_stage(Xtr, ytr, Xva, labels, **kw):
    """(d) Stage 1 active vs Inactivity, stage 2 the class among active customers.
    P(class) = P(active) * P(class | active); P(Inactivity) = P(inactive)."""
    code = {l: i for i, l in enumerate(labels)}
    inactive = labels.index('Inactivity')
    y = ytr.Opportunity.map(code)
    yv = y.to_numpy()
    s1 = _clf(**kw).fit(Xtr, (yv == inactive).astype(int))
    p_inact = s1.predict_proba(Xva)[:, list(s1.classes_).index(1)]
    act = yv != inactive
    s2 = _clf(**kw).fit(Xtr[act], yv[act])
    p_act = _wide(s2.predict_proba(Xva), s2.classes_, labels, Xva.index)
    out = p_act.mul(1 - p_inact, axis=0)
    out['Inactivity'] = p_inact
    return out


def rule_derived(Xtr, ytr, Xva, labels, cfg, tgt_tr, cats, temp=1.0):
    """(e) Predict next-quarter net spend per category, then run the published
    label rules over those predictions instead of classifying the label directly.

    Spend is modelled on a signed log scale (log1p of |spend| keeps the handful
    of return-dominated cells from dominating the fit), and 'at least one
    qualifying basket' is a separate per-category classifier."""
    clean = {c: F._clean([c])[0] for c in cats}
    prev = pd.DataFrame({c: Xva.get(f'q1sp_{clean[c]}', 0.0) for c in cats}, index=Xva.index)
    ever = pd.DataFrame({c: Xva.get(f'ever_{clean[c]}', 0.0) for c in cats}, index=Xva.index)
    fsp = pd.DataFrame(0.0, index=Xva.index, columns=cats)
    fn = pd.DataFrame(0.0, index=Xva.index, columns=cats)
    for c in cats:
        n = clean[c]
        y_sp, y_n = tgt_tr[f'fsp_{n}'], (tgt_tr[f'fn_{n}'] > 0).astype(int).to_numpy()
        s = np.sign(y_sp) * np.log1p(y_sp.abs())        # signed log keeps returns in scale
        h = _reg().fit(Xtr, s).predict(Xva)
        fsp[c] = np.sign(h) * np.expm1(np.abs(h))       # invert the transform
        if len(np.unique(y_n)) < 2:
            fn[c] = float(y_n[0])
        else:
            m = _clf(n=250).fit(Xtr, y_n)
            fn[c] = m.predict_proba(Xva)[:, list(m.classes_).index(1)]
    return rules_to_proba(prev, ever, fsp, fn, cats, cfg, labels, temp)


def rules_to_proba(prev, ever, fsp, fn, cats, cfg, labels, temp=1.0):
    """Apply rules 5-8 to predicted per-category spend. Returns a one-hot-ish
    frame so the result can be averaged with a classifier's probabilities."""
    mapping = cfg['category_mapping']
    buys = fn >= 0.5
    delta = fsp - prev
    grow = (ever > 0) & buys & (delta >= cfg['growth_min_rand'])
    grow &= fsp > prev.clip(lower=0) * (1 + cfg['growth_min_relative'])
    adopt = (ever == 0) & buys & (fsp > 0)
    av = fsp.where(adopt, 0).to_numpy()
    gv = delta.where(grow, 0).to_numpy()
    am, gm = av.max(axis=1), gv.max(axis=1)
    is_adopt = (am > 0) & (am >= gm)
    is_growth = (gm > 0) & ~is_adopt
    out = np.full(len(prev), 'Stable', dtype=object)
    for mask, values, kind in [(is_adopt, av, 'New category adoption'),
                               (is_growth, gv, 'Existing-category growth')]:
        cat = np.array(cats)[values.argmax(axis=1)]
        out[mask] = [kind + ': ' + mapping[c][kind] for c in cat[mask]]
    out[~buys.any(axis=1).to_numpy()] = 'Inactivity'
    p = pd.DataFrame(0.0, index=prev.index, columns=labels)
    for i, l in enumerate(out):
        p.iat[i, labels.index(l)] = 1.0
    return p


# --- (f) class-prior adjustment ----------------------------------------------

def fit_weights(proba, y_true, labels, rounds=4,
                grid=(0.4, 0.6, 0.8, 1.0, 1.25, 1.6, 2.0, 3.0, 5.0, 8.0)):
    """Coordinate ascent on per-class multipliers, maximising weighted F1."""
    w = np.ones(len(labels))
    best = V.f1(y_true, apply_weights(proba, w, labels))
    for _ in range(rounds):
        improved = False
        for j in range(len(labels)):
            cur = w[j]
            for g in grid:
                if g == cur:
                    continue
                w[j] = g
                s = V.f1(y_true, apply_weights(proba, w, labels))
                if s > best + 1e-6:
                    best, cur, improved = s, g, True
            w[j] = cur
        if not improved:
            break
    return w, best


def apply_weights(proba, w, labels):
    return np.array(labels)[(proba.to_numpy() * w).argmax(axis=1)]


def argmax_labels(proba, labels):
    return np.array(labels)[proba.to_numpy().argmax(axis=1)]


# --- stage 2: blend the rule-derived variant in, then fit class priors -------

def stage2(cfg, labels, cats, data, bc, blocks, model, out_tag, detail=False):
    """Takes the winning feature set, then (e) blending and (f) prior weights.

    The prior weights are fitted on one fold and judged only on the other, so a
    weight vector that merely memorises its own fold cannot be kept.
    """
    pc, pr, truth = {}, {}, {}
    fn = {'single': single, 'two_stage': two_stage}[model]
    for v in V.FOLDS:
        D = data[v]
        Xtr, Xva = select(D['Xtr'], blocks, bc), select(D['Xva'], blocks, bc)
        pc[v] = fn(Xtr, D['ytr'], Xva, labels)
        pr[v] = rule_derived(Xtr, D['ytr'], Xva, labels, cfg, D['Ttr'], cats)
        truth[v] = D['yva'].Opportunity.to_numpy()

    def sc(p):
        return {v: V.f1(truth[v], argmax_labels(p[v], labels)) for v in V.FOLDS}

    base_sc = sc(pc)
    rule_sc = sc(pr)
    print('\n--- (e) rule-derived vs classifier ---')
    print(f'  classifier : ' + ' | '.join(f'{v} {base_sc[v]:.4f}' for v in V.FOLDS)
          + f' | mean {np.mean(list(base_sc.values())):.4f}')
    print(f'  rule-derived: ' + ' | '.join(f'{v} {rule_sc[v]:.4f}' for v in V.FOLDS)
          + f' | mean {np.mean(list(rule_sc.values())):.4f}')
    best_w, best_mean = 0.0, np.mean(list(base_sc.values()))
    print('\n  blend  w*rule + (1-w)*clf:')
    for w in np.arange(0.1, 1.0, 0.1):
        b = {v: pr[v] * w + pc[v] * (1 - w) for v in V.FOLDS}
        s = sc(b)
        m = np.mean(list(s.values()))
        both = all(s[v] >= base_sc[v] for v in V.FOLDS)
        flag = 'both folds up' if both else ''
        print(f'    w={w:.1f}  ' + ' | '.join(f'{v} {s[v]:.4f}' for v in V.FOLDS)
              + f' | mean {m:.4f}  {flag}')
        if both and m > best_mean + 1e-6:
            best_w, best_mean = float(w), m
    blended = {v: pr[v] * best_w + pc[v] * (1 - best_w) for v in V.FOLDS} if best_w else pc
    print(f'  -> blend weight kept: {best_w:.1f} (mean F1 {best_mean:.4f})')

    print('\n--- (f) class-prior adjustment (fit on one fold, judged on the other) ---')
    bl_sc = sc(blended)
    weights, held_out = {}, {}
    for fit_on in V.FOLDS:
        other = [v for v in V.FOLDS if v != fit_on][0]
        w, in_f1 = fit_weights(blended[fit_on], truth[fit_on], labels)
        out_f1 = V.f1(truth[other], apply_weights(blended[other], w, labels))
        weights[fit_on] = w
        held_out[other] = out_f1
        print(f'  fit on {fit_on}: {bl_sc[fit_on]:.4f} -> {in_f1:.4f} (in-fold, ignore) | '
              f'held-out {other}: {bl_sc[other]:.4f} -> {out_f1:.4f} '
              f'({out_f1 - bl_sc[other]:+.4f})')
    keep_prior = all(held_out[v] > bl_sc[v] for v in V.FOLDS)
    wmean = np.mean([weights[v] for v in V.FOLDS], axis=0)
    # The only honest estimate of the prior adjustment is the cross-fold one:
    # each fold scored with weights fitted on the OTHER fold. Averaging the two
    # vectors and re-scoring both folds leaks each fold's weights into its own
    # score, so that number is reported separately and never used for GO/NO-GO.
    honest = {v: held_out[v] for v in V.FOLDS}
    honest_mean = float(np.mean(list(honest.values())))
    print(f'  -> prior adjustment {"KEPT" if keep_prior else "DROPPED"} '
          f'(needs a gain on both held-out folds)')
    print(f'  HELD-OUT mean F1 (weights from the other fold): {honest_mean:.4f} '
          f'({honest_mean - np.mean(list(bl_sc.values())):+.4f} vs no adjustment) <- use this')
    if keep_prior:
        final = {v: blended[v] * wmean for v in V.FOLDS}
        print('  averaged weights (only classes moved off 1.0):')
        for l, x in zip(labels, wmean):
            if abs(x - 1) > 1e-9:
                print(f'    {l:<42} {x:.3f}')
    else:
        final = blended
    fin_sc = sc(final)
    mean = float(np.mean(list(fin_sc.values())))
    print('\n--- final label model ---')
    for v in V.FOLDS:
        print(f'  fold {v}: {fin_sc[v]:.4f}')
    if keep_prior:
        print(f'  mean: {mean:.4f}  <- OPTIMISTIC: averaged weights include each '
              f'fold\'s own fit.\n        Honest held-out mean is {honest_mean:.4f}; '
              f'GO/NO-GO uses that.')
        mean = honest_mean
    else:
        print(f'  mean: {mean:.4f}')
    if detail:
        V.report('best variant', {v: (truth[v], argmax_labels(final[v], labels)) for v in V.FOLDS},
                 labels, show_detail=True)
    # out-of-fold probabilities for later ensembling
    PRED_DIR.mkdir(exist_ok=True)
    oof = pd.concat([final[v].assign(_fold=v) for v in V.FOLDS])
    oof.rename_axis('ID').reset_index().to_parquet(PRED_DIR / f'oof_proba_{out_tag}.parquet', index=False)
    print(f'  wrote preds/oof_proba_{out_tag}.parquet ({len(oof)} rows)')
    return dict(blocks=blocks, model=model, blend_w=best_w,
                weights=(wmean.tolist() if keep_prior else None), mean=mean, folds=fin_sc)


# --- sweep -------------------------------------------------------------------

def write_submission(cfg, labels, cats, spec, train_path, test_path, baseline_csv, out_csv):
    """Retrain the winning label model on every snapshot and replace only the
    Opportunity column of the baseline submission."""
    cutoffs = S.monthly(V.FIRST_CUTOFF, V.LAST_CUTOFF)
    snaps = S.load(cutoffs + [TEST_CUTOFF], train_path, cfg, ALL_BLOCKS,
                   label_cutoffs=cutoffs, verbose=False)
    tgts = S.load_category_targets(cutoffs, train_path, cfg, verbose=False)
    bc = block_columns(list(snaps[cutoffs[0]][0].columns))
    Xtr = pd.concat([snaps[c][0] for c in cutoffs])
    ytr = pd.concat([snaps[c][1] for c in cutoffs])
    Ttr = pd.concat([tgts[c] for c in cutoffs])
    Xte = snaps[TEST_CUTOFF][0]
    blocks = tuple(spec['blocks'])
    xa, xb = select(Xtr, blocks, bc), select(Xte, blocks, bc)
    fn = {'single': single, 'two_stage': two_stage}[spec['model']]
    p = fn(xa, ytr, xb, labels)
    print(f'  trained on {len(Xtr)} rows from {len(cutoffs)} snapshots')
    if spec['blend_w']:
        pr = rule_derived(xa, ytr, xb, labels, cfg, Ttr, cats)
        p = pr * spec['blend_w'] + p * (1 - spec['blend_w'])
    if spec['weights']:
        pred = apply_weights(p, np.array(spec['weights']), labels)
    else:
        pred = argmax_labels(p, labels)
    out = pd.Series(pred, index=Xte.index, name='Opportunity')

    base = pd.read_csv(baseline_csv, dtype={'ID': str})
    test_ids = pd.read_csv(test_path, dtype=str).ID
    sub = base.set_index('ID').reindex(test_ids)
    assert sub[['CLV_fuel', 'CLV_nonfuel']].notna().all().all(), 'baseline is missing test IDs'
    new = out.reindex(test_ids)
    assert new.notna().all(), 'some test customers have no label prediction'
    before = sub.Opportunity.to_numpy()
    sub['Opportunity'] = new.to_numpy()
    # check before writing, never after
    bad = set(sub.Opportunity) - set(labels)
    assert not bad, f'labels outside the config: {bad}'
    assert len(sub) == 5488, f'expected 5488 rows, got {len(sub)}'
    assert sub.index.equals(pd.Index(test_ids)), 'IDs do not match data/test.csv'
    assert list(sub.columns) == ['CLV_fuel', 'CLV_nonfuel', 'Opportunity'], sub.columns
    Path(out_csv).parent.mkdir(exist_ok=True)
    sub.rename_axis('ID').reset_index().to_csv(out_csv, index=False)
    print(f'\nWrote {out_csv}: {len(sub)} rows, IDs match data/test.csv, '
          f'all labels in config, CLV columns untouched')
    print(f'changed Opportunity for {(before != sub.Opportunity.to_numpy()).sum()} of {len(sub)} customers')
    print(sub.Opportunity.value_counts(normalize=True).round(4).to_string())
    return sub


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--train', default='data/train.csv')
    ap.add_argument('--test', default='data/test.csv')
    ap.add_argument('--config', default='src/label_config.json')
    ap.add_argument('--only', default='', help='comma-separated variant names')
    ap.add_argument('--detail', action='store_true', help='per-class F1 + confusion matrix')
    ap.add_argument('--mode', default='sweep', choices=['sweep', 'stage2', 'submit'])
    ap.add_argument('--blocks', default='base,growth,recency', help='stage2/submit feature blocks')
    ap.add_argument('--model', default='single', choices=['single', 'two_stage'])
    ap.add_argument('--tag', default='label_v2')
    ap.add_argument('--baseline', default='submissions/submission_baseline_v1.csv')
    ap.add_argument('--out', default='submissions/submission_label_v2.csv')
    ap.add_argument('--spec', default='preds/label_v2_spec.json')
    a = ap.parse_args()
    cfg = json.load(open(a.config))
    labels = cfg['opportunity_labels']
    cats = S.categories(cfg)
    want = set(x.strip() for x in a.only.split(',') if x.strip())

    if a.mode == 'submit':
        spec = json.loads(Path(a.spec).read_text())
        print(f'label model spec: {spec}')
        write_submission(cfg, labels, cats, spec, a.train, a.test, a.baseline, a.out)
        return

    cutoffs = S.monthly(V.FIRST_CUTOFF, V.LAST_CUTOFF)
    snaps = S.load(cutoffs, a.train, cfg, ALL_BLOCKS)
    tgts = S.load_category_targets(cutoffs, a.train, cfg)
    bc = block_columns(list(snaps[cutoffs[0]][0].columns))
    print('feature-block sizes: ' + ', '.join(f'{k} {len(v)}' for k, v in bc.items()))

    data = {}
    for v in V.FOLDS:
        tr = S.trainable_for(cutoffs, v)
        data[v] = dict(
            tr=tr,
            Xtr=pd.concat([snaps[c][0] for c in tr]),
            ytr=pd.concat([snaps[c][1] for c in tr]),
            # Ttr only: the future category spend at the VALIDATION cutoff is
            # unobservable at predict time and must never be loaded here.
            Ttr=pd.concat([tgts[c] for c in tr]),
            Xva=snaps[v][0], yva=snaps[v][1])

    if a.mode == 'stage2':
        blocks = tuple(x.strip() for x in a.blocks.split(',') if x.strip())
        spec = stage2(cfg, labels, cats, data, bc, blocks, a.model, a.tag, detail=a.detail)
        Path(a.spec).write_text(json.dumps(spec, indent=2))
        print(f'  wrote {a.spec}')
        return

    rows, probas = [], {}

    def run(name, fn, blocks, note=''):
        if want and name not in want:
            return None
        res, pr = {}, {}
        for v in V.FOLDS:
            D = data[v]
            Xtr, Xva = select(D['Xtr'], blocks, bc), select(D['Xva'], blocks, bc)
            p = fn(Xtr, D['ytr'], Xva, labels, D)
            pr[v] = p
            res[v] = (D['yva'].Opportunity.to_numpy(), argmax_labels(p, labels))
        mean, sc = V.report(name, res, labels, show_detail=a.detail)
        rows.append([name, sc[V.FOLDS[0]], sc[V.FOLDS[1]], mean, note])
        probas[name] = (pr, res)
        return mean

    # -- a) growth-rule features, b) recency, c) same season last year ---------
    base_mean = run('base', lambda Xtr, ytr, Xva, L_, D: single(Xtr, ytr, Xva, L_), ('base',))
    run('a_growth', lambda Xtr, ytr, Xva, L_, D: single(Xtr, ytr, Xva, L_), ('base', 'growth'))
    run('b_recency', lambda Xtr, ytr, Xva, L_, D: single(Xtr, ytr, Xva, L_), ('base', 'growth', 'recency'))
    run('c_season', lambda Xtr, ytr, Xva, L_, D: single(Xtr, ytr, Xva, L_), ALL_BLOCKS)
    # -- d) two-stage ----------------------------------------------------------
    run('d_two_stage', lambda Xtr, ytr, Xva, L_, D: two_stage(Xtr, ytr, Xva, L_),
        ('base', 'growth', 'recency'))
    # -- e) rule-derived -------------------------------------------------------
    run('e_rule', lambda Xtr, ytr, Xva, L_, D: rule_derived(
        Xtr, ytr, Xva, L_, cfg, D['Ttr'], cats), ('base', 'growth', 'recency'))

    print('\n' + '=' * 78)
    print(f'{"variant":<16}{"fold 2025-06":>14}{"fold 2025-09":>14}{"mean":>10}  note')
    for n, f1a, f1b, m, note in rows:
        print(f'{n:<16}{f1a:>14.4f}{f1b:>14.4f}{m:>10.4f}  {note}')
    print(f'\nbaseline mean for reference: {base_mean if base_mean else float("nan"):.4f}')


if __name__ == '__main__':
    main()
