"""EQ1: add the four previously unused TRAIN months to the Q46 classifier.

The old 2,093 fitting windows remain exact. Additional first difficulty
crossings come from June/December/January/February only. No evaluation month
is fitted. Reused A26 visible events are tuning diagnostics, not fresh tests.
"""
from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

import cache
from research_jobs import snapshot_sources
from research_probe import subset_days
from research_queue_candidate import digest
from research_queue_analog import build_library, encode_history, iou
from research_queue_dynamics import parameters
from research_transport_queue import context, transport_features
from research_first_crossing_queue import libraries, INVENTORY_SHA
from research_first_crossing_student import date_diverse_indices
from research_reopened_phase_experts import quarter_action
from research_reopened_decoder import queries
from research_month_transfer_audit import macro, bootstrap


Q46_SHA = '2816e1d3f3c233416327591c6659cd9e80a25f0054431f1b1d4ee23735b2ab46'
EXTRA_MONTHS = (6, 12, 1, 2)
ARMS = ('all400', 'all800', 'half400', 'half800')


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2))


def prepare(args):
    original = pd.read_csv(args.inventory)
    assert len(original) == 2093
    assert set(pd.to_datetime(original.date).dt.month) <= {7, 8, 9, 10, 11}
    assert not original.duplicated(['panel', 'date', 'slot']).any()
    assert set(original.panel) == set(cache.PANELS[:8])
    inventory_parts, panels = [], []
    for index, panel in enumerate(cache.PANELS[:8]):
        z = cache.load(panel, 'train')
        p = parameters(panel, z['links'])
        extra = subset_days(z, np.isin(pd.to_datetime(z['dates']).month, EXTRA_MONTHS))
        _, first, records = libraries(extra, p)
        available_extra = len(first['day'])
        records.to_csv(args.out/f'{panel}_extra_crossings.csv', index=False)
        selected = date_diverse_indices(first['day'], 240, 20261240+index)
        vf = p.free_speed_kmh.to_numpy()
        new = []
        for i in selected:
            d, s = int(first['day'][i]), int(first['slot'][i])
            future = extra['speed'][d, s+1:s+7]
            known = extra['elig'][d, s+1:s+7].astype(bool) & np.isfinite(future)
            new.append(dict(panel=panel, date=str(extra['dates'][d]), slot=s,
                            cells=int(known.sum()), positives=int(((future <= .6*vf) & known).sum()),
                            origin='extra_four_months'))
        old = original[original.panel == panel].copy()
        old['origin'] = 'original_q46'
        inventory = pd.concat([old, pd.DataFrame(new)], ignore_index=True).sort_values(['date', 'slot'])
        assert not inventory.duplicated(['panel', 'date', 'slot']).any()
        assert set(pd.to_datetime(inventory.date).dt.month) <= {6, 7, 8, 9, 10, 11, 12, 1, 2}
        inventory_parts.append(inventory)
        del extra, first, records
        gc.collect()

        dates = {str(d): i for i, d in enumerate(z['dates'])}
        ctx = context(p)
        features, targets, weight = [], [], []
        for row in inventory.itertuples():
            d, s = dates[str(row.date)], int(row.slot)
            arrays = tuple(z[c][d, s-12:s] for c in ('speed', 'flow', 'occ'))
            assert all(a.shape == (12, len(p)) for a in arrays)
            x = transport_features(arrays, p, ctx, s, pd.Timestamp(row.date).dayofweek >= 5)
            future = z['speed'][d, s+1:s+7]
            known = (z['elig'][d, s+1:s+7].astype(bool) & np.isfinite(future)).ravel()
            y = (future <= .6*vf).ravel()[known].astype(np.float32)
            assert len(y) == int(row.cells) and int(y.sum()) == int(row.positives)
            features.append(x.loc[known])
            targets.append(y)
            weight.append(np.full(len(y), 1/(len(y)*len(inventory)), np.float32))
        frame = pd.concat(features, ignore_index=True)
        frame['_target'] = np.concatenate(targets)
        frame['_weight'] = np.concatenate(weight)
        path = args.out/f'{panel}_fit.parquet'
        frame.to_parquet(path, index=False)
        panels.append(dict(panel=panel, rows=len(frame), old_windows=len(old), new_windows=len(new),
                           available_extra_crossings=available_extra,
                           month_counts=inventory.assign(month=pd.to_datetime(inventory.date).dt.month).groupby('month').size().to_dict(),
                           features_sha256=digest(path)))
        print(panel, 'EQ1 prepared', panels[-1], flush=True)
        del z, frame, features, targets, weight
        gc.collect()
    all_inv = pd.concat(inventory_parts, ignore_index=True).sort_values(['panel', 'date', 'slot'])
    assert set(pd.to_datetime(all_inv.date).dt.month) == {6, 7, 8, 9, 10, 11, 12, 1, 2}
    all_inv.to_csv(args.out/'training_inventory.csv', index=False)
    pd.testing.assert_frame_equal(
        all_inv[all_inv.origin == 'original_q46'][list(original)].sort_values(['panel', 'date', 'slot']).reset_index(drop=True),
        original.sort_values(['panel', 'date', 'slot']).reset_index(drop=True))
    result = dict(study='EQ1', old_windows=2093, windows=len(all_inv), panels=panels,
                  original_inventory_sha256=digest(args.inventory),
                  inventory_sha256=digest(args.out/'training_inventory.csv'),
                  fit_months=[6, 7, 8, 9, 10, 11, 12, 1, 2], eval_months_used=False)
    write_json(args.out/'prepare.json', result)


def fit(args):
    prep = json.loads((args.prepared/'prepare.json').read_text())
    assert prep['study'] == 'EQ1' and prep['old_windows'] == 2093
    assert digest(args.prepared/'training_inventory.csv') == prep['inventory_sha256']
    frames = []
    for item in prep['panels']:
        path = args.prepared/f"{item['panel']}_fit.parquet"
        assert digest(path) == item['features_sha256']
        frames.append(pd.read_parquet(path))
    x = pd.concat(frames, ignore_index=True)
    del frames
    y = x.pop('_target').to_numpy()
    w = np.array(x.pop('_weight').to_numpy(), dtype=np.float64, copy=True)
    w /= w.mean()
    gc.collect()
    settings = dict(objective='binary', learning_rate=.05, num_leaves=63,
        min_data_in_leaf=150, feature_fraction=.9, bagging_fraction=.8, bagging_freq=1,
        lambda_l2=2., num_threads=args.threads, verbose=-1, seed=20261002,
        deterministic=True, force_col_wise=True)
    print('EQ1 fit', len(x), 'rows', len(x.columns), 'features', flush=True)
    model = lgb.train(settings, lgb.Dataset(x, label=y, weight=w), 800,
                      callbacks=[lgb.log_evaluation(100)])
    for n in (400, 800):
        model.save_model(str(args.out/f'transport{n}.txt'), num_iteration=n)
        reread = lgb.Booster(model_file=str(args.out/f'transport{n}.txt'))
        assert reread.current_iteration() == n
        probe = x.iloc[::1009]
        np.testing.assert_array_equal(model.predict(probe, num_iteration=n, num_threads=args.threads),
                                      reread.predict(probe, num_threads=args.threads))
    pd.Series(model.feature_importance('gain'), index=x.columns).sort_values(ascending=False).to_csv(args.out/'feature_gain.csv')
    write_json(args.out/'fit.json', dict(study='EQ1', rows=len(x), columns=list(x),
        parameters=settings, rounds=[400, 800], windows=prep['windows'],
        prepare_sha256=digest(args.prepared/'prepare.json'),
        inventory_sha256=prep['inventory_sha256'],
        model_hashes={f'transport{n}.txt': digest(args.out/f'transport{n}.txt') for n in (400, 800)},
        labels='Eligible observed TRAIN speed <= .6 vf, same Q46 target',
        input='T-60 through T-5 only; static parameters; TRAIN models/libraries',
        eval_months_used=False, uploaded=False))


def evaluate(args):
    fit_report = json.loads((args.models/'fit.json').read_text())
    assert fit_report['study'] == 'EQ1' and not fit_report['eval_months_used']
    assert digest(args.control) == Q46_SHA
    assert digest(args.audit/'query_inventory.csv') == INVENTORY_SHA
    inv = pd.read_csv(args.audit/'query_inventory.csv')
    inv.to_csv(args.out/'query_inventory.csv', index=False)
    models = {'control': lgb.Booster(model_file=str(args.control))}
    for n in (400, 800):
        file = args.models/f'transport{n}.txt'
        assert digest(file) == fit_report['model_hashes'][file.name]
        models[f'all{n}'] = lgb.Booster(model_file=str(file))
    rows = []
    for panel in cache.PANELS[:8]:
        ztr = cache.load(panel, 'train')
        p = parameters(panel, ztr['links'])
        ctx = context(p)
        vf, cap = p.free_speed_kmh.to_numpy(), p.capacity_vph.to_numpy()
        bank = build_library(ztr, p, 'queue_ongoing')
        records, saved_prob, saved_pred, saved_truth, saved_known = [], [], [], [], []
        # 'confirmation' here only suppresses the historical TRAIN June group.
        # Both A26 date parities are deliberately reused tuning diagnostics.
        for split, z, _, target, wid, date, d, s, arrays in queries(panel, ztr, inv, 'confirmation'):
            assert pd.Timestamp(date).month in (3, 4)
            weekend = pd.Timestamp(date).dayofweek >= 5
            x = transport_features(arrays, p, ctx, s, weekend)
            probability = {k: m.predict(x[m.feature_name()], num_threads=args.threads).reshape(6, len(p))
                           for k, m in models.items()}
            probability['half400'] = .5*(probability['control']+probability['all400'])
            probability['half800'] = .5*(probability['control']+probability['all800'])
            qr, qf = encode_history(arrays[0][None], arrays[1][None], vf, cap)
            actions = {k: quarter_action(prob, bank, qr, qf, s, weekend) for k, prob in probability.items()}
            # No targets/availability are passed into features or action selection.
            future = z[target][d, s+1:s+7]
            known = z['elig'][d, s+1:s+7].astype(bool) & np.isfinite(future)
            truth = future <= .6*vf
            common = dict(panel=panel, family=panel.rsplit('_', 1)[0], split=split,
                          date=date, window_id=wid, slot=s)
            for name, pred in actions.items():
                rows.append(dict(**common, method=name, iou=iou(pred, truth, known),
                                 positives=int(pred.sum()), changed=int((pred != actions['control']).sum())))
            records.append(common)
            saved_prob.append(np.stack([probability[k] for k in ('control', *ARMS)]))
            saved_pred.append(np.stack([actions[k] for k in ('control', *ARMS)]))
            saved_truth.append(truth)
            saved_known.append(known)
        pd.DataFrame(records).to_csv(args.out/f'{panel}_queries.csv', index=False)
        np.savez_compressed(args.out/f'{panel}_diagnostics.npz', methods=np.array(['control', *ARMS]),
            probability=np.stack(saved_prob), prediction=np.stack(saved_pred),
            truth=np.stack(saved_truth), known=np.stack(saved_known))
        pd.DataFrame(rows).to_csv(args.out/'windows.csv', index=False)
        print(panel, 'EQ1 evaluated', len(records), flush=True)
        del bank, ztr
        gc.collect()
    frame = pd.DataFrame(rows)
    references = []
    for suffix in ('rq48_decoder', 'rq48_confirm'):
        ref = pd.read_csv(Path('work/research_20261003_reopened')/suffix/'windows.csv')
        references.append(ref[(ref.method == 'prior_quarter') & (ref.split != 'published_train')])
    reference = pd.concat(references, ignore_index=True)
    check = frame[frame.method == 'control'].merge(reference,
        on=['panel', 'split', 'window_id'], validate='one_to_one', suffixes=('', '_reference'))
    assert len(check) == int((frame.method == 'control').sum()) == len(inv)
    np.testing.assert_allclose(check.iou, check.iou_reference, atol=1e-12, rtol=0)
    summary = pd.DataFrame([dict(split=s, method=m, macro_iou=macro(g), windows=len(g))
                           for (s, m), g in frame.groupby(['split', 'method'])])
    summary.to_csv(args.out/'summary.csv', index=False)
    contrasts = [dict(split=s, **bootstrap(g, m, 'control', seed=20261244))
                 for s, g in frame.groupby('split') for m in ARMS]
    write_json(args.out/'contrasts.json', contrasts)
    decisions = []
    for m in ARMS:
        c = [r for r in contrasts if r['left'] == m]
        decisions.append(dict(method=m, both_months_positive=all(r['gain'] > 0 for r in c),
                              worst_gain=min(r['gain'] for r in c), mean_gain=float(np.mean([r['gain'] for r in c]))))
    passing = sorted((r for r in decisions if r['both_months_positive']), key=lambda r: (-r['mean_gain'], r['method']))
    write_json(args.out/'result.json', dict(study='EQ1', decisions=decisions,
        selected=passing[0]['method'] if passing else None,
        model_fit_sha256=digest(args.models/'fit.json'),
        control_exact=True, score_type='REUSED VISIBLE-EVENT TUNING PROXIES, NOT KAGGLE',
        limits='Neither date parity is fresh; June is now fitting data and is not scored as held out.',
        inference='Exactly T-60..T-5; TRAIN-only classifier/libraries; no other evaluation days', uploaded=False))
    print(summary.to_string(index=False), flush=True)
    print(json.dumps(decisions, indent=2), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--stage', choices=['prepare', 'fit', 'evaluate'], required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--prepared', type=Path)
    ap.add_argument('--models', type=Path)
    ap.add_argument('--inventory', type=Path, default=Path('work/research_20261003_deep/q46_crossing_student/training_inventory.csv'))
    ap.add_argument('--control', type=Path, default=Path('work/research_20261003_deep/q46_crossing_student/transport.txt'))
    ap.add_argument('--audit', type=Path, default=Path('work/research_20261003_deep/a26_month_ongoing'))
    ap.add_argument('--threads', type=int, default=6)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    sources = snapshot_sources(Path(__file__), args.out)
    write_json(args.out/'protocol.json', dict(study='EQ1', stage=args.stage, sources=sources,
        arms=ARMS, fit='Original Q46 2093 windows plus <=240 date-diverse extra-month crossings per panel',
        settings='Exact Q46 400-round control settings; additionally 800-round capacity arm',
        decoder='Unchanged ordinary TRAIN library and RQ48 quarter empirical utility',
        decision='Both visible proxy months must improve; rank by mean of the two gains',
        limits='All visible event diagnostics are reused and cannot establish private generalization or a hidden score',
        uploaded=False))
    {'prepare': prepare, 'fit': fit, 'evaluate': evaluate}[args.stage](args)


if __name__ == '__main__':
    main()
