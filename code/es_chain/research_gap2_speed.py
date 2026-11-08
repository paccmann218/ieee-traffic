"""Recenter ES2 speeds on the next short-gap shell, gap exactly 2.

February labels must show the same-day interpolant is unbiased there.
The link offset is read from ES2 on March and April without labels.
Flow, queues, demand, and every other speed stay copied. ES2 is not modified.
A file is written only if the gate passes.
"""
from __future__ import annotations

import argparse
import json
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

import cache
from research_combine_candidates import archive_csv
from research_queue_candidate import digest
from t1_base import interp_day
from t1_features import prev_next


ES2_SHA = "98e9ee0bbdb9d42a6a4bed98e99cec5f7e2ffb5cbd871fe89a10f6364b304e4a"
COLUMNS = ["submission_id", "task", "speed_kmh", "flow_vph", "queue_pred", "path_flow"]
MIN_N = 100


def observed_speed(z, day):
    return np.where(
        z["elig"][day].astype(bool) & np.isfinite(z["mspeed"][day]) & ~z["target"][day].astype(bool),
        z["mspeed"][day], np.nan)


def shell(observed):
    _, gap_prev, _, gap_next = prev_next(observed)
    return (np.maximum(gap_prev, gap_next) == 2) & (gap_prev < 90) & (gap_next < 90), gap_prev, gap_next


def february(panel):
    z = cache.load(panel, "train")
    months = np.array([str(d)[:7] for d in z["dates"]])
    months_all, links_all, err_all = [], [], []
    for day, month in enumerate(months):
        target = z["target"][day].astype(bool) & np.isfinite(z["speed"][day])
        if not target.any():
            continue
        observed = observed_speed(z, day)
        pred = interp_day(observed)
        take_shell, _, _ = shell(observed)
        take = target & take_shell & np.isfinite(pred)
        if not take.any():
            continue
        link = np.broadcast_to(np.arange(take.shape[1]), take.shape)[take]
        months_all.append(np.full(link.shape, month))
        links_all.append(link.astype(np.int16))
        err_all.append((pred - z["speed"][day])[take].astype(np.float64))
    if not links_all:
        return dict(n=0)
    month = np.concatenate(months_all)
    link = np.concatenate(links_all)
    err = np.concatenate(err_all)
    n_links = int(link.max()) + 1
    train = np.array([item < "2031-02" for item in month])
    test = np.array([item == "2031-02" for item in month])
    seen = np.bincount(link[train], minlength=n_links)
    summed = np.bincount(link[train], weights=err[train], minlength=n_links)
    bias = np.zeros(n_links)
    ok = seen >= 20
    bias[ok] = summed[ok] / seen[ok]
    raw = err[test]
    if raw.size == 0:
        return dict(n=0)
    fixed = raw - bias[link[test]]
    def rmse(values):
        return float(np.sqrt(np.mean(values * values)))
    return dict(
        n=int(test.sum()), links=int(ok.sum()),
        mean_abs_bias=float(np.mean(np.abs(bias[ok]))) if ok.any() else None,
        rmse=rmse(raw), rmse_corrected=rmse(fixed),
    )


def shifts_for(z, part, speeds):
    dates = [str(d)[:10] for d in z["dates"]]
    loc = {d: i for i, d in enumerate(dates)}
    link_loc = {link: i for i, link in enumerate(z["links"])}
    day = part.timestamp.str[:10].map(loc)
    link = part.link_id.map(link_loc)
    if day.isna().any() or link.isna().any():
        raise SystemExit("State row missing from the panel cache")
    day = day.to_numpy(int)
    link = link.to_numpy(int)
    slot = part.timestamp.str[11:13].astype(int).to_numpy() * 12 + part.timestamp.str[14:16].astype(int).to_numpy() // 5
    sub = speeds.to_numpy(np.float64)
    order = np.argsort(day, kind="mergesort")
    inv = np.empty_like(order)
    inv[order] = np.arange(len(order))
    day, slot, link, sub = day[order], slot[order], link[order], sub[order]
    pred = np.empty(len(part))
    on_shell = np.zeros(len(part), bool)
    start = 0
    while start < len(day):
        stop = start + 1
        while stop < len(day) and day[stop] == day[start]:
            stop += 1
        d = int(day[start])
        observed = observed_speed(z, d)
        iv = interp_day(observed)
        take_shell, _, _ = shell(observed)
        choose = slice(start, stop)
        pred[choose] = iv[slot[choose], link[choose]]
        on_shell[choose] = take_shell[slot[choose], link[choose]]
        start = stop
    pred, on_shell, sub, link = pred[inv], on_shell[inv], sub[inv], link[inv]
    take = on_shell & np.isfinite(pred) & np.isfinite(sub)
    n_links = len(z["links"])
    counts = np.bincount(link[take], minlength=n_links)
    summed = np.bincount(link[take], weights=(sub - pred)[take], minlength=n_links)
    shift = np.zeros(n_links)
    ok = counts >= MIN_N
    shift[ok] = summed[ok] / counts[ok]
    updated = sub.copy()
    apply = take & ok[link]
    updated[apply] = np.clip(sub[apply] - shift[link[apply]], 5.0, 140.0)
    return updated, apply, shift, counts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--release-root", type=Path, default=cache.REL)
    args = parser.parse_args()
    if (args.out / "submission_ES3.zip").exists():
        raise SystemExit("ES3 archive already exists")
    args.out.mkdir(parents=True, exist_ok=True)
    cache.REL = args.release_root
    february_rows = []
    for panel in cache.PANELS:
        row = february(panel)
        february_rows.append(dict(panel=panel, **row))
        print("FEB", panel, json.dumps(row), flush=True)
    usable = [row for row in february_rows if row.get("n")]
    mean_abs_bias = float(np.mean([row["mean_abs_bias"] for row in usable]))
    rmse = float(np.mean([row["rmse"] for row in usable]))
    rmse_corrected = float(np.mean([row["rmse_corrected"] for row in usable]))
    february_ok = mean_abs_bias < 0.15 and rmse_corrected <= rmse + 0.02
    print("FEBRUARY", mean_abs_bias, rmse, rmse_corrected, february_ok, flush=True)
    if not february_ok:
        report = dict(gate_passed=False, mean_abs_february_bias=mean_abs_bias,
                      mean_february_rmse=rmse, mean_february_rmse_corrected=rmse_corrected,
                      february=february_rows, note="February gate failed. No file written. Not a Kaggle score.")
        (args.out / "gap2.json").write_text(json.dumps(report))
        print("GATE failed", flush=True)
        return
    assert digest(args.base) == ES2_SHA
    key = pd.read_csv(args.release_root / "submission_key.csv", dtype=str,
                      usecols=["submission_id", "task", "split", "panel", "timestamp", "link_id"])
    with zipfile.ZipFile(args.base) as archive:
        base = pd.read_csv(archive.open(archive.namelist()[0]), dtype=str)
    assert list(base.columns) == COLUMNS and len(base) == len(key) == 6980503
    assert (base.submission_id.to_numpy() == key.submission_id.to_numpy()).all()
    original = base.copy()
    state = key.task.eq("state")
    speeds = pd.to_numeric(base.loc[state, "speed_kmh"])
    key_state = key.loc[state]
    panels, deltas, offsets = [], [], []
    for panel in cache.PANELS:
        for split in ("validation", "private"):
            z = cache.load(panel, split)
            part = key_state[(key_state.panel == panel) & (key_state.split == split)]
            updated, apply, shift, counts = shifts_for(z, part, speeds.loc[part.index])
            old = speeds.loc[part.index].to_numpy()
            deltas.append(updated[apply] - old[apply])
            enough = counts >= MIN_N
            offset = float(np.sqrt(np.mean(np.square(shift[enough])))) if enough.any() else 0.0
            offsets.append(offset)
            base.loc[part.index[apply], "speed_kmh"] = [f"{value:.8f}" for value in updated[apply]]
            panels.append(dict(panel=panel, split=split, n_gap2=int(apply.sum()),
                               links=int(enough.sum()), rms_shift=offset))
            print(panel, split, panels[-1], flush=True)
    delta = np.concatenate(deltas) if deltas else np.zeros(1)
    changed = int(np.sum(np.abs(delta) > 1e-8))
    summary = dict(
        changed_cells=changed,
        mean_abs_delta=float(np.mean(np.abs(delta))) if changed else 0.0,
        rms_delta=float(np.sqrt(np.mean(delta ** 2))) if changed else 0.0,
        max_abs=float(np.max(np.abs(delta))) if changed else 0.0,
        mean_march_april_rms=float(np.mean(offsets)),
    )
    passed = summary["mean_march_april_rms"] >= 0.40 and 20_000 <= changed <= 2_500_000 and summary["max_abs"] <= 15
    report = dict(
        gate_passed=passed, mean_abs_february_bias=mean_abs_bias, mean_february_rmse=rmse,
        mean_february_rmse_corrected=rmse_corrected, february=february_rows, panels=panels, **summary,
        note="Gap exactly 2. February is a TRAIN proxy. March and April labels were not read.",
    )
    if not passed:
        (args.out / "gap2.json").write_text(json.dumps(report))
        print("GATE failed", json.dumps(summary), flush=True)
        return
    if not base.drop(columns=["speed_kmh"]).equals(original.drop(columns=["speed_kmh"])):
        raise SystemExit("A non-speed column changed")
    csv_path = args.out / "submission_ES3.csv"
    base.to_csv(csv_path, index=False, lineterminator="\n")
    check = pd.read_csv(csv_path, dtype=str)
    assert list(check.columns) == COLUMNS and len(check) == 6980503
    assert check.drop(columns=["speed_kmh"]).equals(original.drop(columns=["speed_kmh"]))
    archive = archive_csv(csv_path)
    assert digest(args.base) == ES2_SHA
    report["candidate_sha256"] = digest(archive)
    report["uploaded"] = False
    (args.out / "gap2.json").write_text(json.dumps(report))
    print("GATE passed", json.dumps({"candidate_sha256": report["candidate_sha256"], **summary}), flush=True)


if __name__ == "__main__":
    main()
