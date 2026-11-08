"""Recenter ES3 speeds on wider gap shells that pass the ES3 gate.

February labels test the same-day speed interpolant. A shell is eligible only
when that interpolant is unbiased and the correction does not raise February
RMSE. The link offset is read from ES3 on March and April, without labels.
Flow, queues, demand, and every non-eligible speed stay copied. ES3 is not
modified. A file is written only if at least one shell passes.
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


ES3_SHA = "712f772011ede5d63e1ad11d739668ee566074e5687e72224d5c7e6c55c96f71"
COLUMNS = ["submission_id", "task", "speed_kmh", "flow_vph", "queue_pred", "path_flow"]
MIN_FIT = 20
MIN_N = 100
SHELLS = [(3, 3), (4, 4), (5, 5), (6, 6), (7, 7), (8, 8), (9, 12), (13, 24), (25, 48)]


def name_of(lo, hi):
    return f"g{lo}" if lo == hi else f"g{lo}_{hi}"


def observed(z, day, key):
    return np.where(
        z["elig"][day].astype(bool) & np.isfinite(z[key][day]) & ~z["target"][day].astype(bool),
        z[key][day], np.nan)


def gap_matrix(observed_speed):
    _, gap_prev, _, gap_next = prev_next(observed_speed)
    gap = np.maximum(gap_prev, gap_next)
    valid = (gap_prev < 90) & (gap_next < 90)
    return gap, valid


def rmse(values):
    return float(np.sqrt(np.mean(values * values))) if len(values) else None


def february(panel):
    z = cache.load(panel, "train")
    months = np.array([str(d)[:7] for d in z["dates"]])
    held = {name_of(lo, hi): {"month": [], "link": [], "err": [], "ferr": []} for lo, hi in SHELLS}
    for day, month in enumerate(months):
        target = z["target"][day].astype(bool) & np.isfinite(z["speed"][day]) & np.isfinite(z["flow"][day])
        if not target.any():
            continue
        speed_obs = observed(z, day, "mspeed")
        flow_obs = observed(z, day, "mflow")
        pred = interp_day(speed_obs)
        fpred = interp_day(flow_obs)
        gap, valid = gap_matrix(speed_obs)
        finite = target & valid & np.isfinite(pred) & np.isfinite(fpred)
        if not finite.any():
            continue
        link = np.broadcast_to(np.arange(finite.shape[1]), finite.shape)
        for lo, hi in SHELLS:
            take = finite & (gap >= lo) & (gap <= hi)
            if not take.any():
                continue
            bucket = held[name_of(lo, hi)]
            bucket["month"].append(np.full(int(take.sum()), month))
            bucket["link"].append(link[take].astype(np.int16))
            bucket["err"].append((pred - z["speed"][day])[take].astype(np.float64))
            bucket["ferr"].append((fpred - z["flow"][day])[take].astype(np.float64))
    rows = []
    for lo, hi in SHELLS:
        bucket = held[name_of(lo, hi)]
        if not bucket["link"]:
            rows.append(dict(shell=name_of(lo, hi), n=0))
            continue
        month = np.concatenate(bucket["month"])
        link = np.concatenate(bucket["link"])
        err = np.concatenate(bucket["err"])
        ferr = np.concatenate(bucket["ferr"])
        n_links = int(link.max()) + 1
        train = np.array([item < "2031-02" for item in month])
        test = np.array([item == "2031-02" for item in month])
        seen = np.bincount(link[train], minlength=n_links)
        summed = np.bincount(link[train], weights=err[train], minlength=n_links)
        bias = np.zeros(n_links)
        ok = seen >= MIN_FIT
        bias[ok] = summed[ok] / np.maximum(seen[ok], 1)
        raw = err[test]
        fraw = ferr[test]
        if raw.size == 0:
            rows.append(dict(shell=name_of(lo, hi), n=0))
            continue
        fixed = raw - bias[link[test]]
        fseen = np.bincount(link[train], minlength=n_links)
        fsum = np.bincount(link[train], weights=ferr[train], minlength=n_links)
        fbias = np.zeros(n_links)
        fok = fseen >= MIN_FIT
        fbias[fok] = fsum[fok] / np.maximum(fseen[fok], 1)
        rows.append(dict(
            shell=name_of(lo, hi), n=int(test.sum()), links=int(ok.sum()),
            mean_abs_bias=float(np.mean(np.abs(bias[ok]))) if ok.any() else None,
            rmse=rmse(raw), rmse_corrected=rmse(fixed),
            flow_mean_abs_bias=float(np.mean(np.abs(fbias[fok]))) if fok.any() else None,
            flow_rmse=rmse(fraw),
            flow_rmse_corrected=rmse(fraw - fbias[link[test]]),
        ))
    return rows


def shifts_for(z, part, speeds, flows):
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
    fsub = flows.to_numpy(np.float64)
    order = np.argsort(day, kind="mergesort")
    inv = np.empty_like(order)
    inv[order] = np.arange(len(order))
    day, slot, link, sub, fsub = day[order], slot[order], link[order], sub[order], fsub[order]
    n = len(part)
    n_links = len(z["links"])
    pred = np.empty(n)
    fpred = np.empty(n)
    gap = np.empty(n)
    valid = np.zeros(n, bool)
    start = 0
    while start < len(day):
        stop = start + 1
        while stop < len(day) and day[stop] == day[start]:
            stop += 1
        d = int(day[start])
        speed_obs = observed(z, d, "mspeed")
        iv = interp_day(speed_obs)
        fv = interp_day(observed(z, d, "mflow"))
        g, ok = gap_matrix(speed_obs)
        choose = slice(start, stop)
        pred[choose] = iv[slot[choose], link[choose]]
        fpred[choose] = fv[slot[choose], link[choose]]
        gap[choose] = g[slot[choose], link[choose]]
        valid[choose] = ok[slot[choose], link[choose]]
        start = stop
    pred, fpred, gap, valid, sub, fsub, link = (
        pred[inv], fpred[inv], gap[inv], valid[inv], sub[inv], fsub[inv], link[inv])
    shell_rows = []
    for lo, hi in SHELLS:
        take = valid & (gap >= lo) & (gap <= hi) & np.isfinite(pred) & np.isfinite(sub)
        counts = np.bincount(link[take], minlength=n_links)
        summed = np.bincount(link[take], weights=(sub - pred)[take], minlength=n_links)
        shift = np.zeros(n_links)
        enough = counts >= MIN_N
        shift[enough] = summed[enough] / np.maximum(counts[enough], 1)
        apply = take & enough[link]
        corrected = np.clip(sub[apply] - shift[link[apply]], 5.0, 140.0)
        ftake = valid & (gap >= lo) & (gap <= hi) & np.isfinite(fpred) & np.isfinite(fsub)
        fcounts = np.bincount(link[ftake], minlength=n_links)
        fsummed = np.bincount(link[ftake], weights=(fsub - fpred)[ftake], minlength=n_links)
        fenough = fcounts >= MIN_N
        fshift = np.zeros(n_links)
        fshift[fenough] = fsummed[fenough] / np.maximum(fcounts[fenough], 1)
        shell_rows.append(dict(
            shell=name_of(lo, hi), n=int(apply.sum()), links=int(enough.sum()),
            rms_shift=float(np.sqrt(np.mean(np.square(shift[enough])))) if enough.any() else 0.0,
            max_abs_link=float(np.max(np.abs(shift[enough]))) if enough.any() else 0.0,
            flow_n=int((ftake & fenough[link]).sum()),
            flow_rms_shift=float(np.sqrt(np.mean(np.square(fshift[fenough])))) if fenough.any() else 0.0,
            apply_index=np.flatnonzero(apply),
            corrected=corrected,
        ))
    return shell_rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--release-root", type=Path, default=cache.REL)
    args = parser.parse_args()
    if (args.out / "submission_ES4.zip").exists():
        raise SystemExit("ES4 archive already exists")
    args.out.mkdir(parents=True, exist_ok=True)
    cache.REL = args.release_root
    february_rows = []
    for panel in cache.PANELS:
        for row in february(panel):
            row = dict(panel=panel, **row)
            february_rows.append(row)
            print("FEB", json.dumps(row), flush=True)
    eligible = []
    for lo, hi in SHELLS:
        shell = name_of(lo, hi)
        usable = [row for row in february_rows if row["shell"] == shell and row.get("n") and row.get("mean_abs_bias") is not None]
        if not usable:
            continue
        mean_abs_bias = float(np.mean([row["mean_abs_bias"] for row in usable]))
        mean_rmse = float(np.mean([row["rmse"] for row in usable]))
        mean_fixed = float(np.mean([row["rmse_corrected"] for row in usable]))
        passed = mean_abs_bias < 0.15 and mean_fixed <= mean_rmse + 0.02
        print("FEBRUARY", shell, mean_abs_bias, mean_rmse, mean_fixed, passed, flush=True)
        if passed:
            eligible.append(shell)
    assert digest(args.base) == ES3_SHA
    key = pd.read_csv(args.release_root / "submission_key.csv", dtype=str,
                      usecols=["submission_id", "task", "split", "panel", "timestamp", "link_id"])
    with zipfile.ZipFile(args.base) as archive:
        base = pd.read_csv(archive.open(archive.namelist()[0]), dtype=str)
    assert list(base.columns) == COLUMNS and len(base) == len(key) == 6980503
    assert (base.submission_id.to_numpy() == key.submission_id.to_numpy()).all()
    original = base.copy()
    state = key.task.eq("state")
    speeds = pd.to_numeric(base.loc[state, "speed_kmh"])
    flows = pd.to_numeric(base.loc[state, "flow_vph"])
    key_state = key.loc[state]
    panels, pieces = [], []
    chosen = set(eligible)
    for panel in cache.PANELS:
        for split in ("validation", "private"):
            z = cache.load(panel, split)
            part = key_state[(key_state.panel == panel) & (key_state.split == split)]
            shell_rows = shifts_for(z, part, speeds.loc[part.index], flows.loc[part.index])
            position = part.index.to_numpy()
            for row in shell_rows:
                use = row["shell"] in chosen and row["rms_shift"] >= 0.40 and row["max_abs_link"] <= 15 and 0 < row["n"] <= 3_000_000
                corrected = row.pop("corrected")
                apply_index = row.pop("apply_index")
                if use:
                    base.loc[position[apply_index], "speed_kmh"] = [f"{value:.8f}" for value in corrected]
                    pieces.append(corrected - speeds.loc[position[apply_index]].to_numpy())
                row.update(panel=panel, split=split, apply=bool(use))
                panels.append({k: v for k, v in row.items()})
                print(panel, split, row["shell"], row["n"], round(row["rms_shift"], 4), use, flush=True)
    delta = np.concatenate(pieces) if pieces else np.zeros(1)
    changed = int(np.sum(np.abs(delta) > 1e-8)) if pieces else 0
    applied_offsets = [row["rms_shift"] for row in panels if row["apply"]]
    summary = dict(
        eligible_february=eligible,
        changed_cells=changed,
        mean_abs_delta=float(np.mean(np.abs(delta))) if changed else 0.0,
        rms_delta=float(np.sqrt(np.mean(delta ** 2))) if changed else 0.0,
        max_abs=float(np.max(np.abs(delta))) if changed else 0.0,
        mean_applied_rms=float(np.mean(applied_offsets)) if applied_offsets else 0.0,
    )
    passed = bool(eligible) and changed >= 20_000 and changed <= 5_000_000 and summary["max_abs"] <= 15 and summary["mean_applied_rms"] >= 0.40
    report = dict(
        gate_passed=passed, february=february_rows, panels=panels, **summary,
        note="Wider gaps on ES3. February is a TRAIN proxy. March and April labels were not read. Flow was measured and not changed.",
    )
    if not passed:
        (args.out / "gap_ladder.json").write_text(json.dumps(report))
        print("GATE failed", json.dumps(summary), flush=True)
        return
    if not base.drop(columns=["speed_kmh"]).equals(original.drop(columns=["speed_kmh"])):
        raise SystemExit("A non-speed column changed")
    csv_path = args.out / "submission_ES4.csv"
    base.to_csv(csv_path, index=False, lineterminator="\n")
    check = pd.read_csv(csv_path, dtype=str)
    assert list(check.columns) == COLUMNS and len(check) == 6980503
    assert check.drop(columns=["speed_kmh"]).equals(original.drop(columns=["speed_kmh"]))
    archive = archive_csv(csv_path)
    assert digest(args.base) == ES3_SHA
    report["candidate_sha256"] = digest(archive)
    report["uploaded"] = False
    (args.out / "gap_ladder.json").write_text(json.dumps(report))
    print("GATE passed", json.dumps({"candidate_sha256": report["candidate_sha256"], **summary}), flush=True)


if __name__ == "__main__":
    main()
