"""Recenter ES1 one-slot speeds onto the same-day interpolant.

Train labels show that interpolant is unbiased by link (about 0.02 km/h).
ES1's per-link mean on the same cells is not. March and April labels are not
read. Flow, queues and demand are copied unchanged. ES1's zip is not modified.
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


ES1_SHA = "f5fac2c81a5061b1fcb589bc24c59dd37b8bc3d4302baaa15c3564f3d22858c8"
COLUMNS = ["submission_id", "task", "speed_kmh", "flow_vph", "queue_pred", "path_flow"]
MIN_N = 100


def recenter(z, part, speeds):
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
    gap = np.empty(len(part))
    start = 0
    while start < len(day):
        stop = start + 1
        while stop < len(day) and day[stop] == day[start]:
            stop += 1
        d = int(day[start])
        observed = np.where(
            z["elig"][d].astype(bool) & np.isfinite(z["mspeed"][d]) & ~z["target"][d].astype(bool),
            z["mspeed"][d], np.nan)
        iv = interp_day(observed)
        _, gp, _, gn = prev_next(observed)
        choose = slice(start, stop)
        pred[choose] = iv[slot[choose], link[choose]]
        gap[choose] = np.maximum(gp, gn)[slot[choose], link[choose]]
        start = stop
    pred, gap, sub, link = pred[inv], gap[inv], sub[inv], link[inv]
    take = np.isfinite(pred) & np.isfinite(sub) & (gap <= 1)
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
    if args.out.exists():
        raise SystemExit("Output directory already exists")
    args.out.mkdir(parents=True)
    assert digest(args.base) == ES1_SHA
    key = pd.read_csv(args.release_root / "submission_key.csv", dtype=str,
                      usecols=["submission_id", "task", "split", "panel", "timestamp", "link_id"])
    with zipfile.ZipFile(args.base) as archive:
        member = archive.namelist()[0]
        base = pd.read_csv(archive.open(member), dtype=str)
    assert list(base.columns) == COLUMNS and len(base) == len(key) == 6980503
    assert (base.submission_id.to_numpy() == key.submission_id.to_numpy()).all()
    assert (base.task.to_numpy() == key.task.to_numpy()).all()
    original = base.copy()
    state = key.task.eq("state")
    speeds = pd.to_numeric(base.loc[state, "speed_kmh"])
    key_state = key.loc[state]
    deltas = []
    panels = []
    for panel in cache.PANELS:
        for split in ("validation", "private"):
            z = cache.load(panel, split)
            part = key_state[(key_state.panel == panel) & (key_state.split == split)]
            updated, apply, shift, counts = recenter(z, part, speeds.loc[part.index])
            old = speeds.loc[part.index].to_numpy()
            delta = updated[apply] - old[apply]
            deltas.append(delta)
            base.loc[part.index[apply], "speed_kmh"] = [f"{value:.8f}" for value in updated[apply]]
            panels.append(dict(
                panel=panel, split=split, n_gap1=int(apply.sum()),
                links=int((counts >= MIN_N).sum()),
                rms_shift=float(np.sqrt(np.mean(np.square(shift[counts >= MIN_N])))) if np.any(counts >= MIN_N) else 0.0,
                clipped=int(np.sum((updated[apply] <= 5.0 + 1e-6) | (updated[apply] >= 140.0 - 1e-6))),
            ))
            print(panel, split, panels[-1], flush=True)
    delta = np.concatenate(deltas)
    changed = int(np.sum(np.abs(delta) > 1e-8))
    summary = dict(
        changed_cells=changed,
        mean_abs_delta=float(np.mean(np.abs(delta))),
        rms_delta=float(np.sqrt(np.mean(delta ** 2))),
        p50_abs=float(np.quantile(np.abs(delta), 0.50)),
        p90_abs=float(np.quantile(np.abs(delta), 0.90)),
        p99_abs=float(np.quantile(np.abs(delta), 0.99)),
        max_abs=float(np.max(np.abs(delta))),
    )
    print("SUMMARY", json.dumps(summary), flush=True)
    if not (500_000 <= changed <= 2_500_000):
        raise SystemExit("Changed-cell count outside the expected one-slot band; no file written")
    if summary["max_abs"] > 15:
        raise SystemExit("A link shift exceeds 15 km/h; no file written")
    same_other = base.drop(columns=["speed_kmh"]).equals(original.drop(columns=["speed_kmh"]))
    if not same_other:
        raise SystemExit("A non-speed column changed; no file written")
    csv_path = args.out / "submission_ES2.csv"
    base.to_csv(csv_path, index=False, lineterminator="\n")
    check = pd.read_csv(csv_path, dtype=str)
    assert list(check.columns) == COLUMNS and len(check) == 6980503
    assert check.drop(columns=["speed_kmh"]).equals(original.drop(columns=["speed_kmh"]))
    archive = archive_csv(csv_path)
    assert digest(args.base) == ES1_SHA
    report = dict(
        candidate="ES2", base_sha256=ES1_SHA, rows=6980503, min_n=MIN_N,
        preserved="flow, queues, demand, and speeds outside one-slot cells",
        candidate_sha256=digest(archive), uploaded=False, panels=panels, **summary,
        note="Train-label proxies say the interpolant link mean is unbiased. This file removes ES1's link-mean offset on one-slot cells. Not a Kaggle score yet.",
    )
    (args.out / "build.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({"candidate_sha256": report["candidate_sha256"], **summary}), flush=True)


if __name__ == "__main__":
    main()
