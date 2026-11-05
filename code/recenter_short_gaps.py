"""Recenter short-gap speeds onto the same-day interpolant.

This is the speed edit in submission_ES4.zip. A shell is a set of target
cells whose nearest observed neighbors, on both sides, are a stated number of
five-minute slots away. For each link, subtract the mean of
(submitted speed - interpolant) on that shell, and clip to [5, 140].

Flow, queues and demand are copied from the parent. The parent of the final
shell is submission ES3, which already contains the one-slot and two-slot
edits and the ongoing-queue replacement. The selected ES4 archive in
submissions/ is the file that was uploaded.
"""
from __future__ import annotations

import argparse
import os
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
# The shared estimators (cache, t1_base, t1_features) live in tfb/.
sys.path.insert(0, str(ROOT / "tfb"))

# Imported in main(), after TFB_REL and TFB_CACHE point at the release and the
# cache directory, because these modules read both paths at import time.
cache = None
interp_day = None
prev_next = None


COLUMNS = ["submission_id", "task", "speed_kmh", "flow_vph", "queue_pred", "path_flow"]
MIN_N = 100


def observed(values, eligible, target):
    return np.where(eligible.astype(bool) & np.isfinite(values) & ~target.astype(bool), values, np.nan)


def parse_shells(text: str):
    shells = []
    for part in text.split(","):
        part = part.strip()
        if "-" in part:
            lo, hi = part.split("-")
            shells.append((int(lo), int(hi)))
        else:
            shells.append((int(part), int(part)))
    return shells


def link_shifts(z, part, speeds, shells):
    dates = [str(day)[:10] for day in z["dates"]]
    loc = {day: i for i, day in enumerate(dates)}
    link_loc = {link: i for i, link in enumerate(z["links"])}
    day = part.timestamp.str[:10].map(loc)
    link = part.link_id.map(link_loc)
    if day.isna().any() or link.isna().any():
        raise SystemExit("A state row is missing from the cache")
    day = day.to_numpy(int)
    link = link.to_numpy(int)
    slot = part.timestamp.str[11:13].astype(int).to_numpy() * 12 + part.timestamp.str[14:16].astype(int).to_numpy() // 5
    order = np.argsort(day, kind="mergesort")
    inverse = np.empty_like(order)
    inverse[order] = np.arange(len(order))
    day, slot, link = day[order], slot[order], link[order]
    speed = speeds.to_numpy(np.float64)[order]
    pred = np.empty(len(part))
    gap = np.empty(len(part))
    valid = np.zeros(len(part), bool)
    start = 0
    while start < len(day):
        stop = start + 1
        while stop < len(day) and day[stop] == day[start]:
            stop += 1
        observed_speed = observed(z["mspeed"][int(day[start])], z["elig"][int(day[start])], z["target"][int(day[start])])
        interpolated = interp_day(observed_speed)
        _, gap_prev, _, gap_next = prev_next(observed_speed)
        chosen = slice(start, stop)
        pred[chosen] = interpolated[slot[chosen], link[chosen]]
        gap[chosen] = np.maximum(gap_prev, gap_next)[slot[chosen], link[chosen]]
        valid[chosen] = ((gap_prev < 90) & (gap_next < 90))[slot[chosen], link[chosen]]
        start = stop
    pred, gap, valid, speed, link = pred[inverse], gap[inverse], valid[inverse], speed[inverse], link[inverse]
    pieces = []
    n_links = len(z["links"])
    for lo, hi in shells:
        take = valid & (gap >= lo) & (gap <= hi) & np.isfinite(pred) & np.isfinite(speed)
        counts = np.bincount(link[take], minlength=n_links)
        summed = np.bincount(link[take], weights=(speed - pred)[take], minlength=n_links)
        shift = np.zeros(n_links)
        enough = counts >= MIN_N
        shift[enough] = summed[enough] / np.maximum(counts[enough], 1)
        apply = take & enough[link]
        corrected = np.clip(speed[apply] - shift[link[apply]], 5.0, 140.0)
        rms = float(np.sqrt(np.mean(np.square(shift[enough])))) if enough.any() else 0.0
        use = rms >= 0.40 and (enough.any() and float(np.max(np.abs(shift[enough]))) <= 15) and 0 < int(apply.sum()) <= 3_000_000
        pieces.append((apply, corrected, use, int(apply.sum()), rms))
    return pieces


def load_modules(release_root: Path, cache_root: Path):
    global cache, interp_day, prev_next
    os.environ["TFB_REL"] = str(release_root)
    os.environ["TFB_CACHE"] = str(cache_root)
    import cache as cache_module
    from t1_base import interp_day as interp
    from t1_features import prev_next as neighbours
    cache_module.REL = release_root
    cache_module.OUT = cache_root
    cache, interp_day, prev_next = cache_module, interp, neighbours


def load_cache(panel: str, split: str):
    """Load the dense cache for one panel and split, building it from the release if absent."""
    if not (cache.OUT / f"{panel}_{split}.npz").is_file():
        cache.build(panel, split)
    return cache.load(panel, split)


def main():
    parser = argparse.ArgumentParser(description="Recenter short-gap speeds of a parent submission.")
    parser.add_argument("--release-root", type=Path, required=True,
                        help="Directory with submission_key.csv, corridors/ and task1/")
    parser.add_argument("--parent", type=Path, required=True, help="Parent submission zip (ES3 for the ES4 shell)")
    parser.add_argument("--out", type=Path, required=True, help="Output directory")
    parser.add_argument("--shells", default="3-6", help="Comma-separated gap shells, e.g. 1,2,3-6")
    parser.add_argument("--cache-root", type=Path, default=None,
                        help="Dense cache directory (default: <out>/cache); built from the release when missing")
    args = parser.parse_args()
    load_modules(args.release_root.resolve(), (args.cache_root or args.out / "cache").resolve())
    shells = parse_shells(args.shells)
    key = pd.read_csv(args.release_root / "submission_key.csv", dtype=str)
    with zipfile.ZipFile(args.parent) as archive:
        frame = pd.read_csv(archive.open(archive.namelist()[0]), dtype=str)
    if list(frame.columns) != COLUMNS or len(frame) != len(key):
        raise SystemExit("Parent submission does not match the release key")
    state = key.task.eq("state")
    speeds = pd.to_numeric(frame.loc[state, "speed_kmh"])
    changed = 0
    for panel in cache.PANELS:
        for split in ("validation", "private"):
            part = key.loc[state & key.panel.eq(panel) & key.split.eq(split)]
            z = load_cache(panel, split)
            for apply, corrected, use, count, rms in link_shifts(z, part, speeds.loc[part.index], shells):
                print(panel, split, count, round(rms, 4), use, flush=True)
                if not use:
                    continue
                position = part.index.to_numpy()[apply]
                frame.loc[position, "speed_kmh"] = [f"{value:.8f}" for value in corrected]
                changed += int(apply.sum())
    args.out.mkdir(parents=True, exist_ok=True)
    csv_path = args.out / "submission_ES4_recentered.csv"
    frame.to_csv(csv_path, index=False, lineterminator="\n")
    print("WROTE", csv_path, "changed", changed)


if __name__ == "__main__":
    main()
