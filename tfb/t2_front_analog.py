"""Train-only analog model for ongoing queue-front movement.

The model does not predict cells independently.  It learns which simple queue
front transformation tends to work for legally observable traffic states, using
all train episodes that satisfy the published ongoing trigger and hard-window
filter.  A held query selects 50 nearest train episodes from other months, then
chooses the transformation with the highest mean IoU in those analogues.

Acceptance is evaluated only on the 40 released train ongoing windows with the
whole query month excluded from the analogue pool.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.ndimage import median_filter

sys.path.insert(0, "tfb")
import cache
import t2_oracles as oracle
import t2_persist as tp
from t2 import daytype, statics


OUT = Path.home() / "tfb/work/t2_front_analog"
K_NEIGHBOURS = 50
DELTAS = tuple(range(-4, 5))
PARAMS = [("front", a, b) for a in DELTAS for b in DELTAS] + [("clear", c, c) for c in range(2, 7)] + [("ffill", 0, 0)]
PARAM_NAMES = [f"{kind}_{a}_{b}" for kind, a, b in PARAMS]


def candidate(q0: np.ndarray, param: tuple[str, int, int]) -> np.ndarray:
    kind, a, b = param
    if kind == "ffill":
        return np.repeat(q0[None], 6, axis=0)
    if kind == "clear":
        return oracle.front_path(q0, 0, 0, clear_after=a)
    return oracle.front_path(q0, a, b)


def trigger(qe: np.ndarray) -> bool:
    return qe.sum() >= 2 and (qe.sum(0) >= 2).any()


def features(z: dict, d: int, s0: int, vcut: np.ndarray, cap: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    el = z["elig"][d, s0 - 12:s0].astype(bool)
    speed = z["speed"][d, s0 - 12:s0]
    flow = z["flow"][d, s0 - 12:s0]
    v_el = np.where(el, speed, np.nan)
    v_all = speed.copy()
    q0 = tp.variants(v_el, v_all, vcut)["ffill_inel"]
    vf = pd.DataFrame(v_all).ffill().to_numpy()
    qf = pd.DataFrame(flow).ffill().to_numpy()
    ratio = vf / vcut[None, :]
    queued_hist = np.nan_to_num(ratio, nan=9) <= 1
    counts = queued_hist.sum(1) / max(len(vcut), 1)
    cap_sum = max(float(np.nansum(cap)), 1.0)
    totals = np.nansum(qf, 1) / cap_sum
    last_ratio = ratio[-1]
    pos = np.flatnonzero(q0)
    first = pos[0] / max(len(q0) - 1, 1) if len(pos) else -1
    last = pos[-1] / max(len(q0) - 1, 1) if len(pos) else -1
    ramps = np.where(
        z["relig"][d, s0 - 12:s0].astype(bool),
        z["rflow"][d, s0 - 12:s0],
        np.nan,
    )
    ramp_total = np.nansum(ramps, 1) / cap_sum if ramps.shape[1] else np.zeros(12)
    x = np.r_[
        counts,
        totals[-6:],
        ramp_total[-6:],
        [
            q0.mean(), first, last, len(oracle.runs(q0)),
            np.nanmin(last_ratio), np.nanmean(last_ratio),
            np.nanpercentile(last_ratio, 25), np.nanpercentile(last_ratio, 75),
            totals[-1] - totals[-4], ramp_total[-1] - ramp_total[-4],
            el.mean(),
            np.sin(2 * np.pi * s0 / 288), np.cos(2 * np.pi * s0 / 288),
        ],
    ].astype(float)
    return x, q0


def event_scores(z: dict, d: int, s0: int, vcut: np.ndarray, cap: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
    el = z["elig"][d].astype(bool)
    speed = z["speed"][d]
    hist_q = (np.nan_to_num(speed[s0 - 12:s0], nan=1e9) <= vcut) & el[s0 - 12:s0]
    if el[s0 - 12:s0].mean() < 0.7 or not trigger(hist_q):
        return None
    x, q0 = features(z, d, s0, vcut, cap)
    noisy, latent, known = oracle.label_views(z, d, s0, vcut)
    if not (noisy & known).any():
        return None
    baseline = np.repeat(q0[None], 6, axis=0)
    if oracle.iou(baseline, noisy, known) > 0.9:
        return None
    noisy_scores = np.asarray([oracle.iou(candidate(q0, p), noisy, known) for p in PARAMS])
    latent_scores = np.asarray([oracle.iou(candidate(q0, p), latent, known) for p in PARAMS])
    return x, q0, noisy_scores, latent_scores


def build_pool(panel: str, z: dict, vcut: np.ndarray, cap: np.ndarray) -> pd.DataFrame:
    rows = []
    weekends = (np.asarray(daytype(z["dates"])) >= 5).astype(int)
    # Thirty-minute spacing limits serial duplication while retaining thousands
    # of hard episodes across nine train months.
    for d, date in enumerate(z["dates"]):
        for s0 in range(12, 282, 6):
            result = event_scores(z, d, s0, vcut, cap)
            if result is None:
                continue
            x, q0, noisy, latent = result
            rec = {
                "panel": panel, "date": str(date), "month": str(date)[:7],
                "weekend": weekends[d], "slot": s0, "q0_n": int(q0.sum()),
            }
            rec.update({f"x{i}": value for i, value in enumerate(x)})
            rec.update({f"{name}|noisy": value for name, value in zip(PARAM_NAMES, noisy)})
            rec.update({f"{name}|latent": value for name, value in zip(PARAM_NAMES, latent)})
            rows.append(rec)
    return pd.DataFrame(rows)


def choose(pool: pd.DataFrame, query: np.ndarray, weekend: int, label: str) -> tuple[str, np.ndarray]:
    candidates = pool[pool.weekend == weekend]
    if len(candidates) < K_NEIGHBOURS:
        candidates = pool
    xcols = [f"x{i}" for i in range(len(query))]
    x = candidates[xcols].to_numpy(float)
    scale = np.nanstd(x, axis=0)
    scale = np.where(np.isfinite(scale) & (scale > 1e-6), scale, 1.0)
    distance = np.sqrt(np.nanmean(((x - query) / scale) ** 2, axis=1))
    near = candidates.iloc[np.argsort(distance)[: min(K_NEIGHBOURS, len(candidates))]]
    score_cols = [f"{name}|{label}" for name in PARAM_NAMES]
    best = str(near[score_cols].mean(axis=0).idxmax()).split("|")[0]
    return best, near.index.to_numpy()


def evaluate() -> pd.DataFrame:
    rows = []
    OUT.mkdir(parents=True, exist_ok=True)
    for panel in tp.Q_PANELS:
        z = cache.load(panel, "train")
        links = list(z["links"])
        dates = list(z["dates"])
        vcut = statics(panel, links)["vcut"]
        cap = pd.read_csv(
            tp.REL / "corridors" / panel / "network" / "fd_parameters.csv",
            dtype={"link_id": str},
        ).groupby("link_id").capacity_vph.mean().reindex(links).to_numpy()
        pool = build_pool(panel, z, vcut, cap)
        pool.to_parquet(OUT / f"pool_{panel}.parquet", index=False)
        wi = pd.read_csv(tp.REL / "task2" / panel / "train" / "window_index.csv")
        weekend = (np.asarray(daytype(z["dates"])) >= 5).astype(int)
        for w in wi[wi.condition == "queue_ongoing"].itertuples():
            d = dates.index(w.date)
            s0 = tp.slot_of(w.forecast_origin)
            query, q0 = features(z, d, s0, vcut, cap)
            held = pool[pool.month != w.date[:7]]
            noisy, latent, known = oracle.label_views(z, d, s0, vcut)
            rec = {
                "panel": panel, "date": w.date, "window_id": w.window_id,
                "n_pool": len(pool), "n_held_pool": len(held),
            }
            baseline = np.repeat(q0[None], 6, axis=0)
            for train_label in ("noisy", "latent"):
                name, _ = choose(held, query, int(weekend[d]), train_label)
                pred = candidate(q0, PARAMS[PARAM_NAMES.index(name)])
                rec[f"choice_{train_label}"] = name
                rec[f"analog_{train_label}_noisy"] = oracle.iou(pred, noisy, known)
                rec[f"analog_{train_label}_latent"] = oracle.iou(pred, latent, known)
            rec["ffill_noisy"] = oracle.iou(baseline, noisy, known)
            rec["ffill_latent"] = oracle.iou(baseline, latent, known)
            rec["oracle_noisy"] = max(oracle.iou(candidate(q0, p), noisy, known) for p in PARAMS)
            rec["oracle_latent"] = max(oracle.iou(candidate(q0, p), latent, known) for p in PARAMS)
            rows.append(rec)
        print(panel, "pool", len(pool), flush=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(OUT / "eval_windows.csv", index=False)
    summary = []
    for method in ("analog_noisy", "analog_latent", "oracle"):
        rec = {"method": method, "n": len(frame), "n_dates": frame.date.nunique()}
        for label in ("noisy", "latent"):
            col = f"{method}_{label}"
            delta = frame[col] - frame[f"ffill_{label}"]
            by_date = pd.DataFrame({"date": frame.date, "x": delta}).groupby("date").x.mean()
            rec[label] = frame[col].mean()
            rec[f"delta_{label}"] = delta.mean()
            rec[f"se_date_{label}"] = by_date.std(ddof=1) / np.sqrt(len(by_date))
        summary.append(rec)
    result = pd.DataFrame(summary)
    result.to_csv(OUT / "summary.csv", index=False)
    print(result.round(4).to_string(index=False))
    print("Noisy-trained choices:", frame.choice_noisy.value_counts().to_dict())
    return frame


if __name__ == "__main__":
    evaluate()
