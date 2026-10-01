"""Legal ongoing front from motion already visible in [T-60, T).

No labels, no slot T.  Queue masks over the 12 history slots are the same
``ffill_inel`` rule as U2/J1.  The largest contiguous run's two ends are
tracked; a line is fit on the last six observed fronts and rolled six steps.
If the history does not support a slope (short queue, missing fronts, or a
fit that would move more than four links in 30 minutes) the action is ffill.

Evaluated leave-one-date-out only in the sense that the rule has no fitted
parameters; the 40 released train ongoing windows are scored against ffill.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, "tfb")
import cache
import t2_oracles as oracle
import t2_persist as tp
from t2 import statics


OUT = Path.home() / "tfb/work/t2_hist_front"
MAX_SHIFT = 4


def history_queue(v_el: np.ndarray, v_all: np.ndarray, vcut: np.ndarray) -> np.ndarray:
    masks = []
    for t in range(len(v_all)):
        masks.append(tp.variants(v_el[: t + 1], v_all[: t + 1], vcut)["ffill_inel"])
    return np.stack(masks)


def primary_fronts(q_hist: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    left = np.full(len(q_hist), np.nan)
    right = np.full(len(q_hist), np.nan)
    for t, row in enumerate(q_hist):
        runs = oracle.runs(tp.fill(row))
        if not runs:
            continue
        a, b = max(runs, key=lambda ab: ab[1] - ab[0])
        left[t], right[t] = a, b
    return left, right


def slope(y: np.ndarray) -> tuple[float, float]:
    t = np.arange(len(y), dtype=float)
    ok = np.isfinite(y)
    if ok.sum() < 3:
        last = float(y[ok][-1]) if ok.any() else np.nan
        return 0.0, last
    coef = np.polyfit(t[ok], y[ok], 1)
    return float(coef[0]), float(coef[1])


def rollout(q0: np.ndarray, left: np.ndarray, right: np.ndarray) -> np.ndarray:
    baseline = np.repeat(q0[None], 6, axis=0)
    recent_l, recent_r = left[-6:], right[-6:]
    if np.isfinite(recent_l).sum() < 3 or np.isfinite(recent_r).sum() < 3:
        return baseline
    sl, il = slope(recent_l)
    sr, ir = slope(recent_r)
    if abs(sl) * 6 > MAX_SHIFT or abs(sr) * 6 > MAX_SHIFT:
        return baseline
    if abs(sl) < 0.05 and abs(sr) < 0.05:
        return baseline
    out = np.zeros((6, len(q0)), bool)
    t0 = 5.0
    for h in range(1, 7):
        a = int(round(il + sl * (t0 + h)))
        b = int(round(ir + sr * (t0 + h)))
        a, b = max(0, min(a, b)), min(len(q0) - 1, max(a, b))
        if a <= b:
            out[h - 1, a:b + 1] = True
    return out if out.any() else baseline


def predict_window(z: dict, d: int, s0: int, vcut: np.ndarray) -> np.ndarray:
    el = z["elig"][d, s0 - 12:s0].astype(bool)
    v_el = np.where(el, z["speed"][d, s0 - 12:s0], np.nan)
    v_all = z["speed"][d, s0 - 12:s0].copy()
    q0 = tp.variants(v_el, v_all, vcut)["ffill_inel"]
    q_hist = history_queue(v_el, v_all, vcut)
    left, right = primary_fronts(q_hist)
    return rollout(q0, left, right)


def evaluate() -> pd.DataFrame:
    rows = []
    for panel in tp.Q_PANELS:
        z = cache.load(panel, "train")
        vcut = statics(panel, z["links"])["vcut"]
        dates = list(z["dates"])
        wi = pd.read_csv(cache.REL / "task2" / panel / "train" / "window_index.csv")
        for w in wi[wi.condition == "queue_ongoing"].itertuples():
            d = dates.index(w.date)
            s0 = tp.slot_of(w.forecast_origin)
            noisy, latent, known = oracle.label_views(z, d, s0, vcut)
            el = z["elig"][d, s0 - 12:s0].astype(bool)
            v_el = np.where(el, z["speed"][d, s0 - 12:s0], np.nan)
            v_all = z["speed"][d, s0 - 12:s0].copy()
            q0 = tp.variants(v_el, v_all, vcut)["ffill_inel"]
            ffill = np.repeat(q0[None], 6, axis=0)
            pred = predict_window(z, d, s0, vcut)
            rec = {"panel": panel, "date": w.date, "window_id": w.window_id}
            for name, truth in (("noisy", noisy), ("latent", latent)):
                rec[f"ffill_{name}"] = oracle.iou(ffill, truth, known)
                rec[f"histfront_{name}"] = oracle.iou(pred, truth, known)
            rows.append(rec)
    out = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT / "windows.csv", index=False)
    for label in ("noisy", "latent"):
        delta = out[f"histfront_{label}"] - out[f"ffill_{label}"]
        by_date = pd.DataFrame({"date": out.date, "x": delta}).groupby("date").x.mean()
        se = by_date.std(ddof=1) / np.sqrt(len(by_date))
        print(
            f"{label}: ffill {out[f'ffill_{label}'].mean():.4f}  "
            f"histfront {out[f'histfront_{label}'].mean():.4f}  "
            f"delta {delta.mean():+.4f} se_date {se:.4f}",
            flush=True,
        )
    return out


def build_rows(split: str) -> pd.DataFrame:
    parts = []
    for panel in tp.Q_PANELS:
        z = cache.load(panel, split)
        links = list(z["links"])
        vcut = statics(panel, links)["vcut"]
        zz = dict(z)
        zz["speed"], zz["flow"] = z["mspeed"], z["mflow"]
        dates = list(z["dates"])
        wi = pd.read_csv(cache.REL / "task2" / panel / split / "window_index.csv")
        for w in wi[wi.condition == "queue_ongoing"].itertuples():
            d = dates.index(w.date)
            s0 = tp.slot_of(w.forecast_origin)
            assert 12 <= s0 and s0 + 6 < 288, f"window {w.window_id} crosses midnight"
            pred = predict_window(zz, d, s0, vcut).astype(int)
            ts = [f"{w.date}T{(s0 + h) // 12:02d}:{(s0 + h) % 12 * 5:02d}:00Z" for h in range(1, 7)]
            parts.append(pd.DataFrame({
                "window_id": w.window_id,
                "timestamp": np.repeat(ts, len(links)),
                "link_id": np.tile(links, 6),
                "queue_pred": pred.reshape(-1),
            }))
    return pd.concat(parts, ignore_index=True)


if __name__ == "__main__":
    evaluate()
