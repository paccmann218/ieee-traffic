"""Non-learned persist / clear / grow mixture for ongoing Task 2 queues.

Default remains J1 ffill_inel. Grow/shrink uses the triangular-FD wave
w = -capacity / (k_jam - k_crit) ≈ -25 km/h, in kilometres not link counts
(25 m cells would otherwise jump 80 links per slot).

Gate (history [T-60,T) only, never slot T):
  clear if bottleneck discharge exceeds upstream arrival and the queue is already shrinking
  grow  if arrival exceeds discharge and the queue is already growing
  else persist

Must beat ffill_inel on leave-date noisy *and* latent IoU or it does not ship.

Usage: python tfb/t2_moe.py selftest | eval
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.ndimage import median_filter

sys.path.insert(0, "tfb")
import cache
import legal as L
import t1_newell as tn
import t2_persist as tp
from t2 import statics

REL = cache.REL
OUT = L.ARCH_C
Q_PANELS = cache.PANELS[:8]
DT_H = tn.DT_H


def fd_wave(panel: str, links) -> dict:
    fd = tn.fd_authority(panel, links)
    return dict(
        x=fd.x_km.to_numpy(),
        w=np.abs(fd.w_kmh.to_numpy()),
        vcut=0.6 * fd.free_speed_kmh.to_numpy(),
        cap=fd.capacity_vph.to_numpy(),
        lanes=fd.lanes.to_numpy(),
        vf=fd.free_speed_kmh.to_numpy(),
    )


def shrink(q: np.ndarray, x: np.ndarray, dx: float) -> np.ndarray:
    """Clear from the upstream front: queued cells within dx of the upstream edge go free."""
    if not q.any() or dx <= 0:
        return q.copy()
    xs = x[q]
    up = xs.min()
    out = q.copy()
    out &= ~((x >= up) & (x < up + dx) & q)
    return out


def grow(q: np.ndarray, x: np.ndarray, dx: float) -> np.ndarray:
    """Grow upstream of the current upstream front."""
    if dx <= 0:
        return q.copy()
    out = q.copy()
    if not q.any():
        return out
    up = x[q].min()
    out |= (x >= up - dx) & (x < up)
    return out


def rollout(q0: np.ndarray, x: np.ndarray, w_abs: np.ndarray, mode: str) -> np.ndarray:
    """q0 [L] -> [6, L] horizon. Wave speed is the mean |w| of currently queued links."""
    q = q0.astype(bool).copy()
    steps = []
    for _ in range(6):
        w = float(np.mean(w_abs[q])) if q.any() else float(np.mean(w_abs))
        dx = w * DT_H
        if mode == "clear":
            q = shrink(q, x, dx)
        elif mode == "grow":
            q = grow(q, x, dx)
        steps.append(q.copy())
    return np.stack(steps, 0)


def gate(v_hist, q_hist, vcut, cap) -> str:
    """History-only persist/clear/grow decision. v_hist/q_hist: [12, L]."""
    queued = np.nan_to_num(v_hist, nan=1e9) <= vcut[None, :]
    n_now = int(queued[-1].sum())
    n_prev = int(queued[-4].sum()) if queued.shape[0] >= 4 else int(queued[0].sum())
    flow = np.where(np.isfinite(q_hist), q_hist, np.nan)
    if n_now == 0:
        return "persist"
    bn = np.where(queued[-1], np.arange(queued.shape[1]), -1).max()
    if bn < 0:
        return "persist"
    dis = float(np.nanmean(flow[-3:, bn])) if np.isfinite(flow[-3:, bn]).any() else 0.0
    up = slice(0, max(bn, 1))
    free_up = ~queued[-1, up]
    arr = float(np.nanmean(flow[-3:, up][:, free_up])) if free_up.any() else float(np.nanmean(flow[-3:, up]))
    cap_b = float(cap[bn]) if np.isfinite(cap[bn]) else arr
    if not np.isfinite(arr):
        arr = dis
    if dis > arr * 1.05 and n_now <= n_prev and dis > 0.5 * cap_b:
        return "clear"
    if arr > dis * 1.05 and n_now >= n_prev:
        return "grow"
    return "persist"


def iou(pred, truth, known) -> float:
    u = ((pred | truth) & known).sum()
    return 1.0 if u == 0 else float((pred & truth & known).sum() / u)


def evaluate() -> pd.DataFrame:
    rows = []
    for panel in Q_PANELS:
        z = cache.load(panel, "train")
        fd = fd_wave(panel, z["links"])
        vcut, cap, x, w = fd["vcut"], fd["cap"], fd["x"], fd["w"]
        dates = list(z["dates"])
        path = REL / "task2" / panel / "train" / "window_index.csv"
        if not path.exists():
            continue
        wi = pd.read_csv(path)
        wi = wi[wi.condition == "queue_ongoing"]
        for wrow in wi.itertuples():
            d = dates.index(wrow.date)
            s0 = tp.slot_of(wrow.forecast_origin)
            el = z["elig"][d].astype(bool)
            v_all = z["speed"][d, s0 - 12:s0]
            q_all = z["flow"][d, s0 - 12:s0]
            v_el = np.where(el[s0 - 12:s0], v_all, np.nan)
            ffa = pd.DataFrame(v_all).ffill().to_numpy()
            q0 = np.nan_to_num(ffa[-1], nan=1e9) <= vcut
            y = np.nan_to_num(z["speed"][d, s0 + 1:s0 + 7], nan=1e9) <= vcut[None, :]
            ylat = tp.fill(median_filter(
                np.nan_to_num(z["speed"][d], nan=1e9), size=(3, 1), mode="nearest",
            )[s0 + 1:s0 + 7] <= vcut[None, :])
            known = el[s0 + 1:s0 + 7] & np.isfinite(z["speed"][d, s0 + 1:s0 + 7])
            persist = np.repeat(q0[None], 6, 0)
            mode = gate(v_all, q_all, vcut, cap)
            pred = persist if mode == "persist" else rollout(q0, x, w, mode)
            rec = dict(panel=panel, window_id=wrow.date, mode=mode)
            rec["ffill_noisy"] = iou(persist, y, known)
            rec["ffill_latent"] = iou(persist, ylat, known)
            rec["moe_noisy"] = iou(pred, y, known)
            rec["moe_latent"] = iou(pred, ylat, known)
            rec["oracle_clear_noisy"] = iou(rollout(q0, x, w, "clear"), y, known)
            rec["oracle_grow_noisy"] = iou(rollout(q0, x, w, "grow"), y, known)
            rows.append(rec)
    df = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    if len(df):
        df.to_csv(OUT / "t2_moe_eval.csv", index=False)
        d_n = df.moe_noisy - df.ffill_noisy
        d_l = df.moe_latent - df.ffill_latent
        beat = (d_n.mean() > 0) and (d_l.mean() > 0)
        print(df["mode"].value_counts().to_string(), flush=True)
        print(
            f"ffill noisy={df.ffill_noisy.mean():.4f} latent={df.ffill_latent.mean():.4f}  "
            f"moe noisy={df.moe_noisy.mean():.4f} latent={df.moe_latent.mean():.4f}  "
            f"d_noisy={d_n.mean():+.4f} d_latent={d_l.mean():+.4f}  "
            f"SHIP={beat}",
            flush=True,
        )
        print("modes", df["mode"].value_counts().to_dict(), "gate_beats_ffill", beat, flush=True)
        if not beat:
            print("MoE does not beat ffill on both labels; keep J1 ffill_inel.", flush=True)
    else:
        print("no train ongoing windows on this machine", flush=True)
    return df


def selftest() -> None:
    L = 20
    x = np.cumsum(np.full(L, 0.4)) - 0.2
    w = np.full(L, 25.0)
    q0 = np.zeros(L, bool)
    q0[10:16] = True
    cl = rollout(q0, x, w, "clear")
    gr = rollout(q0, x, w, "grow")
    assert cl[-1].sum() < q0.sum(), (cl[-1].sum(), q0.sum())
    assert gr[-1].sum() > q0.sum()
    persist = rollout(q0, x, w, "persist")
    assert persist[-1].sum() == q0.sum()
    T = 12
    vcut = np.full(L, 0.6 * 115.0)
    cap = np.full(L, 8000.0)
    v_hist = np.full((T, L), 100.0)
    q_hist = np.full((T, L), 3000.0)
    v_hist[:, 10:16] = 30.0
    q_hist[:, 15] = 7000.0
    q_hist[:, :10] = 2000.0
    v_hist[-4:, 10] = 100.0
    assert gate(v_hist, q_hist, vcut, cap) in {"persist", "clear", "grow"}
    print("t2_moe selftest OK  clear", int(cl[-1].sum()), "grow", int(gr[-1].sum()), flush=True)


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "selftest"
    if cmd == "eval":
        evaluate()
    else:
        selftest()
