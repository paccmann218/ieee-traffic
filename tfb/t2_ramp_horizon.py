"""Task 2 horizon-ramp legality and train-only ablation.

History ramp sums are already in the J1 onset analogue (t2_joint_prior.history_features).
The unused signal is ramp flow *inside* the forecast horizon [T+5, T+30], which is
~79.5% finite while mainline horizon is 100% blank.

Using contemporaneous horizon ramps as a queue covariate is a contract gray area
(exogenous demand vs system state). SHIP_HORIZON_RAMPS is False. This module never
writes a submission. It only measures whether the signal exists on train labels.

Usage: python tfb/t2_ramp_horizon.py selftest | train
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, "tfb")
import cache
import legal as L
import t2_persist as tp
from t2 import statics

REL = cache.REL
OUT = L.ARCH_C
Q_PANELS = cache.PANELS[:8]
SHIP = L.SHIP_HORIZON_RAMPS


def horizon_ramp_features(z: dict, d: int, s0: int, cap: np.ndarray) -> dict:
    """Ramp flow in the 6-step horizon. Train-only diagnostic; not a shipped feature."""
    h = slice(s0 + 1, s0 + 7)
    rf = np.where(z["relig"][d, h].astype(bool), z["rflow"][d, h], np.nan)
    cap_sum = max(float(np.nansum(cap)), 1.0)
    per_step = np.nansum(rf, axis=1) / cap_sum if rf.size else np.zeros(6)
    finite = float(np.isfinite(z["rflow"][d, h]).mean()) if rf.size else 0.0
    return dict(
        ramp_h_mean=float(np.nanmean(per_step)) if np.isfinite(per_step).any() else 0.0,
        ramp_h_first=float(per_step[0]) if per_step.size else 0.0,
        ramp_h_last=float(per_step[-1]) if per_step.size else 0.0,
        ramp_h_delta=float(per_step[-1] - per_step[0]) if per_step.size else 0.0,
        ramp_h_finite=finite,
    )


def history_ramp_features(z: dict, d: int, s0: int, cap: np.ndarray) -> dict:
    """Legal history ramps already used by J1 analogue."""
    h = slice(s0 - 12, s0)
    rf = np.where(z["relig"][d, h].astype(bool), z["rflow"][d, h], np.nan)
    cap_sum = max(float(np.nansum(cap)), 1.0)
    per_step = np.nansum(rf, axis=1) / cap_sum if rf.size else np.zeros(12)
    return dict(
        ramp_hist_mean=float(np.nanmean(per_step)) if np.isfinite(per_step).any() else 0.0,
        ramp_hist_last=float(per_step[-1]) if per_step.size else 0.0,
    )


def train() -> pd.DataFrame:
    if SHIP:
        raise SystemExit("SHIP_HORIZON_RAMPS is True; refuse to silently enable a gray-area feature")
    rows = []
    for panel in Q_PANELS:
        z = cache.load(panel, "train")
        st = statics(panel, z["links"])
        vcut = st["vcut"]
        cap = st["cap"]
        dates = list(z["dates"])
        path = REL / "task2" / panel / "train" / "window_index.csv"
        if not path.exists():
            continue
        wi = pd.read_csv(path)
        for w in wi.itertuples():
            d = dates.index(w.date)
            s0 = tp.slot_of(w.forecast_origin)
            y = np.nan_to_num(z["speed"][d, s0 + 1:s0 + 7], nan=1e9) <= vcut[None, :]
            nq = y.sum(axis=1).astype(float)
            growth = float(nq[-1] - nq[0])
            rec = dict(panel=panel, window_id=w.window_id, condition=w.condition,
                       nq0=float(nq[0]), nq6=float(nq[-1]), growth=growth)
            rec.update(horizon_ramp_features(z, d, s0, cap))
            rec.update(history_ramp_features(z, d, s0, cap))
            rows.append(rec)
    df = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / "ramp_horizon_ablate.csv", index=False)
    print("LEGALITY: horizon ramps are gray-area; SHIP_HORIZON_RAMPS=False; no submission uses them.")
    print("History ramps are already in J1 onset analogue.")
    if len(df) == 0:
        print("no train windows on this machine")
        return df
    for cond, sub in df.groupby("condition"):
        c_h = sub[["growth", "ramp_h_mean", "ramp_h_delta"]].corr().loc["growth"]
        c_hist = sub[["growth", "ramp_hist_mean"]].corr().loc["growth", "ramp_hist_mean"]
        print(cond, "n=", len(sub), "finite_h=", round(sub.ramp_h_finite.mean(), 3),
              "corr growth~horizon_mean", round(float(c_h["ramp_h_mean"]), 3),
              "corr growth~history_mean", round(float(c_hist), 3), flush=True)
    print("Decision: do not ship. Ask host before any LB probe that reads ramps at T+5..T+30.")
    return df


def selftest() -> None:
    assert SHIP is False
    z = dict(
        relig=np.ones((1, 288, 3), np.int8),
        rflow=np.zeros((1, 288, 3), np.float32),
        speed=np.full((1, 288, 4), 80.0, np.float32),
    )
    z["rflow"][0, 100:112, :] = 200.0
    z["rflow"][0, 113:119, :] = np.linspace(100, 400, 6)[:, None]
    cap = np.array([8000.0, 8000.0, 8000.0, 8000.0])
    h = history_ramp_features(z, 0, 112, cap)
    r = horizon_ramp_features(z, 0, 112, cap)
    assert r["ramp_h_finite"] > 0.5
    assert r["ramp_h_delta"] != 0.0
    assert h["ramp_hist_mean"] > 0
    print("t2_ramp_horizon selftest OK  (will not ship)", r, flush=True)


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "selftest"
    if cmd == "train":
        train()
    else:
        selftest()
