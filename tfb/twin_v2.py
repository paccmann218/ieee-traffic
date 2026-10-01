"""Capacity-recovery CTM ensemble for legal Task 2 ongoing forecasts.

This extends :mod:`twin` without changing its recorded v0/v1 baseline.  The
main failure of v0 was permanently holding the measured incident discharge
capacity for all 30 forecast minutes.  Here that interface capacity can recover,
while boundary demand and ramp inflow are varied in a small train-independent
ensemble.  Legal forward-fill is always included as a fallback action.

The script reports:
  expected   empirical-expected-IoU action under the model ensemble (no labels)
  lodo       one fixed ensemble member selected leave-one-date-out
  oracle     best ensemble member per window (representation upper bound only)
"""
from __future__ import annotations

import copy
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.ndimage import median_filter

sys.path.insert(0, "tfb")
import cache
import t2_persist as tp
import twin
from t2 import statics


OUT = Path.home() / "tfb/work/twin_v2"
HALF_LIVES = (0.0, 10.0, 20.0, 40.0, np.inf)
DEMAND_STRENGTHS = (0.0, 0.5, 1.0)
RAMP_MULTIPLIERS = (0.8, 1.0, 1.2)


def iou(pred: np.ndarray, truth: np.ndarray, known: np.ndarray) -> float:
    u = ((pred | truth) & known).sum()
    return 1.0 if u == 0 else float((pred & truth & known).sum() / u)


def run_recovery(
    corr: twin.Corridor,
    st: dict,
    demand_trend: np.ndarray,
    half_life_min: float,
    demand_strength: float,
    ramp_multiplier: float,
    nslots: int = 6,
) -> np.ndarray:
    k = st["k"].copy()
    length = len(k)
    vf, cap, kj, wave = corr.vf, corr.cap, corr.kj, corr.w
    base_icap = np.minimum(cap[:-1], cap[1:])
    speeds = np.zeros((nslots, length))
    for slot in range(nslots):
        if half_life_min == 0:
            recovered = 1.0
        elif np.isinf(half_life_min):
            recovered = 0.0
        else:
            recovered = 1.0 - 0.5 ** (((slot + 1) * 5.0) / half_life_min)
        icap = st["icap"] + recovered * (base_icap - st["icap"])
        trend = np.clip(1.0 + demand_strength * (demand_trend[slot] - 1.0), 0.5, 1.5)
        on = st["on"] * ramp_multiplier * trend
        demand = st["demand"] * trend
        vacc = np.zeros(length)
        for _ in range(corr.nsub):
            sending = np.minimum(vf * k, cap)
            receiving = np.minimum(cap, wave * np.maximum(kj - k, 0))
            send_main = sending[:-1] * (1 - st["beta"][:-1])
            recv = np.minimum(receiving[1:], icap)
            mainline = np.minimum(send_main, recv)
            ramp = np.minimum(on[1:], np.maximum(receiving[1:] - mainline, 0))
            in0 = min(demand, receiving[0]) + min(on[0], max(receiving[0] - demand, 0))
            inflow = np.r_[in0, mainline + ramp]
            outflow = np.r_[mainline / np.maximum(1 - st["beta"][:-1], 1e-6), sending[-1]]
            k = np.clip(k + corr.dt_h / corr.len * (inflow - outflow), 0, kj)
            vacc += np.where(
                k <= corr.kc,
                vf,
                wave * np.maximum(kj - k, 0) / np.maximum(k, 1e-6),
            )
        speeds[slot] = vacc / corr.nsub
    return speeds


def legal_history(z: dict, d: int, s0: int, include_ineligible: bool) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    speed_name = "speed" if "speed" in z else "mspeed"
    flow_name = "flow" if "flow" in z else "mflow"
    speed, flow = z[speed_name][d], z[flow_name][d]
    if include_ineligible:
        v = speed[s0 - 12:s0].copy()
        q = flow[s0 - 12:s0].copy()
    else:
        eligible = z["elig"][d].astype(bool)
        v = np.where(eligible[s0 - 12:s0], speed[s0 - 12:s0], np.nan)
        q = np.where(eligible[s0 - 12:s0], flow[s0 - 12:s0], np.nan)
    ramps = np.where(
        z["relig"][d, s0 - 12:s0].astype(bool),
        z["rflow"][d, s0 - 12:s0],
        np.nan,
    )
    return v, q, ramps


def demand_trend(q_hist: np.ndarray) -> np.ndarray:
    upstream = pd.DataFrame(q_hist[:, :1]).ffill().to_numpy()[:, 0]
    if not np.isfinite(upstream).any():
        return np.ones(6)
    upstream = pd.Series(upstream).interpolate(limit_direction="both").to_numpy()
    recent = np.mean(upstream[-3:])
    prior = np.mean(upstream[-6:-3])
    step = np.clip((recent - prior) / max(abs(recent), 1.0), -0.15, 0.15)
    return np.clip(1.0 + step * np.arange(1, 7) / 3.0, 0.6, 1.4)


def ensemble(corr: twin.Corridor, z: dict, d: int, s0: int) -> list[tuple[str, np.ndarray]]:
    named: list[tuple[str, np.ndarray]] = []
    for include_ineligible in (False, True):
        v, q, ramps = legal_history(z, d, s0, include_ineligible)
        st = corr.calibrate(v, q, ramps)
        trend = demand_trend(q)
        evidence = "all" if include_ineligible else "eligible"
        for half_life in HALF_LIVES:
            htag = "inf" if np.isinf(half_life) else str(int(half_life))
            for demand_strength in DEMAND_STRENGTHS:
                for ramp_multiplier in RAMP_MULTIPLIERS:
                    speed = run_recovery(
                        corr, st, trend, half_life, demand_strength, ramp_multiplier
                    )
                    mask = speed <= corr.vcut[None, :]
                    name = f"{evidence}_hl{htag}_d{demand_strength:g}_r{ramp_multiplier:g}"
                    named.append((name, mask))
    # Legal carry-forward is an action even if no CTM member reproduces it.
    v_all, _, _ = legal_history(z, d, s0, True)
    q0 = tp.variants(v_all, v_all, corr.vcut)["ffill_inel"]
    named.append(("ffill", np.repeat(q0[None], 6, axis=0)))
    return named


def expected_action(members: list[tuple[str, np.ndarray]], known: np.ndarray) -> tuple[str, np.ndarray]:
    values = []
    for name, action in members:
        values.append(np.mean([iou(action, outcome, known) for _, outcome in members]))
    return members[int(np.argmax(values))]


def add_lodo(frame: pd.DataFrame, candidate_names: list[str]) -> pd.DataFrame:
    out = frame.copy()
    for label in ("noisy", "latent"):
        chosen, values = [], []
        cols = [f"{name}|{label}" for name in candidate_names]
        for row in frame.itertuples():
            train = frame.date != row.date
            means = frame.loc[train, cols].mean(axis=0)
            col = str(means.idxmax())
            chosen.append(col.split("|")[0])
            values.append(float(frame.loc[frame.window_id == row.window_id, col].iloc[0]))
        out[f"lodo_{label}_name"] = chosen
        out[f"lodo_{label}"] = values
    return out


def clustered(frame: pd.DataFrame, col: str, ref: str) -> tuple[float, float]:
    delta = frame[col] - frame[ref]
    group = pd.DataFrame({"date": frame.date, "delta": delta}).groupby("date").delta.mean()
    return float(delta.mean()), float(group.std(ddof=1) / np.sqrt(len(group)))


def evaluate() -> pd.DataFrame:
    rows = []
    candidate_names: set[str] = set()
    for panel in tp.Q_PANELS:
        z = cache.load(panel, "train")
        corr = twin.Corridor(panel, z)
        dates = list(z["dates"])
        wi = pd.read_csv(tp.REL / "task2" / panel / "train" / "window_index.csv")
        for w in wi[wi.condition == "queue_ongoing"].itertuples():
            d = dates.index(w.date)
            s0 = tp.slot_of(w.forecast_origin)
            members = ensemble(corr, z, d, s0)
            candidate_names.update(name for name, _ in members)
            speed = z["speed"][d]
            known = z["elig"][d, s0 + 1:s0 + 7].astype(bool) & np.isfinite(speed[s0 + 1:s0 + 7])
            noisy = np.nan_to_num(speed[s0 + 1:s0 + 7], nan=1e9) <= corr.vcut[None, :]
            smooth = median_filter(np.nan_to_num(speed, nan=1e9), size=(3, 1), mode="nearest")
            latent = tp.fill(smooth[s0 + 1:s0 + 7] <= corr.vcut[None, :])
            exp_name, exp_mask = expected_action(members, known)
            rec = {
                "panel": panel,
                "date": w.date,
                "window_id": w.window_id,
                "n_members": len(members),
                "expected_name": exp_name,
            }
            for label, truth in (("noisy", noisy), ("latent", latent)):
                scores = {}
                for name, mask in members:
                    scores[name] = iou(mask, truth, known)
                    rec[f"{name}|{label}"] = scores[name]
                rec[f"expected_{label}"] = iou(exp_mask, truth, known)
                rec[f"oracle_{label}"] = max(scores.values())
            rows.append(rec)
    frame = pd.DataFrame(rows)
    names = sorted(candidate_names)
    frame = add_lodo(frame, names)
    OUT.mkdir(parents=True, exist_ok=True)
    frame.to_csv(OUT / "windows.csv", index=False)
    rows_out = []
    for method in ("expected", "lodo", "oracle"):
        rec = {"method": method, "n": len(frame), "n_dates": frame.date.nunique()}
        for label in ("noisy", "latent"):
            col, ref = f"{method}_{label}", f"ffill|{label}"
            rec[label] = frame[col].mean()
            rec[f"delta_{label}"], rec[f"se_date_{label}"] = clustered(frame, col, ref)
        rows_out.append(rec)
    summary = pd.DataFrame(rows_out)
    summary.to_csv(OUT / "summary.csv", index=False)
    print(summary.round(4).to_string(index=False))
    print("Expected-action choices:", frame.expected_name.value_counts().to_dict())
    print("LODO latent choices:", frame.lodo_latent_name.value_counts().to_dict())
    return frame


if __name__ == "__main__":
    evaluate()
