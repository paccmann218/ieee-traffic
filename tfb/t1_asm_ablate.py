"""Standalone Adaptive Smoothing reconstructions, including FD wave speeds.

This is the cheap graph/PDE-style Task 1 gate: reconstruct from the masked
layer only, no GBDT, profiles from June-November only.  The official structural
Kalman smoother already lost to time interpolation in ``t1_ablate.py``.  If ASM
does not beat interpolation on both S_state and local S_physics, the remaining
graph-smoother branch is closed.
"""
from __future__ import annotations

import copy
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, "tfb")
import cache
import score_local as sl
import t1_base as tb
import t1_features as tf


OUT = Path.home() / "tfb/work/t1_asm"
TEST_MONTHS = (12, 1, 2)


def subset(z: dict, keep: np.ndarray) -> dict:
    days = len(z["dates"])
    return {
        key: (
            value[keep]
            if isinstance(value, np.ndarray)
            and value.ndim >= 1
            and value.shape[0] == days
            and key not in ("links", "ramp_ids", "ramp_type")
            else value
        )
        for key, value in z.items()
    }


def fd_waves(panel: str, links: list[str]) -> tuple[float, float]:
    fd = pd.read_csv(
        cache.REL / "corridors" / panel / "network" / "fd_parameters.csv",
        dtype={"link_id": str},
    ).drop_duplicates("link_id").set_index("link_id").reindex(links)
    vf = pd.to_numeric(fd.free_speed_kmh, errors="coerce")
    cap = pd.to_numeric(fd.capacity_vph, errors="coerce")
    kc = pd.to_numeric(fd.critical_density, errors="coerce")
    kj = pd.to_numeric(fd.k_jam, errors="coerce")
    wave = cap / np.maximum(kj - kc, 1.0)
    return float(np.nanmean(vf)), -float(np.nanmean(wave))


def asm_fill(z: dict, xpos: np.ndarray, c_free: float, c_cong: float) -> tuple[np.ndarray, np.ndarray]:
    fallback_v = tb.time_interp(z, "speed")
    fallback_q = tb.time_interp(z, "flow")
    speed = fallback_v.copy()
    flow = fallback_q.copy()
    observed_v = tb.usable(z, "speed")
    observed_q = tb.usable(z, "flow")
    for day in range(observed_v.shape[0]):
        va, qa, _, _ = tf.asm(observed_v[day], observed_q[day], xpos, c_free=c_free, c_cong=c_cong)
        speed[day] = np.where(np.isfinite(va), va, fallback_v[day])
        flow[day] = np.where(np.isfinite(qa), qa, fallback_q[day])
    return speed, flow


def run_panel(panel: str) -> list[dict]:
    z = cache.load(panel, "train")
    month = pd.to_datetime(z["dates"]).month.to_numpy()
    keep = np.isin(month, TEST_MONTHS)
    ztest = subset(z, keep)
    links = list(z["links"])
    xpos, _ = tf.positions(panel, links)
    c_free, c_cong = fd_waves(panel, links)
    candidates = {
        "tinterp": (tb.time_interp(ztest, "speed"), tb.time_interp(ztest, "flow")),
        "asm_default": asm_fill(ztest, xpos, 100.0, -18.0),
        "asm_fd": asm_fill(ztest, xpos, c_free, c_cong),
    }
    lanes = sl.lanes_t1(panel, links)
    t3 = copy.copy(sl.T3(panel, z))
    t3.ramp_ok = t3.ramp_ok[keep]
    rows = []
    for name, (speed, flow) in candidates.items():
        state = sl.score_t1(ztest, speed, flow, lanes)
        physics = t3.score(ztest, speed, flow)
        for regime in sl.REGIMES:
            rows.append({
                "panel": panel,
                "family": panel.rsplit("_", 1)[0],
                "model": name,
                "regime": regime,
                "c_free": c_free,
                "c_cong": c_cong,
                "S_state": state[regime]["S"],
                "S_physics_local": physics[regime]["S_phys"],
                "S_LWR_local": physics[regime]["S_LWR"],
            })
    return rows


def main() -> None:
    rows = []
    for panel in cache.PANELS:
        rows.extend(run_panel(panel))
        print(panel, "done", flush=True)
    detail = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    detail.to_csv(OUT / "detail.csv", index=False)
    family = detail.groupby(["model", "regime", "family"])[
        ["S_state", "S_physics_local", "S_LWR_local"]
    ].mean()
    macro = family.groupby(["model", "regime"]).mean()
    overall = macro.groupby("model").mean()
    overall["J_local"] = 0.35 * overall.S_state + 0.15 * overall.S_physics_local
    print(detail.groupby("panel")[["c_free", "c_cong"]].first().round(2).to_string())
    print("\nBy regime:\n", macro.round(4).to_string())
    print("\nOverall:\n", overall.round(4).to_string())


if __name__ == "__main__":
    main()
