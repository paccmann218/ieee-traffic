"""Faithful Task 1 structural ablations on held-out train months.

Profiles are built from June-November only; December-February are never used in
profile construction or smoother tuning.  The released R1/R2/R3 target masks
are used directly and scores are reported by regime and family.
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


OUT = Path.home() / "tfb/work/t1_ablate"
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


def kalman_smooth(y: np.ndarray, q: float, r: float) -> np.ndarray:
    finite = np.flatnonzero(np.isfinite(y))
    if len(finite) == 0:
        return np.zeros(len(y))
    n = len(y)
    xf, pf, xp, pp = (np.zeros(n) for _ in range(4))
    state, variance = float(y[finite[0]]), r
    for t in range(n):
        if t:
            variance += q
        xp[t], pp[t] = state, variance
        if np.isfinite(y[t]):
            gain = variance / (variance + r)
            state += gain * (y[t] - state)
            variance *= 1.0 - gain
        xf[t], pf[t] = state, variance
    smooth = xf.copy()
    for t in range(n - 2, -1, -1):
        gain = pf[t] / max(pp[t + 1], 1e-12)
        smooth[t] = xf[t] + gain * (smooth[t + 1] - xp[t + 1])
    return smooth


def structural(z: dict, profile: np.ndarray, channel: str, qfac: float = 0.35, rfac: float = 0.8) -> np.ndarray:
    observed = tb.usable(z, channel)
    pred = profile.copy()
    for d in range(observed.shape[0]):
        deviation = observed[d] - profile[d]
        for link in range(observed.shape[2]):
            row = deviation[:, link]
            if not np.isfinite(row).any():
                continue
            scale = max(float(np.nanstd(row)), 1e-3)
            pred[d, :, link] += kalman_smooth(
                row, q=(qfac * scale) ** 2, r=(rfac * scale) ** 2
            )
    return np.clip(pred, 0.0, None)


def t3_subset(panel: str, zfull: dict, ztest: dict, keep: np.ndarray) -> sl.T3:
    obj = copy.copy(sl.T3(panel, zfull))
    obj.ramp_ok = obj.ramp_ok[keep]
    return obj


def run_panel(panel: str) -> list[dict]:
    z = cache.load(panel, "train")
    month = pd.to_datetime(z["dates"]).month.to_numpy()
    test = np.isin(month, TEST_MONTHS)
    ztrain, ztest = subset(z, ~test), subset(z, test)
    lanes = sl.lanes_t1(panel, z["links"])
    profile_v = tb.profile(ztrain, ztest, "speed")
    profile_q = tb.profile(ztrain, ztest, "flow")
    candidates = {
        "profile": (profile_v, profile_q),
        "tinterp": (tb.time_interp(ztest, "speed"), tb.time_interp(ztest, "flow")),
        "structural_kf": (
            structural(ztest, profile_v, "speed"),
            structural(ztest, profile_q, "flow"),
        ),
    }
    t3 = t3_subset(panel, z, ztest, test)
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
                "S_state": state[regime]["S"],
                "rmse_v": state[regime]["rmse_v"],
                "rmse_q_lane": state[regime]["rmse_q"],
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
    macro.to_csv(OUT / "regime_macro.csv")
    overall.to_csv(OUT / "overall.csv")
    print("\nBy regime:\n", macro.round(4).to_string())
    print("\nOverall:\n", overall.round(4).to_string())


if __name__ == "__main__":
    main()
