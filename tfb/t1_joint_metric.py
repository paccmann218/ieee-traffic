"""Joint Task 1 metric: S_state + mixed-field LWR, R3 isolated, near-window, D12_I405 canary.

This is the selection metric for Architecture C. Do not select on S_state alone.
Topology flux is never used as the conservation RHS; score_local.T3 uses ΔN of the
released observation field (organizer-flux equivalent on train).

Usage: python tfb/t1_joint_metric.py selftest | train [panel ...]
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, "tfb")
import cache
import legal as L
import score_local as sl
import t1_base as tb
import t1_near_window as nw

OUT = L.ARCH_C
REGIMES = sl.REGIMES
CANARY = "D12_I405"


def score_t1_masked(z: dict, pv, pq, lanes, cell_mask) -> dict:
    """Official S_state but only on target cells also in cell_mask."""
    z2 = dict(z)
    z2["target"] = z["target"].astype(bool) & np.asarray(cell_mask, bool)
    return sl.score_t1(z2, pv, pq, lanes)


def score_panel(panel: str, z: dict, pv, pq, near_mask=None) -> dict:
    lanes = sl.lanes_t1(panel, z["links"])
    s1 = sl.score_t1(z, pv, pq, lanes)
    t3 = sl.T3(panel, z)
    s3 = t3.score(z, pv, pq)
    row = {
        "panel": panel,
        "family": panel.rsplit("_", 1)[0],
        "S_state": float(s1["S_state"]),
        "S_physics": float(s3["S_physics"]),
        "J": float(L.paid_J(s1["S_state"], s3["S_physics"])),
        "is_d12_i405": panel.startswith(CANARY),
        "n_targets": int((z["target"].astype(bool) & np.isfinite(z["speed"]) & np.isfinite(z["flow"])).sum()),
    }
    for r in REGIMES:
        row[f"S_state_{r}"] = float(s1[r]["S"])
        row[f"rmse_v_{r}"] = float(s1[r].get("rmse_v", np.nan))
        row[f"rmse_q_{r}"] = float(s1[r].get("rmse_q", np.nan))
        row[f"S_FD_{r}"] = float(s3[r]["S_FD"])
        row[f"S_LWR_{r}"] = float(s3[r]["S_LWR"])
        row[f"S_phys_{r}"] = float(s3[r]["S_phys"])
        row[f"n_{r}"] = int(s1[r].get("n", 0))
    row["S_state_R3"] = row["S_state_R3"]
    row["S_LWR_R3"] = row["S_LWR_R3"]
    if near_mask is not None and near_mask.any():
        sn = score_t1_masked(z, pv, pq, lanes, near_mask)
        row["S_state_near"] = float(sn["S_state"])
        row["n_near"] = int((z["target"].astype(bool) & near_mask & np.isfinite(z["speed"])).sum())
        for r in REGIMES:
            row[f"S_state_near_{r}"] = float(sn[r]["S"])
    else:
        row["S_state_near"] = np.nan
        row["n_near"] = 0
    return row


def summarize(rows: pd.DataFrame) -> pd.DataFrame:
    """Unweighted panel mean, plus D12_I405 canary and R3-only view."""
    num = [c for c in rows.columns if c.startswith("S_") or c in ("J", "n_targets", "n_near")]
    out = []
    def add(name, sub):
        if sub.empty:
            return
        rec = {c: float(sub[c].mean()) for c in num if c in sub}
        rec["slice"] = name
        rec["n_panels"] = len(sub)
        out.append(rec)
    add("all", rows)
    add("d12_i405_canary", rows[rows.is_d12_i405])
    add("not_d12_i405", rows[~rows.is_d12_i405])
    return pd.DataFrame(out)


def gate(base: dict, cand: dict, j_tol: float = 0.0) -> dict:
    """Architecture C ships only if paid J does not fall and LWR rises (or J rises)."""
    dJ = cand["J"] - base["J"]
    dLWR = cand.get("S_LWR_R3", cand.get("S_physics", 0)) - base.get("S_LWR_R3", base.get("S_physics", 0))
    dS = cand["S_state"] - base["S_state"]
    # Identity (alpha=0) is numerical noise. A probe needs a real paid-J lift and higher LWR.
    ok = (dJ >= 0.005) and (dLWR > 0.01)
    return dict(dJ=dJ, dLWR=dLWR, dS_state=dS, pass_gate=bool(ok))


def train(panels=None) -> pd.DataFrame:
    panels = list(panels) if panels else list(cache.PANELS)
    rows = []
    for panel in panels:
        zpath = cache.OUT / f"{panel}_train.npz"
        if not zpath.exists():
            print(f"skip {panel}: no cache at {zpath}", flush=True)
            continue
        z = cache.load(panel, "train")
        pv = tb.time_interp(z, "speed")
        pq = tb.time_interp(z, "flow")
        near, _hole = nw.near_window_mask(panel, z, "train")
        row = score_panel(panel, z, pv, pq, near)
        row["model"] = "tinterp"
        rows.append(row)
        print(panel, f"S_state={row['S_state']:.4f}", f"S_phys={row['S_physics']:.4f}",
              f"J={row['J']:.4f}", f"near={row['S_state_near']}", flush=True)
    df = pd.DataFrame(rows)
    if df.empty:
        print("no train caches; skip joint tinterp score")
        return df
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / "joint_tinterp.csv", index=False)
    print(summarize(df).round(4).to_string(index=False), flush=True)
    return df


def selftest() -> None:
    D, T, L = 4, 24, 8
    rng = np.random.default_rng(0)
    v = 90 + rng.normal(0, 2, (D, T, L))
    q = 4000 + rng.normal(0, 40, (D, T, L))
    elig = np.ones((D, T, L), np.int8)
    target = np.zeros((D, T, L), np.int8)
    target[:, 4:8, 2:6] = 1
    z = dict(
        links=np.array([f"L{i}" for i in range(L)]),
        dates=np.array(["2030-06-0%d" % (i + 1) for i in range(D)]),
        speed=v.astype(np.float32), flow=q.astype(np.float32),
        mspeed=v.astype(np.float32), mflow=q.astype(np.float32),
        elig=elig, target=target,
        regime=np.array(["R1", "R2", "R3", "R1"]),
        occ=np.full((D, T, L), 10.0, np.float32),
        pct=np.full((D, T, L), 100, np.int16),
        relig=np.zeros((D, T, 1), np.int8), rflow=np.zeros((D, T, 1), np.float32),
        ramp_ids=np.array(["R0"]), ramp_type=np.array(["OR"]),
    )
    lanes = np.full(L, 4.0)
    s1 = sl.score_t1(z, v, q, lanes)
    assert s1["S_state"] > 0.999, s1
    near = np.zeros_like(target, bool)
    near[:, 3:5, :] = True
    sn = score_t1_masked(z, v, q, lanes, near)
    assert sn["S_state"] >= 0.0
    g = gate(dict(J=0.40, S_state=0.93, S_physics=0.64, S_LWR_R3=0.50),
             dict(J=0.41, S_state=0.93, S_physics=0.70, S_LWR_R3=0.62))
    assert g["pass_gate"]
    g2 = gate(dict(J=0.40, S_state=0.93, S_physics=0.64, S_LWR_R3=0.50),
              dict(J=0.401, S_state=0.931, S_physics=0.65, S_LWR_R3=0.505))
    assert not g2["pass_gate"]
    g3 = gate(dict(J=0.40, S_state=0.93, S_physics=0.64, S_LWR_R3=0.50),
              dict(J=0.40, S_state=0.93, S_physics=0.64, S_LWR_R3=0.50001))
    assert not g3["pass_gate"]
    print("t1_joint_metric selftest OK", flush=True)


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "selftest"
    if cmd == "train":
        train(sys.argv[2:] or None)
    else:
        selftest()
