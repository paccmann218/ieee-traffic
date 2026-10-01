"""Architecture C runner: joint metric, Newell fill, conservation projection, near-window.

Wave 1 of the research plan. Does not retune J1, does not submit, does not touch Task 4.

Usage:
  python tfb/arch_c_run.py selftest
  python tfb/arch_c_run.py train [panel ...]
  python tfb/arch_c_run.py all
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, "tfb")
import cache
import freeze_j1
import legal as L
import t1_base as tb
import t1_joint_metric as jm
import t1_near_window as nw
import t1_newell as tn
import t1_occ_ablate as occ
import t1_project as tpj
import t1_r3_imputer as r3
import t2_moe
import t2_ramp_horizon as rh
import build_c_probe as probe

OUT = L.ARCH_C
ALPHAS = tpj.ALPHAS


def selftest_all() -> None:
    freeze_j1.selftest()
    jm.selftest()
    nw.selftest()
    tn.selftest()
    tpj.selftest()
    occ.selftest()
    rh.selftest()
    t2_moe.selftest()
    r3.selftest()
    probe.selftest()
    print("arch_c_run: all selftests OK", flush=True)


def eval_panel(panel: str, z: dict, alpha: float) -> dict:
    prior_v = tb.time_interp(z, "speed")
    prior_q = tb.time_interp(z, "flow")
    near, hole = nw.near_window_mask(panel, z, "train")
    disable = nw.disable_tinterp_mask(panel, z, "train")
    # Where tinterp is invalid, replace the prior with Newell-only (spatial).
    Vn, Qn, fd, meta_n = tn.newell_fill_split(panel, z, prior_v, prior_q)
    prior_v2 = np.where(disable, Vn, prior_v)
    prior_q2 = np.where(disable, Qn, prior_q)
    Vc, Qc, fd, meta = tpj.arch_c_split(panel, z, prior_v2, prior_q2, alpha=alpha)
    base = jm.score_panel(panel, z, prior_v, prior_q, near)
    cand = jm.score_panel(panel, z, Vc, Qc, near)
    row = dict(panel=panel, alpha=alpha)
    for k, v in base.items():
        if k != "panel":
            row[f"base_{k}"] = v
    for k, v in cand.items():
        if k != "panel":
            row[f"cand_{k}"] = v
    g = jm.gate(
        dict(J=base["J"], S_state=base["S_state"], S_physics=base["S_physics"], S_LWR_R3=base["S_LWR_R3"]),
        dict(J=cand["J"], S_state=cand["S_state"], S_physics=cand["S_physics"], S_LWR_R3=cand["S_LWR_R3"]),
    )
    row.update(g)
    row.update({f"newell_{k}": meta[k] for k in meta})
    row["n_disable"] = int(disable.sum())
    row["n_near"] = int(near.sum())
    row["n_hole"] = int(hole.sum())
    return row


def train(panels=None) -> pd.DataFrame:
    freeze_j1.check()
    panels = list(panels) if panels else list(cache.PANELS)
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for panel in panels:
        zpath = cache.OUT / f"{panel}_train.npz"
        if not zpath.exists():
            print(f"skip {panel}: no cache at {zpath}", flush=True)
            continue
        z = cache.load(panel, "train")
        best = None
        for alpha in ALPHAS:
            row = eval_panel(panel, z, alpha)
            rows.append(row)
            print(
                panel, f"α={alpha}",
                f"J {row['base_J']:.4f}->{row['cand_J']:.4f}",
                f"state {row['base_S_state']:.4f}->{row['cand_S_state']:.4f}",
                f"phys {row['base_S_physics']:.4f}->{row['cand_S_physics']:.4f}",
                f"near {row.get('base_S_state_near')}->{row.get('cand_S_state_near')}",
                f"gate={row['pass_gate']}",
                flush=True,
            )
            if best is None or row["cand_J"] > best["cand_J"]:
                best = row
        if best is not None:
            print(panel, "best_alpha", best["alpha"], "gate", best["pass_gate"], flush=True)
    df = pd.DataFrame(rows)
    if df.empty:
        print("no train caches; ran nothing. selftest still covers the algorithms.")
        return df
    df.to_csv(OUT / "arch_c_train.csv", index=False)
    # Pick one alpha: mean J across panels.
    pick = df.groupby("alpha")["cand_J"].mean().idxmax()
    sub = df[df.alpha == pick]
    gate_ok = bool(sub.pass_gate.all()) if "pass_gate" in sub else False
    # Panel-mean gate vs tinterp at the picked alpha.
    g = jm.gate(
        dict(J=float(sub.base_J.mean()), S_state=float(sub.base_S_state.mean()),
             S_physics=float(sub.base_S_physics.mean()), S_LWR_R3=float(sub.base_S_LWR_R3.mean())),
        dict(J=float(sub.cand_J.mean()), S_state=float(sub.cand_S_state.mean()),
             S_physics=float(sub.cand_S_physics.mean()), S_LWR_R3=float(sub.cand_S_LWR_R3.mean())),
    )
    metrics = pd.DataFrame([{
        "picked_alpha": float(pick),
        "base_J": float(sub.base_J.mean()),
        "cand_J": float(sub.cand_J.mean()),
        "base_S_state": float(sub.base_S_state.mean()),
        "cand_S_state": float(sub.cand_S_state.mean()),
        "base_S_physics": float(sub.base_S_physics.mean()),
        "cand_S_physics": float(sub.cand_S_physics.mean()),
        "base_S_state_near": float(sub.base_S_state_near.mean()) if "base_S_state_near" in sub else np.nan,
        "cand_S_state_near": float(sub.cand_S_state_near.mean()) if "cand_S_state_near" in sub else np.nan,
        "base_S_LWR_R3": float(sub.base_S_LWR_R3.mean()),
        "cand_S_LWR_R3": float(sub.cand_S_LWR_R3.mean()),
        **g,
        "n_panels": int(sub.panel.nunique()),
    }])
    metrics.to_csv(OUT / "metrics.csv", index=False)
    print("picked α", pick, "gate", g, flush=True)
    print(metrics.round(4).to_string(index=False), flush=True)
    if g["pass_gate"]:
        print("GATE PASS: isolated LB state probe (sub_C0) is allowed via build_c_probe.py")
    else:
        print("GATE FAIL: do not submit. Keep J1. Inspect near-window and R3 slices.")
    return df


def subset_days(z: dict, keep: np.ndarray) -> dict:
    days = len(z["dates"])
    skip = {"links", "ramp_ids", "ramp_type"}
    return {
        key: (
            value[keep]
            if isinstance(value, np.ndarray) and value.ndim >= 1 and value.shape[0] == days and key not in skip
            else value
        )
        for key, value in z.items()
    }


def gbdt_field(panel: str, z: dict):
    """Dec–Feb GBDT predictions on targets; time interpolation elsewhere. None if no CV file."""
    path = L.WORK / "t1feat" / "cv_preds.parquet"
    if not path.exists():
        return None
    preds = pd.read_parquet(path)
    pi = cache.PANELS.index(panel)
    sub = preds[preds.panel == pi]
    if sub.empty:
        return None
    month = pd.to_datetime(z["dates"]).month.to_numpy()
    keep = np.isin(month, [12, 1, 2])
    zz = subset_days(z, keep)
    prior_v = tb.time_interp(zz, "speed")
    prior_q = tb.time_interp(zz, "flow")
    dmap = {int(d): i for i, d in enumerate(np.flatnonzero(keep))}
    dd = sub.d.map(dmap)
    good = dd.notna()
    if not good.any():
        return None
    d = dd[good].astype(int).to_numpy()
    t = sub.loc[good, "t"].to_numpy(int)
    l = sub.loc[good, "l"].to_numpy(int)
    prior_v = prior_v.copy()
    prior_q = prior_q.copy()
    prior_v[d, t, l] = sub.loc[good, "pv"].to_numpy(np.float32)
    prior_q[d, t, l] = sub.loc[good, "pq"].to_numpy(np.float32)
    return zz, prior_v, prior_q


def eval_blend(panel: str, z: dict, prior_v, prior_q, alpha: float) -> dict:
    """Blend physics fill toward an untouched prior. alpha=0 reproduces the prior."""
    near, hole = nw.near_window_mask(panel, z, "train")
    Vc, Qc, fd, meta = tpj.arch_c_split(panel, z, prior_v, prior_q, alpha=alpha)
    base = jm.score_panel(panel, z, prior_v, prior_q, near)
    cand = jm.score_panel(panel, z, Vc, Qc, near)
    row = dict(panel=panel, alpha=alpha, prior="gbdt")
    for k, v in base.items():
        if k != "panel":
            row[f"base_{k}"] = v
    for k, v in cand.items():
        if k != "panel":
            row[f"cand_{k}"] = v
    g = jm.gate(
        dict(J=base["J"], S_state=base["S_state"], S_physics=base["S_physics"], S_LWR_R3=base["S_LWR_R3"]),
        dict(J=cand["J"], S_state=cand["S_state"], S_physics=cand["S_physics"], S_LWR_R3=cand["S_LWR_R3"]),
    )
    row.update(g)
    row.update({f"newell_{k}": meta[k] for k in meta})
    row["n_near"] = int(near.sum())
    row["n_hole"] = int(hole.sum())
    return row


def train_gbdt(panels=None) -> pd.DataFrame:
    freeze_j1.check()
    panels = list(panels) if panels else list(cache.PANELS)
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for panel in panels:
        zpath = cache.OUT / f"{panel}_train.npz"
        if not zpath.exists():
            print(f"skip {panel}: no cache", flush=True)
            continue
        z = cache.load(panel, "train")
        built = gbdt_field(panel, z)
        if built is None:
            print(f"skip {panel}: no cv_preds", flush=True)
            continue
        zz, prior_v, prior_q = built
        for alpha in ALPHAS:
            row = eval_blend(panel, zz, prior_v, prior_q, alpha)
            rows.append(row)
            print(
                panel, f"α={alpha}",
                f"J {row['base_J']:.4f}->{row['cand_J']:.4f}",
                f"state {row['base_S_state']:.4f}->{row['cand_S_state']:.4f}",
                f"phys {row['base_S_physics']:.4f}->{row['cand_S_physics']:.4f}",
                f"near {row.get('base_S_state_near')}->{row.get('cand_S_state_near')}",
                f"gate={row['pass_gate']}",
                flush=True,
            )
    df = pd.DataFrame(rows)
    if df.empty:
        print("no GBDT predictions")
        return df
    df.to_csv(OUT / "arch_c_gbdt.csv", index=False)
    pick = df.groupby("alpha")["cand_J"].mean().idxmax()
    sub = df[df.alpha == pick]
    g = jm.gate(
        dict(J=float(sub.base_J.mean()), S_state=float(sub.base_S_state.mean()),
             S_physics=float(sub.base_S_physics.mean()), S_LWR_R3=float(sub.base_S_LWR_R3.mean())),
        dict(J=float(sub.cand_J.mean()), S_state=float(sub.cand_S_state.mean()),
             S_physics=float(sub.cand_S_physics.mean()), S_LWR_R3=float(sub.cand_S_LWR_R3.mean())),
    )
    if float(pick) <= 0.0:
        g["pass_gate"] = False
        g["reason"] = "best weight is 0, which leaves the GBDT field unchanged"
    metrics = pd.DataFrame([{
        "prior": "gbdt_dec_feb",
        "picked_alpha": float(pick),
        "base_J": float(sub.base_J.mean()),
        "cand_J": float(sub.cand_J.mean()),
        **g,
        "n_panels": int(sub.panel.nunique()),
    }])
    metrics.to_csv(OUT / "metrics_gbdt.csv", index=False)
    print("GBDT blend picked α", pick, "gate", g, flush=True)
    print(metrics.round(4).to_string(index=False), flush=True)
    if not g["pass_gate"]:
        print("GATE FAIL vs GBDT: do not submit. Keep J1.", flush=True)
    return df


def main() -> None:
    cmd = sys.argv[1] if len(sys.argv) > 1 else "selftest"
    if cmd == "selftest":
        selftest_all()
    elif cmd == "train":
        train(sys.argv[2:] or None)
    elif cmd == "gbdt":
        train_gbdt(sys.argv[2:] or None)
    elif cmd == "all":
        selftest_all()
        train()
    else:
        raise SystemExit(f"unknown command {cmd}")


if __name__ == "__main__":
    main()
