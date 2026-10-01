"""Isolated Architecture C state probe vs frozen J1.

Builds a *new* submission (sub_C0) that changes only Task 1 state rows.
Queue stays J1 onset analogue + ffill_inel. ODME stays V1. Never writes sub_J1.

Gate: require work/arch_c/metrics.csv with pass_gate=True. If the joint metric
did not improve, refuse to build (the LB probe would be wasted).

Usage:
  python tfb/build_c_probe.py selftest
  python tfb/build_c_probe.py build [--alpha 0.5]
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, "tfb")
import cache
import legal as L
import t1_base as tb
import t1_newell as tn
import t1_project as tpj

SUBS = L.SUBS
WORK = L.WORK
ARCH = L.ARCH_C
SPLITS = ("validation", "private")
PROBE_NAME = "sub_C0"


def load_gate() -> dict:
    path = ARCH / "metrics.csv"
    if not path.exists():
        return dict(pass_gate=False, reason="no metrics.csv; run arch_c_run.py train first")
    df = pd.read_csv(path)
    if "pass_gate" in df.columns:
        ok = bool(df.pass_gate.astype(str).str.lower().isin(("true", "1")).any())
        return dict(pass_gate=ok, reason="metrics.csv", table=df)
    if "dJ" in df.columns:
        ok = float(df.dJ.iloc[-1]) >= 0
        return dict(pass_gate=ok, reason="dJ", table=df)
    return dict(pass_gate=False, reason="metrics.csv missing pass_gate", table=df)


def isolate(new: pd.DataFrame, reference: pd.DataFrame, name: str) -> dict:
    assert np.array_equal(new.submission_id.to_numpy(), reference.submission_id.to_numpy())
    out = {}
    for task, columns in (
        ("state", ["speed_kmh", "flow_vph"]),
        ("queue", ["queue_pred"]),
        ("odme", ["path_flow"]),
    ):
        rows = new.task == task
        changed = ~np.isclose(
            new.loc[rows, columns].to_numpy(),
            reference.loc[rows, columns].to_numpy(),
            rtol=0, atol=1e-9,
        ).all(axis=1)
        out[task] = int(changed.sum())
        print(f"{name} vs J1: {task} rows differing = {out[task]}", flush=True)
    return out


def _overlay_state(prior_v, prior_q, z, state: pd.DataFrame, panel: str, split: str):
    """Scatter GBDT predictions onto the tinterp prior at Task 1 targets."""
    if state is None or state.empty or "speed_kmh" not in state.columns:
        return prior_v, prior_q
    links = list(z["links"])
    li = {x: i for i, x in enumerate(links)}
    di = {str(d): i for i, d in enumerate(z["dates"])}
    sub = state.copy()
    if "panel" in sub.columns:
        sub = sub[sub.panel.astype(str) == panel]
    if sub.empty:
        return prior_v, prior_q
    ts = sub.timestamp.astype(str)
    dd = ts.str[:10].map(di)
    tt = ts.str[11:13].astype(int) * 12 + ts.str[14:16].astype(int) // 5
    ll = sub.link_id.astype(str).map(li)
    good = dd.notna() & ll.notna()
    if not good.any():
        return prior_v, prior_q
    d = dd[good].astype(int).to_numpy()
    t = tt[good].to_numpy()
    l = ll[good].astype(int).to_numpy()
    prior_v = prior_v.copy()
    prior_q = prior_q.copy()
    prior_v[d, t, l] = sub.loc[good, "speed_kmh"].to_numpy(float)
    prior_q[d, t, l] = sub.loc[good, "flow_vph"].to_numpy(float)
    return prior_v, prior_q


def apply_arch_c_to_state(state: pd.DataFrame, alpha: float) -> pd.DataFrame:
    """Replace target cells with Newell+projection on the GBDT field."""
    out = []
    for panel in cache.PANELS:
        for split in SPLITS:
            z = cache.load(panel, split)
            prior_v = tb.time_interp(z, "speed")
            prior_q = tb.time_interp(z, "flow")
            prior_v, prior_q = _overlay_state(prior_v, prior_q, z, state, panel, split)
            PV, PQ, fd, meta = tpj.arch_c_split(panel, z, prior_v, prior_q, alpha=alpha)
            print(panel, split, meta, flush=True)
            dates = z["dates"]
            links = list(z["links"])
            li = {x: i for i, x in enumerate(links)}
            di = {str(d): i for i, d in enumerate(dates)}
            tpl = pd.read_csv(
                cache.REL / "task1" / panel / split / "sample_submission_state.csv",
                dtype=str,
            )
            ts = tpl.timestamp.str
            dd = tpl.timestamp.str[:10].map(di).to_numpy()
            tt = (ts[11:13].astype(int) * 12 + ts[14:16].astype(int) // 5).to_numpy()
            ll = tpl.link_id.map(li).to_numpy()
            part = tpl.copy()
            part["speed_kmh"] = np.clip(PV[dd, tt, ll], tn.V_MIN, tn.V_MAX)
            part["flow_vph"] = np.clip(PQ[dd, tt, ll], 0.0, None)
            out.append(part)
    return pd.concat(out, ignore_index=True)


def merge_probe(state_csv: Path, out_name: str) -> Path:
    L.refuse_j1_write(SUBS / f"{out_name}.zip")
    L.refuse_j1_write(SUBS / f"{out_name}.csv")
    j1 = SUBS / "sub_J1.csv"
    if not j1.exists():
        raise SystemExit("sub_J1.csv is required as the queue+ODME donor")
    key = cache.REL / "submission_key.csv"
    queue = WORK / "u1u2" / "queue_J1.csv"
    if not queue.exists():
        # Fall back to extracting queue rows from J1 itself after merge.
        queue = None
    odme = Path.home() / "tfb/subs/odme_ref.csv"
    cwd = str(Path.home() / "tfb")
    if queue is not None and odme.exists() and key.exists():
        r = subprocess.run(
            [sys.executable, "repo/src/merge_submissions.py",
             "--state", str(state_csv), "--queue", str(queue),
             "--odme", str(odme), "--key", str(key),
             "--output", str(SUBS / f"{out_name}.csv")],
            capture_output=True, text=True, cwd=cwd,
        )
        print(out_name, "merge:", r.stdout.strip().splitlines()[-2:], flush=True)
        if r.returncode != 0:
            raise SystemExit(r.stderr[-500:] if r.stderr else r.stdout[-500:])
    else:
        raise SystemExit("Need J1 queue csv, odme_ref.csv, and submission_key to merge a probe")
    new = pd.read_csv(SUBS / f"{out_name}.csv")
    ref = pd.read_csv(j1)
    diff = isolate(new, ref, out_name)
    if diff["queue"] != 0 or diff["odme"] != 0:
        raise SystemExit(f"isolation failed: {diff}")
    if diff["state"] == 0:
        raise SystemExit("state rows did not change; probe is not informative")
    return SUBS / f"{out_name}.csv"


def build(alpha: float = 0.5, force: bool = False) -> None:
    gate = load_gate()
    print("gate", {k: gate[k] for k in gate if k != "table"}, flush=True)
    if not gate["pass_gate"] and not force:
        raise SystemExit(
            "Joint metric did not pass the Architecture C gate. "
            "Refusing to spend an LB submission. Pass --force only to debug the builder."
        )
    ARCH.mkdir(parents=True, exist_ok=True)
    gbdt = WORK / "t1" / "state_submission.csv"
    if gbdt.exists():
        state = pd.read_csv(gbdt, dtype={"panel": str, "link_id": str})
    else:
        state = pd.DataFrame()
    filled = apply_arch_c_to_state(state, alpha=alpha)
    out_state = ARCH / "state_submission_c.csv"
    filled.to_csv(out_state, index=False)
    print("wrote", out_state, "rows", len(filled), flush=True)
    merge_probe(out_state, PROBE_NAME)
    print(
        f"Built {PROBE_NAME} (isolated state vs J1). Do not upload unless this is the "
        f"pre-registered Wave 2 probe. Daily cap 5.",
        flush=True,
    )


def selftest() -> None:
    L.refuse_j1_write(SUBS / "sub_C0.zip")
    try:
        L.refuse_j1_write(SUBS / "sub_J1.zip")
        raise AssertionError("must refuse J1")
    except PermissionError:
        pass
    a = pd.DataFrame({
        "submission_id": ["1", "2", "3"],
        "task": ["state", "queue", "odme"],
        "speed_kmh": [80.0, np.nan, np.nan],
        "flow_vph": [1000.0, np.nan, np.nan],
        "queue_pred": [np.nan, 1.0, np.nan],
        "path_flow": [np.nan, np.nan, 3.0],
    })
    b = a.copy()
    b.loc[0, "speed_kmh"] = 81.0
    d = isolate(b, a, "selftest")
    assert d["state"] == 1 and d["queue"] == 0 and d["odme"] == 0
    g = load_gate()
    pv = np.zeros((2, 4, 3), np.float32)
    pq = np.zeros_like(pv)
    z = dict(links=np.array(["a", "b", "c"]), dates=np.array(["2030-06-01", "2030-06-02"]))
    st = pd.DataFrame({
        "panel": ["D12_I5_N", "D12_I5_N"],
        "timestamp": ["2030-06-01T00:10:00Z", "2030-06-02T00:05:00Z"],
        "link_id": ["b", "c"],
        "speed_kmh": [77.0, 88.0],
        "flow_vph": [1111.0, 2222.0],
    })
    ov_v, ov_q = _overlay_state(pv, pq, z, st, "D12_I5_N", "validation")
    assert ov_v[0, 2, 1] == 77.0 and ov_q[1, 1, 2] == 2222.0
    print("t1/build_c_probe selftest OK  gate_exists", (ARCH / "metrics.csv").exists(), g.get("reason"), flush=True)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "selftest":
        selftest()
    else:
        p = argparse.ArgumentParser()
        p.add_argument("cmd", nargs="?", default="build")
        p.add_argument("--alpha", type=float, default=0.5)
        p.add_argument("--force", action="store_true")
        args, _ = p.parse_known_args()
        if args.cmd == "build":
            build(alpha=args.alpha, force=args.force)
        else:
            selftest()
