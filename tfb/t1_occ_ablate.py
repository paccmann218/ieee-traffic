"""Ablate occupancy, k_occ vs q/v residual, and pct_observed-at-target as Task 1 features.

J1 uses occupancy only as `op` (previous occupancy). Occupancy is blanked at Task 1
targets (same outage as speed/flow) but interpolated occupancy and neighbour
occupancy are legal. pct_observed remains at blanked cells.

Does not change the production GBDT unless T1_OCC_FEATS=1 is set at feature build.

Usage: python tfb/t1_occ_ablate.py selftest | train [panel]
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, "tfb")
import cache
import legal as L
import score_local as sl
import t1_base as tb
import t1_features as tf

OUT = L.ARCH_C
OCC_COLS = ("oi", "ko", "kqv", "kres", "pctt", "o+1", "o-1")


def occ_block(panel: str, z: dict, d: int, vi, qi, O):
    """Occupancy / quality columns on a [T,L] day, aligned with t1_features.build."""
    links = list(z["links"])
    p = sl.net_params(panel, links)
    lanes = p.lanes.to_numpy()
    kj = p.k_jam.to_numpy()
    oi = tb.interp_day(O)
    k_occ = (np.clip(oi, 0.0, 100.0) / 100.0) * kj
    kqv = qi / np.maximum(vi, 1.0)
    return {
        "oi": oi,
        "ko": k_occ / lanes,
        "kqv": kqv / lanes,
        "kres": np.abs(kqv - k_occ) / np.maximum(lanes, 1.0),
        "pctt": z["pct"][d].astype(np.float32),
        "o+1": tf.shift_l(O, 1),
        "o-1": tf.shift_l(O, -1),
    }


def attach_occ(panel: str, z: dict, X: pd.DataFrame, idx: np.ndarray) -> pd.DataFrame:
    """Join occupancy ablation columns onto a feature table of target cells."""
    mv = tb.usable(z, "speed")
    mq = tb.usable(z, "flow")
    mo = np.where(z["elig"].astype(bool), z["mocc"], np.nan)
    extra = {c: np.empty(len(X), np.float32) for c in OCC_COLS}
    for d in np.unique(idx[:, 0]):
        sel = idx[:, 0] == d
        tt, ll = idx[sel, 1], idx[sel, 2]
        vi = tb.interp_day(mv[d])
        qi = tb.interp_day(mq[d])
        block = occ_block(panel, z, int(d), vi, qi, mo[d])
        for c, grid in block.items():
            extra[c][sel] = grid[tt, ll]
    for c, arr in extra.items():
        X[c] = arr
    return X


def corr_report(X: pd.DataFrame, y_v, y_q) -> pd.DataFrame:
    rows = []
    for c in OCC_COLS:
        if c not in X.columns:
            continue
        xv = pd.to_numeric(X[c], errors="coerce").to_numpy()
        ok = np.isfinite(xv) & np.isfinite(y_v) & np.isfinite(y_q)
        if ok.sum() < 20:
            rows.append(dict(feat=c, n=int(ok.sum()), corr_v=np.nan, corr_q=np.nan))
            continue
        rows.append(dict(
            feat=c, n=int(ok.sum()),
            corr_v=float(np.corrcoef(xv[ok], y_v[ok])[0, 1]),
            corr_q=float(np.corrcoef(xv[ok], y_q[ok])[0, 1]),
        ))
    return pd.DataFrame(rows)


def train(panel: str | None = None) -> pd.DataFrame:
    panels = [panel] if panel else list(cache.PANELS)
    parts = []
    for p in panels:
        z = cache.load(p, "train")
        pv, pq = tb.profile(z, ch="speed"), tb.profile(z, ch="flow")
        X, idx = tf.build(p, z, pv, pq)
        X = attach_occ(p, z, X, idx)
        y_v = z["speed"][idx[:, 0], idx[:, 1], idx[:, 2]]
        y_q = z["flow"][idx[:, 0], idx[:, 1], idx[:, 2]] / X.lanes.to_numpy()
        resid_v = y_v - X.vi.to_numpy()
        resid_q = y_q - X.qi.to_numpy()
        r = corr_report(X, resid_v, resid_q)
        r["panel"] = p
        parts.append(r)
        print(p, r.round(3).to_string(index=False), flush=True)
    out = pd.concat(parts, ignore_index=True)
    OUT.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT / "occ_ablate.csv", index=False)
    keep = out.groupby("feat")[["corr_v", "corr_q"]].mean().reset_index()
    keep["keep"] = keep.corr_v.abs().gt(0.03) | keep.corr_q.abs().gt(0.03)
    print("mean |corr| keep-if>|0.03|:\n", keep.round(3).to_string(index=False), flush=True)
    return out


def selftest() -> None:
    T, L = 30, 6
    lanes = np.full(L, 4.0)
    kj = np.full(L, 400.0)
    k = np.linspace(20, 180, T)[:, None] * np.ones(L)
    v = np.clip(115 * np.maximum(1 - k / kj, 0.15), 5, 140)
    q = k * v
    occ = 100.0 * k / kj
    pct = np.full((T, L), 90.0)
    pct[10:12, :] = 40
    oi = occ.copy()
    k_occ = (oi / 100.0) * kj
    kqv = q / np.maximum(v, 1.0)
    kres = np.abs(kqv - k_occ)
    assert kres.mean() < 1e-4, kres.mean()
    # pct_observed stays at "blanked" cells
    blank = np.ones((T, L), bool)
    blank[5:8, 2] = False
    assert pct[6, 2] == 40 or pct[6, 2] == 90
    r = np.corrcoef(k_occ.ravel(), kqv.ravel())[0, 1]
    assert r > 0.99
    print("t1_occ_ablate selftest OK  k_occ vs q/v corr", round(float(r), 4), flush=True)


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "selftest"
    if cmd == "train":
        train(sys.argv[2] if len(sys.argv) > 2 else None)
    else:
        selftest()
