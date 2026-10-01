"""Density-aware decoding experiment on CV predictions (test months Dec-Feb).

q' = v_hat * k_mix,  k_mix = a * (q_hat / v_hat) + (1 - a) * k_interp   (k_interp: same-link time interpolation of observed density)
Also tests shrinking the target density toward the neighbour average in time (b).
Scores J = 0.35*S_state + 0.15*S_physics with the local scorers.
"""
import sys, numpy as np, pandas as pd
sys.path.insert(0, "tfb")
import cache, score_local as sl, t1_base as tb
from pathlib import Path
W = Path.home()/"tfb/work/t1feat"
te = pd.read_parquet(W/"cv_preds.parquet")
panels = sys.argv[1:] or ["D7_I10_E", "D12_I5_N", "D7_I405_N", "D12_I405_S"]
rows = []
for panel in panels:
    pi = cache.PANELS.index(panel); z = cache.load(panel, "train"); lanes = sl.lanes_t1(panel, z["links"]); t3 = sl.T3(panel, z)
    mon = pd.to_datetime(z["dates"]).month.to_numpy(); keep = np.isin(mon, [12, 1, 2])
    zz = {k: (v[keep] if isinstance(v, np.ndarray) and v.ndim >= 1 and v.shape[0] == len(mon) else v) for k, v in z.items()}
    t3.ramp_ok = t3.ramp_ok[keep]
    sub = te[te.panel == pi]; dmap = {d: i for i, d in enumerate(np.flatnonzero(keep))}; dd = sub.d.map(dmap).to_numpy()
    PV = np.full(zz["speed"].shape, np.nan, np.float32); PQ = PV.copy(); PV[dd, sub.t, sub.l] = sub.pv; PQ[dd, sub.t, sub.l] = sub.pq
    ko = tb.usable(zz, "flow") / np.maximum(tb.usable(zz, "speed"), 1.0)
    KI = np.stack([tb.interp_day(ko[d]) for d in range(ko.shape[0])])
    for a in (1.0, 0.8, 0.6, 0.4, 0.2, 0.0):
        kmix = a * PQ / np.maximum(PV, 1.0) + (1 - a) * KI
        for mode in ("q", "v"):  # keep v, change q ; or keep q, change v
            if mode == "q": pv, pq = PV, PV * kmix
            else: pv, pq = PQ / np.maximum(kmix, 1e-3), PQ
            s1 = sl.score_t1(zz, pv, pq, lanes); s3 = t3.score(zz, pv, pq)
            rows.append(dict(panel=panel, a=a, mode=mode, S_state=s1["S_state"], S_phys=s3["S_physics"], J=0.35 * s1["S_state"] + 0.15 * s3["S_physics"]))
    print(panel, flush=True)
r = pd.DataFrame(rows); print(r.groupby(["mode", "a"])[["S_state", "S_phys", "J"]].mean().round(4).to_string())
