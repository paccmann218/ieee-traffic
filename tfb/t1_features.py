"""Task 1 feature construction on dense caches. Identical code path for train (CV) and validation/private.

All features are built from the MASKED layer only (targets and blanks are NaN), plus static network data and
the train-only historical profile. Rows = target cells (template) of the split.
"""
from __future__ import annotations
import os
import numpy as np, pandas as pd
import cache, score_local as sl, t1_base as tb

REL = cache.REL
V2 = True   # extended feature set (t1 v2)
OCC_FEATS = os.environ.get("T1_OCC_FEATS", "0") == "1"  # occupancy / pct_observed ablation; off for J1


def positions(panel, links):
    n = pd.read_csv(REL/"corridors"/panel/"network"/"links.csv", dtype={"link_id": str}).drop_duplicates("link_id").set_index("link_id")
    L = pd.to_numeric(n.length_km.reindex(links), errors="coerce").fillna(0.5).to_numpy()
    x = np.cumsum(L) - L / 2
    return x, L


def prev_next(x):
    """For [T, L] with NaN: value and gap (slots) of the nearest observed cell strictly before / after each cell."""
    T, L = x.shape; ok = np.isfinite(x); idx = np.arange(T)[:, None]
    prev_inc = np.maximum.accumulate(np.where(ok, idx, -1), axis=0)
    next_inc = np.minimum.accumulate(np.where(ok, idx, 10**6)[::-1], axis=0)[::-1]
    prev = np.vstack([np.full((1, L), -1), prev_inc[:-1]]); nxt = np.vstack([next_inc[1:], np.full((1, L), 10**6)])
    cols = np.arange(L)[None, :]
    vp = np.where(prev >= 0, x[np.clip(prev, 0, T - 1), cols], np.nan); vn = np.where(nxt < T, x[np.clip(nxt, 0, T - 1), cols], np.nan)
    gp = np.where(prev >= 0, idx - prev, 99).astype(np.float32); gn = np.where(nxt < T, nxt - idx, 99).astype(np.float32)
    return vp, gp, vn, gn


def asm(v, q, xpos, c_free=100.0, c_cong=-18.0, sig_x=0.8, tau=1.0, dts=(-3, -2, -1, 0, 1, 2, 3), dls=range(-6, 7)):
    """Adaptive smoothing (Treiber-Helbing) on a [T, L] grid; returns (v_asm, q_asm, v_free, v_cong).

    Kernel phi(dx, dt) = exp(-|dx|/sig_x - |dt|/tau) evaluated along characteristics dx - c*dt.
    Excludes the cell itself so the estimate at an observed cell is a leave-one-out estimate.
    """
    T, L = v.shape; ok = np.isfinite(v) & np.isfinite(q)
    v0 = np.where(ok, v, 0.0); q0 = np.where(ok, q, 0.0)
    acc = {k: np.zeros((T, L)) for k in ("vf", "wf", "qf", "vc", "wc", "qc")}
    for dt in dts:
        for dl in dls:
            if dt == 0 and dl == 0: continue
            # neighbour (t+dt, l+dl) contributes to (t, l)
            ts, te = max(0, -dt), min(T, T - dt); ls_, le = max(0, -dl), min(L, L - dl)
            if ts >= te or ls_ >= le: continue
            dx = xpos[ls_ + dl: le + dl] - xpos[ls_: le]  # km, neighbour minus target
            hrs = dt * 5 / 60.0
            for tag, c in (("f", c_free), ("c", c_cong)):
                w = np.exp(-np.abs(dx - c * hrs) / sig_x - abs(dt) / tau)[None, :] * ok[ts + dt: te + dt, ls_ + dl: le + dl]
                acc["v" + tag][ts:te, ls_:le] += w * v0[ts + dt: te + dt, ls_ + dl: le + dl]
                acc["q" + tag][ts:te, ls_:le] += w * q0[ts + dt: te + dt, ls_ + dl: le + dl]
                acc["w" + tag][ts:te, ls_:le] += w
    with np.errstate(invalid="ignore", divide="ignore"):
        vf, vc = acc["vf"] / acc["wf"], acc["vc"] / acc["wc"]; qf, qc = acc["qf"] / acc["wf"], acc["qc"] / acc["wc"]
    vmin = np.fmin(vf, vc); w = 0.5 * (1 + np.tanh((60.0 - vmin) / 20.0))  # weight on congested estimate
    return (w * vc + (1 - w) * vf).astype(np.float32), (w * qc + (1 - w) * qf).astype(np.float32), vf.astype(np.float32), vc.astype(np.float32)


def shift_l(a, k):
    out = np.full_like(a, np.nan)
    if k > 0: out[:, :-k] = a[:, k:]
    elif k < 0: out[:, -k:] = a[:, :k]
    else: out = a.copy()
    return out


def ramp_on_link(panel, z):
    """Sum of released ramp flows (eligible only) attached to each mainline link, split on/off: [D, 288, L] x2."""
    links = list(z["links"]); li = {x: i for i, x in enumerate(links)}
    amap = pd.read_csv(REL/"corridors"/panel/"network"/"ramp_attachment_map.csv", dtype=str)
    rid = list(z["ramp_ids"]); ri = {x: i for i, x in enumerate(rid)}
    rf = np.where(z["relig"].astype(bool), z["rflow"], np.nan)
    D = rf.shape[0]; on = np.full((D, 288, len(links)), np.nan, np.float32); off = on.copy()
    for row in amap.itertuples():
        if row.ramp_link_id not in ri or row.nearest_mainline_link_id not in li: continue
        r, l = ri[row.ramp_link_id], li[row.nearest_mainline_link_id]; tgt = on if str(z["ramp_type"][r]).upper() in ("OR", "ON") else off
        tgt[:, :, l] = np.where(np.isnan(tgt[:, :, l]), rf[:, :, r], tgt[:, :, l] + np.nan_to_num(rf[:, :, r]))
    return on, off


def build(panel, z, prof_v, prof_q, days=None):
    """Return (X DataFrame, index arrays (d, t, l)) for the target cells of z (optionally restricted to days)."""
    links = list(z["links"]); L = len(links); xpos, Lk = positions(panel, links)
    p = sl.net_params(panel, links); lanes = p.lanes.to_numpy(); vf_link = p.free_speed_kmh.to_numpy(); cap = p.capacity_vph.to_numpy()
    kj = p.k_jam.to_numpy()
    mv = tb.usable(z, "speed"); mq = tb.usable(z, "flow"); mo = np.where(z["elig"].astype(bool), z["mocc"], np.nan)
    inel_v = np.where(~z["elig"].astype(bool), z["mspeed"], np.nan); inel_q = np.where(~z["elig"].astype(bool), z["mflow"], np.nan)
    ron, roff = ramp_on_link(panel, z)
    wd = pd.to_datetime(z["dates"]).weekday.to_numpy()
    reg = {"R1": 0, "R2": 1, "R3": 2}
    days = range(len(z["dates"])) if days is None else days
    feats, idx = [], []
    for d in days:
        tg = z["target"][d].astype(bool)
        if not tg.any(): continue
        tt, ll = np.nonzero(tg)
        V, Q, O = mv[d], mq[d], mo[d]
        vi = tb.interp_day(V); qi = tb.interp_day(Q)
        vp, gp, vn, gn = prev_next(V); qp, _, qn, _ = prev_next(Q)
        va, qa, vfree, vcong = asm(V, Q, xpos)
        rv = V - vi; rq = Q - qi  # innovations at observed cells are 0 by construction for interp; use LOO below
        # leave-one-out temporal innovation at observed cells (for neighbour features)
        lv = V - np.where(np.isfinite(vp) & np.isfinite(vn), (vp * gn + vn * gp) / (gp + gn), np.fmax(vp, vn))
        lq = Q - np.where(np.isfinite(qp) & np.isfinite(qn), (qp * gn + qn * gp) / (gp + gn), np.fmax(qp, qn))
        f = {
            "vi": vi, "qi": qi / lanes, "vp": vp, "vn": vn, "gp": gp, "gn": gn, "qp": qp / lanes, "qn": qn / lanes,
            "va": va, "qa": qa / lanes, "vfree": vfree, "vcong": vcong,
            "pv": prof_v[d], "pq": prof_q[d] / lanes,
            "ron": ron[d], "roff": roff[d],
            "iv": inel_v[d], "iq": inel_q[d] / lanes,
            "op": prev_next(O)[0],
        }
        if OCC_FEATS:
            oi = tb.interp_day(O)
            k_occ = (np.clip(oi, 0.0, 100.0) / 100.0) * kj
            kqv = qi / np.maximum(vi, 1.0)
            f["oi"] = oi
            f["ko"] = k_occ / lanes
            f["kqv"] = kqv / lanes
            f["kres"] = np.abs(kqv - k_occ) / np.maximum(lanes, 1.0)
            f["pctt"] = z["pct"][d].astype(np.float32)
            f["o+1"] = shift_l(O, 1)
            f["o-1"] = shift_l(O, -1)
        for k in (1, 2, 3):
            for s in (k, -k):
                f[f"v{s:+d}"] = shift_l(V, s); f[f"q{s:+d}"] = shift_l(Q, s) / lanes
                f[f"lv{s:+d}"] = shift_l(lv, s); f[f"lq{s:+d}"] = shift_l(lq, s) / lanes
        # neighbour innovations one slot before/after (same time innovations propagate)
        for s in (1, -1):
            f[f"lv{s:+d}_tm1"] = np.vstack([np.full((1, L), np.nan), shift_l(lv, s)[:-1]])
            f[f"lv{s:+d}_tp1"] = np.vstack([shift_l(lv, s)[1:], np.full((1, L), np.nan)])
            if V2:
                f[f"lq{s:+d}_tm1"] = np.vstack([np.full((1, L), np.nan), shift_l(lq, s)[:-1]]) / lanes
                f[f"lq{s:+d}_tp1"] = np.vstack([shift_l(lq, s)[1:], np.full((1, L), np.nan)]) / lanes
        if V2:
            # second-nearest same-link observations (two steps out)
            f["v_tm2"] = np.vstack([np.full((2, L), np.nan), V[:-2]]); f["v_tp2"] = np.vstack([V[2:], np.full((2, L), np.nan)])
            f["q_tm2"] = np.vstack([np.full((2, L), np.nan), Q[:-2]]) / lanes; f["q_tp2"] = np.vstack([Q[2:], np.full((2, L), np.nan)]) / lanes
            # ramps one slot before/after
            f["ron_tm1"] = np.vstack([np.full((1, L), np.nan), ron[d][:-1]]); f["ron_tp1"] = np.vstack([ron[d][1:], np.full((1, L), np.nan)])
            f["roff_tm1"] = np.vstack([np.full((1, L), np.nan), roff[d][:-1]]); f["roff_tp1"] = np.vstack([roff[d][1:], np.full((1, L), np.nan)])
            # local link-day level: observed minus profile, averaged over +-6 slots (excluding the cell itself)
            from scipy.ndimage import uniform_filter1d
            for nm, obs, pr, sc in (("dv", V, prof_v[d], 1.0), ("dq", Q, prof_q[d], lanes)):
                dev = obs - pr; okd = np.isfinite(dev)
                s_ = uniform_filter1d(np.where(okd, dev, 0.0), 13, axis=0, mode="constant") * 13 - np.where(okd, dev, 0.0)
                n_ = uniform_filter1d(okd.astype(float), 13, axis=0, mode="constant") * 13 - okd
                f[f"lvl_{nm}"] = np.where(n_ > 0, s_ / np.maximum(n_, 1), np.nan) / sc
            va2, qa2, _, _ = asm(V, Q, xpos, c_cong=-12.0, sig_x=1.5, tau=1.5)
            f["va2"] = va2; f["qa2"] = qa2 / lanes
        X = {k: v[tt, ll] for k, v in f.items()}
        X.update(lanes=lanes[ll], vfl=vf_link[ll], capl=cap[ll] / lanes[ll], Lk=Lk[ll], pos=ll / L, slot=tt, wd=np.full(len(tt), wd[d]),
                 regime=np.full(len(tt), reg[str(z["regime"][d])]), obsrate=np.isfinite(V).mean(), panel_L=L)
        feats.append(pd.DataFrame({k: np.asarray(v, np.float32) for k, v in X.items()}))
        idx.append(np.c_[np.full(len(tt), d), tt, ll])
    return pd.concat(feats, ignore_index=True), np.vstack(idx)
