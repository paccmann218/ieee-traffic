"""Newell three-detector / Daganzo variational fill on triangular FD.

Not Godunov CTM. Links can be 0.025 km; CFL at vf would be ~0.8 s per step.
Newell tracks upstream/downstream characteristics on cumulative count N(x,t):

    N(x,t) = min{ N_up(t - x/v_f),  N_down(t - remaining/|w|) + k_jam * remaining }

with w = -capacity / (k_jam - k_crit) ≈ -25 km/h on these diagrams.

FD source is fd_parameters.csv only. OSM links.csv supplies length, never vf/cap.

Observed cells are copied through. Holes take the binding characteristic, then
the triangular FD. A prior (tinterp/GBDT) is used only when neither detector
is reachable.

Usage: python tfb/t1_newell.py selftest
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, "tfb")
import cache
import legal as L
import t1_base as tb

REL = cache.REL
DT_H = 5.0 / 60.0
V_MIN, V_MAX = 5.0, 140.0


def fd_authority(panel: str, links) -> pd.DataFrame:
    """Released triangle in cache link order. Refuses OSM speed/capacity fallback."""
    links = list(links)
    npath = REL / "corridors" / panel / "network" / "links.csv"
    fpath = REL / "corridors" / panel / "network" / "fd_parameters.csv"
    L.assert_fd_authority(pd.DataFrame(), str(npath))
    n = pd.read_csv(npath, dtype={"link_id": str}).drop_duplicates("link_id").set_index("link_id")
    f = pd.read_csv(fpath, dtype={"link_id": str}).groupby("link_id")[
        ["free_speed_kmh", "capacity_vph", "lanes", "k_jam"]
    ].mean()
    out = pd.DataFrame(index=pd.Index(links, name="link_id"))
    out["length_km"] = pd.to_numeric(n.length_km.reindex(links), errors="coerce").fillna(0.5)
    for c in ("free_speed_kmh", "capacity_vph", "lanes", "k_jam"):
        out[c] = f[c].reindex(links)
    if out[["free_speed_kmh", "capacity_vph", "lanes", "k_jam"]].isna().any().any():
        raise PermissionError(
            f"{panel}: fd_parameters.csv incomplete; refusing OSM links.csv fallback"
        )
    out["lanes"] = out.lanes.clip(lower=1.0)
    out["critical_density"] = out.capacity_vph / out.free_speed_kmh.clip(lower=1.0)
    out["k_jam"] = np.maximum(out.k_jam, out.critical_density * 1.05)
    out["w_kmh"] = -out.capacity_vph / np.maximum(out.k_jam - out.critical_density, 1e-6)
    x = np.cumsum(out.length_km.to_numpy()) - out.length_km.to_numpy() / 2.0
    out["x_km"] = x
    return out


def fd_vq_from_k(k, vf, cap, kc, kj):
    """Triangular FD: density k (veh/km, all lanes) -> speed, flow."""
    q = np.where(
        k <= kc,
        vf * k,
        cap * np.maximum(kj - k, 0.0) / np.maximum(kj - kc, 1e-9),
    )
    q = np.clip(q, 0.0, None)
    v = np.where(k > 1e-6, q / np.maximum(k, 1e-6), vf)
    return np.clip(v, V_MIN, V_MAX), q


def nearest_up_dn(ok_t: np.ndarray):
    """Strict upstream / downstream observed indices for a [L] boolean slice."""
    L = ok_t.size
    idx = np.arange(L)
    last = np.where(ok_t, idx, -1)
    up = np.maximum.accumulate(last)
    up = np.concatenate([[-1], up[:-1]])
    nxt = np.where(ok_t, idx, L + 1)
    dn_at = np.minimum.accumulate(nxt[::-1])[::-1]
    dn = np.concatenate([dn_at[1:], [L + 1]])
    dn = np.where(dn > L - 1, -1, dn)
    return up, dn


def _gather(A, ok, t, l_idx, dt_slots, T):
    """Value of A at (t - dt, l), searching ±2 slots if the exact time is blank."""
    tt = np.clip(np.rint(t - dt_slots).astype(int), 0, T - 1)
    out = np.full(l_idx.shape, np.nan, np.float64)
    found = np.zeros(l_idx.shape, bool)
    for delta in (0, -1, 1, -2, 2):
        t2 = np.clip(tt + delta, 0, T - 1)
        good = ok[t2, l_idx] & (l_idx >= 0) & (~found)
        if good.any():
            out[good] = A[t2[good], l_idx[good]]
            found |= good
        if found.all():
            break
    return out


def newell_fill_day(
    V, Q, ok, x, vf, cap, kc, kj, lanes,
    prior_v=None, prior_q=None,
):
    """Fill one day [T,L]. Observed cells unchanged. Returns Vf, Qf, meta.

    At 5-minute resolution a 25 m link is crossed in well under one slot, so
    the three-detector rule is a same-time copy of the binding characteristic:
    congested downstream if that detector is queued, otherwise free-flow upstream.
    Time shifts apply only when travel time is at least half a slot.
    """
    T, nL = V.shape
    wabs = np.abs(-cap / np.maximum(kj - kc, 1e-6))
    Vf = np.where(ok, V, np.nan).astype(np.float64)
    Qf = np.where(ok, Q, np.nan).astype(np.float64)
    if prior_v is not None:
        Vf = np.where(ok, V, prior_v)
        Qf = np.where(ok, Q, prior_q)
    n_up = n_dn = n_prior = 0
    for t in range(T):
        holes = np.flatnonzero(~ok[t])
        if holes.size == 0:
            continue
        lup, ldn = nearest_up_dn(ok[t])
        hu, hd = lup[holes], ldn[holes]
        has_u = hu >= 0
        has_d = hd >= 0
        dx_u = np.where(has_u, x[holes] - x[np.clip(hu, 0, nL - 1)], 0.0)
        dx_d = np.where(has_d, x[np.clip(hd, 0, nL - 1)] - x[holes], 0.0)
        dt_u = dx_u / np.maximum(vf[holes], 1.0) / DT_H
        dt_d = dx_d / np.maximum(wabs[holes], 1.0) / DT_H
        dt_u = np.where(dt_u >= 0.5, dt_u, 0.0)
        dt_d = np.where(dt_d >= 0.5, dt_d, 0.0)
        v_u = _gather(V, ok, t, np.where(has_u, hu, 0), dt_u, T)
        q_u = _gather(Q, ok, t, np.where(has_u, hu, 0), dt_u, T)
        v_d = _gather(V, ok, t, np.where(has_d, hd, 0), dt_d, T)
        q_d = _gather(Q, ok, t, np.where(has_d, hd, 0), dt_d, T)
        v_u = np.where(has_u, v_u, np.nan)
        q_u = np.where(has_u, q_u, np.nan)
        v_d = np.where(has_d, v_d, np.nan)
        q_d = np.where(has_d, q_d, np.nan)
        vcut = 0.6 * vf[holes]
        k_d = q_d / np.maximum(v_d, 1.0)
        cong = np.isfinite(v_d) & ((v_d <= vcut) | (k_d >= kc[np.clip(hd, 0, nL - 1)]))
        use_dn = cong & np.isfinite(v_d)
        use_up = (~use_dn) & np.isfinite(v_u)
        lane_u = lanes[np.clip(hu, 0, nL - 1)]
        lane_d = lanes[np.clip(hd, 0, nL - 1)]
        v_hat = np.where(use_dn, v_d, np.where(use_up, v_u, np.nan))
        q_hat = np.where(
            use_dn, q_d * lanes[holes] / np.maximum(lane_d, 1.0),
            np.where(use_up, q_u * lanes[holes] / np.maximum(lane_u, 1.0), np.nan),
        )
        take = np.isfinite(v_hat) & np.isfinite(q_hat)
        Vf[t, holes] = np.where(take, v_hat, Vf[t, holes])
        Qf[t, holes] = np.where(take, q_hat, Qf[t, holes])
        n_dn += int(use_dn.sum())
        n_up += int(use_up.sum())
        n_prior += int((~take).sum())
    Vf = np.clip(np.where(ok, V, Vf), V_MIN, V_MAX)
    Qf = np.clip(np.where(ok, Q, Qf), 0.0, None)
    return Vf.astype(np.float32), Qf.astype(np.float32), dict(n_up=n_up, n_dn=n_dn, n_prior=n_prior)


def newell_fill_split(panel: str, z: dict, prior_v=None, prior_q=None, use_masked=True):
    """Apply Newell on every day of a cache dict. Returns PV, PQ [D,T,L] and FD table."""
    fd = fd_authority(panel, z["links"])
    x = fd.x_km.to_numpy()
    vf, cap = fd.free_speed_kmh.to_numpy(), fd.capacity_vph.to_numpy()
    kc, kj, lanes = fd.critical_density.to_numpy(), fd.k_jam.to_numpy(), fd.lanes.to_numpy()
    if use_masked:
        V = tb.usable(z, "speed")
        Q = tb.usable(z, "flow")
    else:
        V = np.where(z["elig"].astype(bool), z["speed"], np.nan)
        Q = np.where(z["elig"].astype(bool), z["flow"], np.nan)
    ok = np.isfinite(V) & np.isfinite(Q)
    D = V.shape[0]
    PV = np.empty_like(V, np.float32)
    PQ = np.empty_like(Q, np.float32)
    meta = dict(n_up=0, n_dn=0, n_prior=0)
    for d in range(D):
        pv = None if prior_v is None else prior_v[d]
        pq = None if prior_q is None else prior_q[d]
        PV[d], PQ[d], m = newell_fill_day(V[d], Q[d], ok[d], x, vf, cap, kc, kj, lanes, pv, pq)
        for k in meta:
            meta[k] += m[k]
    return PV, PQ, fd, meta


def synthetic_shock(T=48, L=24, seed=0):
    """Backward-moving shock on a homogeneous triangular FD. Returns V, Q, fd arrays."""
    rng = np.random.default_rng(seed)
    length = np.full(L, 0.4)
    x = np.cumsum(length) - length / 2.0
    vf, cap, lanes = 115.0, 8000.0, 4.0
    kc = cap / vf
    kj = 400.0
    w = -cap / (kj - kc)
    v_f, k_f = vf, 0.35 * kc
    q_f = vf * k_f
    k_c = 0.70 * kj
    v_c, q_c = fd_vq_from_k(np.array([k_c]), vf, cap, kc, kj)
    v_c, q_c = float(v_c[0]), float(q_c[0])
    xs0 = x[-1] - 0.5
    V = np.full((T, L), v_f)
    Q = np.full((T, L), q_f)
    for t in range(T):
        xs = xs0 + w * t * DT_H
        cong = x >= xs
        V[t, cong] = v_c
        Q[t, cong] = q_c
    V = V + rng.normal(0, 0.8, V.shape)
    Q = Q + rng.normal(0, 20.0, Q.shape)
    fd = dict(x=x, vf=np.full(L, vf), cap=np.full(L, cap), kc=np.full(L, kc),
              kj=np.full(L, kj), lanes=np.full(L, lanes), w=w, length=length)
    return V.astype(np.float32), Q.astype(np.float32), fd


def selftest() -> None:
    V, Q, fd = synthetic_shock()
    T, L = V.shape
    rng = np.random.default_rng(1)
    ok = rng.random((T, L)) > 0.35
    # Spatial block at a shock-crossing time: tinterp must cross regimes, Newell should not.
    t_hit = 20
    ok[t_hit, 8:16] = False
    prior_v = tb.interp_day(np.where(ok, V, np.nan))
    prior_q = tb.interp_day(np.where(ok, Q, np.nan))
    Vn, Qn, meta = newell_fill_day(
        V, Q, ok, fd["x"], fd["vf"], fd["cap"], fd["kc"], fd["kj"], fd["lanes"],
        prior_v, prior_q,
    )
    miss = ~ok
    def rmse(p, y, m):
        d = np.asarray(p)[m] - np.asarray(y)[m]
        d = d[np.isfinite(d)]
        return float(np.sqrt(np.mean(d ** 2))) if d.size else np.nan
    rmse_n = rmse(Vn, V, miss)
    rmse_t = rmse(prior_v, V, miss)
    block = np.zeros((T, L), bool)
    block[t_hit, 8:16] = True
    rmse_n_b = rmse(Vn, V, block)
    rmse_t_b = rmse(prior_v, V, block)
    assert np.allclose(Vn[ok], V[ok], atol=1e-4), "Newell must copy observations"
    assert meta["n_up"] + meta["n_dn"] > 0
    assert float(np.nanmax(Vn)) <= V_MAX + 1e-5, float(np.nanmax(Vn))
    print(
        f"t1_newell selftest OK  rmse_all newell={rmse_n:.3f} tinterp={rmse_t:.3f}  "
        f"block newell={rmse_n_b:.3f} tinterp={rmse_t_b:.3f}  meta={meta}",
        flush=True,
    )
    if rmse_n_b > rmse_t_b + 8.0:
        raise AssertionError(
            f"Newell lost badly on the spatial shock block ({rmse_n_b:.3f} vs {rmse_t_b:.3f})"
        )


if __name__ == "__main__":
    selftest()
