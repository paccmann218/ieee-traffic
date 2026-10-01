"""Observation-consistent conservation fill on a GBDT/tinterp prior (Architecture C cheap path).

Task 3 scores a mixed field: submission at Task 1 targets, released observations
elsewhere. The organizer flux is the projection of that observation field, so a
fill that restores N-curve interpolation at holes (without touching observations)
is the joint RMSE+LWR objective. Topology flux is not the RHS.

Time interpolation of N is refused across gaps longer than GAP_DISABLE slots
(the 90-minute Task 2 hole). Those cells keep the Newell/spatial fill.

Usage: python tfb/t1_project.py selftest
"""
from __future__ import annotations

import sys

import numpy as np

sys.path.insert(0, "tfb")
import t1_base as tb
import t1_near_window as nw
import t1_newell as tn

DT_H = tn.DT_H
ALPHAS = (0.0, 0.25, 0.5, 0.75, 1.0)


def interp_day_gap_limit(a, max_gap: int = nw.GAP_DISABLE):
    """Linear time interp per column, but do not interpolate across large holes."""
    T, L = a.shape
    out = np.full_like(a, np.nan, dtype=np.float64)
    t = np.arange(T)
    for l in range(L):
        ok = np.isfinite(a[:, l])
        if ok.sum() == 0:
            continue
        filled = np.interp(t, t[ok], a[ok, l])
        if ok.sum() == 1:
            out[:, l] = np.where(ok, a[:, l], np.nan)
            continue
        prev = np.maximum.accumulate(np.where(ok, t, -1))
        nxt = np.minimum.accumulate(np.where(ok, t, 10**6)[::-1])[::-1]
        gap = nxt - prev
        out[:, l] = np.where(ok | ((gap <= max_gap) & (prev >= 0) & (nxt < T)), filled, np.nan)
        out[ok, l] = a[ok, l]
    return out


def n_from_vq(V, Q, length):
    k = Q / np.maximum(V, 1.0)
    return k * length[None, :]


def project_day(V, Q, ok, length, vf, cap, kc, kj, prior_v, prior_q, alpha=0.5):
    """Fix observations; replace holes by (1-α) prior + α FD(N_time_interp)."""
    V = np.asarray(V, np.float64)
    Q = np.asarray(Q, np.float64)
    N_obs = np.where(ok, n_from_vq(V, Q, length), np.nan)
    N_i = interp_day_gap_limit(N_obs)
    k_i = N_i / np.maximum(length[None, :], 1e-6)
    v_fd, q_fd = tn.fd_vq_from_k(k_i, vf[None, :], cap[None, :], kc[None, :], kj[None, :])
    pv = np.asarray(prior_v, np.float64)
    pq = np.asarray(prior_q, np.float64)
    have = np.isfinite(v_fd) & np.isfinite(q_fd)
    v_b = (1.0 - alpha) * pv + alpha * v_fd
    q_b = (1.0 - alpha) * pq + alpha * q_fd
    Vf = np.where(ok, V, np.where(have, v_b, pv))
    Qf = np.where(ok, Q, np.where(have, q_b, pq))
    return np.clip(Vf, tn.V_MIN, tn.V_MAX).astype(np.float32), np.clip(Qf, 0.0, None).astype(np.float32)


def blend_fields(a_v, a_q, b_v, b_q, ok, alpha):
    """Observed stay; holes are (1-α) a + α b."""
    v = np.where(ok, a_v, (1.0 - alpha) * a_v + alpha * b_v)
    q = np.where(ok, a_q, (1.0 - alpha) * a_q + alpha * b_q)
    return np.clip(v, tn.V_MIN, tn.V_MAX).astype(np.float32), np.clip(q, 0.0, None).astype(np.float32)


def arch_c_day(V, Q, ok, x, length, vf, cap, kc, kj, lanes, prior_v, prior_q, alpha=0.5):
    """Newell fill, then N-curve FD, then blend that physics field toward the untouched prior.

    alpha=0 keeps the prior (tinterp or GBDT). alpha=1 replaces holes with the
    physics fill. Observations are never changed.
    """
    Vn, Qn, meta = tn.newell_fill_day(
        V, Q, ok, x, vf, cap, kc, kj, lanes, prior_v, prior_q,
    )
    Vp, Qp = project_day(V, Q, ok, length, vf, cap, kc, kj, Vn, Qn, alpha=1.0)
    return blend_fields(prior_v, prior_q, Vp, Qp, ok, alpha) + (meta,)


def arch_c_split(panel, z, prior_v, prior_q, alpha=0.5, use_masked=True):
    fd = tn.fd_authority(panel, z["links"])
    x = fd.x_km.to_numpy()
    length = fd.length_km.to_numpy()
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
        PV[d], PQ[d], m = arch_c_day(
            V[d], Q[d], ok[d], x, length, vf, cap, kc, kj, lanes,
            prior_v[d], prior_q[d], alpha=alpha,
        )
        for k in meta:
            meta[k] += m[k]
    return PV, PQ, fd, meta


def conservation_residual(V, Q, ok, length):
    """Mean |ΔN_mixed - ΔN_obs_interp| on consecutive observed-or-filled slots."""
    N = n_from_vq(V, Q, length)
    dN = np.diff(N, axis=0)
    return float(np.nanmean(np.abs(dN)))


def selftest() -> None:
    V, Q, fd = tn.synthetic_shock()
    T, L = V.shape
    rng = np.random.default_rng(2)
    ok = rng.random((T, L)) > 0.4
    ok[20, 8:16] = False
    prior_v = tb.interp_day(np.where(ok, V, np.nan))
    prior_q = tb.interp_day(np.where(ok, Q, np.nan))
    Vp, Qp, meta = arch_c_day(
        V, Q, ok, fd["x"], fd["length"], fd["vf"], fd["cap"], fd["kc"], fd["kj"], fd["lanes"],
        prior_v, prior_q, alpha=0.5,
    )
    assert np.allclose(Vp[ok], V[ok], atol=1e-5), "projection must not touch observations"
    miss = ~ok
    rmse_p = np.sqrt(np.mean((Vp[miss] - V[miss]) ** 2))
    rmse_t = np.sqrt(np.mean((prior_v[miss] - V[miss]) ** 2))
    # N-interp across a 1-slot hole should be defined
    a = np.full((10, 3), np.nan)
    a[[0, 2, 9], 0] = [0.0, 2.0, 9.0]
    ni = interp_day_gap_limit(a, max_gap=3)
    assert np.isfinite(ni[1, 0])
    a2 = np.full((20, 1), np.nan)
    a2[0, 0] = 0.0
    a2[19, 0] = 19.0
    ni2 = interp_day_gap_limit(a2, max_gap=12)
    assert not np.isfinite(ni2[10, 0]), "must refuse interpolation across a 90-min-scale gap"
    print(
        f"t1_project selftest OK  rmse_hole project={rmse_p:.3f} tinterp={rmse_t:.3f}  meta={meta}",
        flush=True,
    )


if __name__ == "__main__":
    selftest()
