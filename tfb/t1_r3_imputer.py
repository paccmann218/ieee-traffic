"""CPU R3 specialist prototype and EnKS / LTM-style ensemble smoother.

Full ImputeFormer / PriSTI needs GPU and is gated on Architecture C first beating
the joint metric. This module is the CPU prototype that can run on the current
r7i.2xlarge: a graph Jacobi smoother for 50% holes, and a Newell ensemble
(perturbed vf, w, k_jam) whose mean is an EnKS stand-in.

Never starts a Godunov CTM on 25 m links.

Usage: python tfb/t1_r3_imputer.py selftest | proto [panel]
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "tfb")
import cache
import legal as L
import t1_base as tb
import t1_newell as tn

OUT = L.ARCH_C
N_JACOBI = 20
N_ENS = 10


def shift_t(a, k):
    out = np.full_like(a, np.nan)
    if k > 0:
        out[k:] = a[:-k]
    elif k < 0:
        out[:k] = a[-k:]
    else:
        out = a.copy()
    return out


def shift_l(a, k):
    out = np.full_like(a, np.nan)
    if k > 0:
        out[:, :-k] = a[:, k:]
    elif k < 0:
        out[:, -k:] = a[:, :k]
    else:
        out = a.copy()
    return out


def jacobi_fill(A, ok, n_iter: int = N_JACOBI):
    """4-neighbour Jacobi on missing cells. Observations are frozen. Seed = time interp."""
    seed = tb.interp_day(np.where(ok, A, np.nan))
    X = np.where(ok, A, seed).astype(np.float64)
    for _ in range(n_iter):
        parts = [shift_t(X, 1), shift_t(X, -1), shift_l(X, 1), shift_l(X, -1)]
        stack = np.stack(parts, 0)
        with np.errstate(all="ignore"):
            mean = np.nanmean(stack, axis=0)
        X = np.where(ok, A, np.where(np.isfinite(mean), mean, X))
    return X.astype(np.float32)


def r3_fill_day(V, Q, ok):
    return jacobi_fill(V, ok), jacobi_fill(Q, ok)


def enks_newell_day(V, Q, ok, x, vf, cap, kc, kj, lanes, prior_v, prior_q, n_ens=N_ENS, seed=0):
    """Newell ensemble with FD jitter. Mean is the EnKS prototype; std is uncertainty."""
    rng = np.random.default_rng(seed)
    vs, qs = [], []
    for i in range(n_ens):
        vf_i = vf * rng.normal(1.0, 0.03, vf.shape)
        kj_i = kj * rng.normal(1.0, 0.04, kj.shape)
        cap_i = cap * rng.normal(1.0, 0.03, cap.shape)
        kc_i = cap_i / np.maximum(vf_i, 1.0)
        kj_i = np.maximum(kj_i, kc_i * 1.05)
        v, q, _ = tn.newell_fill_day(V, Q, ok, x, vf_i, cap_i, kc_i, kj_i, lanes, prior_v, prior_q)
        vs.append(v)
        qs.append(q)
    Vmean = np.mean(vs, 0).astype(np.float32)
    Qmean = np.mean(qs, 0).astype(np.float32)
    Vstd = np.std(vs, 0).astype(np.float32)
    return np.where(ok, V, Vmean), np.where(ok, Q, Qmean), Vstd


def proto(panel: str):
    """One-panel prototype: Jacobi vs tinterp vs Newell-EnKS on train R3 days."""
    z = cache.load(panel, "train")
    fd = tn.fd_authority(panel, z["links"])
    V = tb.usable(z, "speed")
    Q = tb.usable(z, "flow")
    ok = np.isfinite(V) & np.isfinite(Q)
    r3 = np.array([str(r) == "R3" for r in z["regime"]])
    if not r3.any():
        print(panel, "no R3 days")
        return
    rows = []
    x = fd.x_km.to_numpy()
    vf, cap = fd.free_speed_kmh.to_numpy(), fd.capacity_vph.to_numpy()
    kc, kj, lanes = fd.critical_density.to_numpy(), fd.k_jam.to_numpy(), fd.lanes.to_numpy()
    truth_v, truth_q = z["speed"], z["flow"]
    tgt = z["target"].astype(bool)
    for d in np.flatnonzero(r3):
        prior_v = tb.interp_day(V[d])
        prior_q = tb.interp_day(Q[d])
        vj, qj = r3_fill_day(V[d], Q[d], ok[d])
        ve, qe, _ = enks_newell_day(
            V[d], Q[d], ok[d], x, vf, cap, kc, kj, lanes, prior_v, prior_q, seed=d,
        )
        m = tgt[d] & np.isfinite(truth_v[d])
        if not m.any():
            continue
        def rmse(p, y):
            return float(np.sqrt(np.mean((p[m] - y[m]) ** 2)))
        rows.append(dict(
            d=int(d),
            tinterp=rmse(prior_v, truth_v[d]),
            jacobi=rmse(vj, truth_v[d]),
            enks=rmse(ve, truth_v[d]),
        ))
    if not rows:
        print(panel, "no R3 targets")
        return
    df = __import__("pandas").DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / f"r3_proto_{panel}.csv", index=False)
    print(panel, df.mean(numeric_only=True).round(3).to_string(), flush=True)
    print("GPU PriSTI/ImputeFormer is not run here (no GPU track until C beats joint J).")


def selftest() -> None:
    V, Q, fd = tn.synthetic_shock(T=40, L=16)
    rng = np.random.default_rng(3)
    ok = rng.random(V.shape) > 0.50
    prior_v = tb.interp_day(np.where(ok, V, np.nan))
    vj = jacobi_fill(V, ok)
    ve, qe, vs = enks_newell_day(
        V, Q, ok, fd["x"], fd["vf"], fd["cap"], fd["kc"], fd["kj"], fd["lanes"],
        prior_v, tb.interp_day(np.where(ok, Q, np.nan)), n_ens=5, seed=0,
    )
    miss = ~ok
    assert np.allclose(vj[ok], V[ok])
    assert np.allclose(ve[ok], V[ok])
    rmse_j = np.sqrt(np.mean((vj[miss] - V[miss]) ** 2))
    rmse_t = np.sqrt(np.mean((prior_v[miss] - V[miss]) ** 2))
    assert vs.shape == V.shape
    print(
        f"t1_r3_imputer selftest OK  jacobi={rmse_j:.3f} tinterp={rmse_t:.3f}  "
        f"enks={np.sqrt(np.mean((ve[miss]-V[miss])**2)):.3f}  (PriSTI deferred to GPU)",
        flush=True,
    )


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "selftest"
    if cmd == "proto":
        proto(sys.argv[2] if len(sys.argv) > 2 else "D12_I5_N")
    else:
        selftest()
