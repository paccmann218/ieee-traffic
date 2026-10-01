"""Task 4 variants. Usage: t4_build.py REL VARIANT OUT.csv

V1 (NNLS λ=0.05 on the split's synthetic_weak_prior.csv) is frozen with J1.
Do not read base_od.csv (private-month prior). Research variants require
TFB_ALLOW_T4_VARIANT=1.
"""
import os, sys, numpy as np, pandas as pd
from pathlib import Path
from scipy import sparse
from scipy.optimize import least_squares, nnls
REL = Path(sys.argv[1]); VAR = sys.argv[2]; OUT = Path(sys.argv[3])
if VAR != "v1" and os.environ.get("TFB_ALLOW_T4_VARIANT", "0") != "1":
    raise SystemExit("Task 4 is frozen at V1. Set TFB_ALLOW_T4_VARIANT=1 to run research variants.")
PANELS = "D7_I10_E D7_I10_W D7_I210_E D7_I210_W D7_I405_N D7_I405_S D12_I5_N D12_I5_S D12_I405_N D12_I405_S".split()

def setup(panel, split):
    net = REL/"corridors"/panel/"network"
    ps = pd.read_csv(net/"path_set.csv", dtype=str); inc = pd.read_csv(net/"path_link_incidence.csv", dtype=str)
    pid = ps.path_id.tolist(); lid = inc.link_id.drop_duplicates().tolist(); li = {x: i for i, x in enumerate(lid)}; pi = {x: i for i, x in enumerate(pid)}
    A = sparse.csr_matrix((np.ones(len(inc)), (inc.link_id.map(li), inc.path_id.map(pi))), shape=(len(lid), len(pid))); A.data[:] = 1
    cdf = pd.read_csv(REL/"task4"/panel/split/"synthetic_link_counts.csv", dtype={"link_id": str})
    Am = A[[li[x] for x in cdf.link_id]].toarray(); c = cdf["count"].to_numpy(float)
    bdf = pd.read_csv(REL/"task4"/panel/split/"synthetic_weak_prior.csv", dtype={"path_id": str})
    # landmine: never load corridors/.../base_od.csv (private-month prior)
    b = bdf.set_index("path_id").reindex(pid).path_flow.to_numpy(float)
    zo = ps.origin_zone.str[1:].astype(int).to_numpy(); zd = ps.destination_zone.str[1:].astype(int).to_numpy(); nz = max(zo.max(), zd.max()) + 1
    L = (ps.link_seq.str.count(";").to_numpy() + 1).astype(float)
    return ps, Am, c, b, zo / nz, zd / nz, L / L.max(), str(bdf.departure_time.iloc[0])

def l2_exact(Am, c, b, lam=0.05):
    # organizer-style NNLS: min ||A f - c||^2 + lam ||f - b||^2, f >= 0
    aa = np.vstack([Am, np.sqrt(lam) * np.eye(Am.shape[1])]); bb = np.concatenate([c, np.sqrt(lam) * b])
    return nnls(aa, bb, maxiter=50 * Am.shape[1])[0]

def smooth_corr(Am, c, b, po, pdd, pl):
    X = np.c_[np.ones_like(b), pl, po, pdd, pl**2, po**2, pdd**2, po * pdd]
    _, u = np.unique(Am, axis=0, return_index=True); Au, cu = Am[u], c[u]
    r = least_squares(lambda th: (Au @ (b * np.exp(X @ th)) - cu) / cu.mean(), np.zeros(X.shape[1]), method="lm")
    return b * np.exp(X @ r.x)

def gravity_seed(ps, b, nb=12, panel=None):
    """Fit log b = FE_o + FE_d + piecewise-linear g(path km); return exp(g(km)) (FE are re-set by the KL fit)."""
    from scipy.sparse.linalg import lsqr
    links = pd.read_csv(REL/"corridors"/panel/"network"/"links.csv", dtype={"link_id": str}).drop_duplicates("link_id").set_index("link_id")
    km = ps.link_seq.str.split(";").apply(lambda s: pd.to_numeric(links.length_km.reindex(s), errors="coerce").fillna(0).sum()).to_numpy()
    zo = ps.origin_zone.str[1:].astype(int).to_numpy(); zd = ps.destination_zone.str[1:].astype(int).to_numpy(); n = len(b)
    knots = np.quantile(km, np.linspace(0, 1, nb)); B = np.stack([np.interp(km, knots, np.eye(nb)[k]) for k in range(nb)], 1)
    Mo = sparse.csr_matrix((np.ones(n), (np.arange(n), zo)), shape=(n, zo.max() + 1)); Md = sparse.csr_matrix((np.ones(n), (np.arange(n), zd)), shape=(n, zd.max() + 1))
    X = sparse.hstack([Mo, Md, sparse.csr_matrix(B)]).tocsr()
    coef = lsqr(X, np.log(b), atol=1e-10, btol=1e-10, iter_lim=20000)[0]
    return np.exp(X @ coef)


def kl_exact(Am, c, g, iters=2000):
    """Entropy projection: min sum f log(f/g) - f  s.t. A f = c  ->  f = g * exp(-A^T lam). Solved in the dual."""
    from scipy.optimize import minimize
    _, u = np.unique(Am, axis=0, return_index=True); Au, cu = Am[u], c[u]
    s = cu.mean()
    def dual(lam):
        f = g * np.exp(-(Au.T @ lam)); return (f.sum() + cu @ lam) / s, (cu - Au @ f) / s
    r = minimize(dual, np.zeros(len(cu)), jac=True, method="L-BFGS-B", options=dict(maxiter=iters, gtol=1e-12, ftol=1e-15))
    return g * np.exp(-(Au.T @ r.x))


out = []
for panel in PANELS:
    for split in ("validation", "private"):
        ps, Am, c, b, po, pdd, pl, dep = setup(panel, split)
        if VAR == "v1": f = l2_exact(Am, c, b)
        elif VAR == "v2": g = smooth_corr(Am, c, b, po, pdd, pl); f = l2_exact(Am, c, g)
        elif VAR == "v3": g = smooth_corr(Am, c, b, po, pdd, pl); f = kl_exact(Am, c, g)
        elif VAR == "v4": f = kl_exact(Am, c, b)
        elif VAR == "v5": f = kl_exact(Am, c, gravity_seed(ps, b, panel=panel))
        elif VAR.startswith("v6_"):  # geometric shrink of prior toward gravity: seed = b^(1-w) * grav^w
            w = float(VAR[3:]); f = kl_exact(Am, c, b ** (1 - w) * gravity_seed(ps, b, panel=panel) ** w)
        elif VAR == "prior": f = b.copy()
        else: raise SystemExit(VAR)
        f = np.maximum(f, 0)
        _, u = np.unique(Am, axis=0, return_index=True)
        print(f"{panel:11s} {split:10s} S_link={1-np.abs(Am@f-c).sum()/c.sum():.4f} D/b={np.abs(f-b).sum()/b.sum():.3f} zeros={np.mean(f<1e-9):.2f}", flush=True)
        x = ps[["path_id", "origin_zone", "destination_zone"]].copy(); x.insert(0, "departure_time", dep); x.insert(0, "panel", panel); x["path_flow"] = f
        out.append(x)
pd.concat(out).to_csv(OUT, index=False); print("wrote", OUT)
