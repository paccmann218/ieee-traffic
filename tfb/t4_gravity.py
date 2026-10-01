"""Is the prior = u_i * w_j * exp(g(d_ij)) * lognormal noise?  Fit log b = FE_o + FE_d + spline(d), inspect residuals.

d candidates: path km, number of links, zone gap. Reports R2, residual sd, cross-split correlation of residuals and of g.
"""
import sys, numpy as np, pandas as pd
from pathlib import Path
from scipy import sparse
from scipy.sparse.linalg import lsqr
REL = Path(sys.argv[1])
PANELS = "D7_I10_E D7_I10_W D7_I210_E D7_I210_W D7_I405_N D7_I405_S D12_I5_N D12_I5_S D12_I405_N D12_I405_S".split()


def dist(panel):
    net = REL/"corridors"/panel/"network"
    ps = pd.read_csv(net/"path_set.csv", dtype=str)
    links = pd.read_csv(net/"links.csv", dtype={"link_id": str}).drop_duplicates("link_id").set_index("link_id")
    km = ps.link_seq.str.split(";").apply(lambda s: pd.to_numeric(links.length_km.reindex(s), errors="coerce").fillna(0).sum()).to_numpy()
    zo = ps.origin_zone.str[1:].astype(int).to_numpy(); zd = ps.destination_zone.str[1:].astype(int).to_numpy()
    return ps, km, zo, zd


def design(zo, zd, x, nb=12):
    n = len(zo); no, nd = zo.max() + 1, zd.max() + 1
    knots = np.quantile(x, np.linspace(0, 1, nb))
    B = np.stack([np.interp(x, knots, np.eye(nb)[k]) for k in range(nb)], 1)  # piecewise-linear hat basis
    Mo = sparse.csr_matrix((np.ones(n), (np.arange(n), zo)), shape=(n, no)); Md = sparse.csr_matrix((np.ones(n), (np.arange(n), zd)), shape=(n, nd))
    return sparse.hstack([Mo, Md, sparse.csr_matrix(B)]).tocsr(), knots, no, nd


rows = []
for panel in PANELS:
    ps, km, zo, zd = dist(panel)
    pid = ps.path_id.tolist(); res = {}; gs = {}
    for split in ("train", "validation", "private"):
        b = pd.read_csv(REL/"task4"/panel/split/"synthetic_weak_prior.csv", dtype={"path_id": str}).set_index("path_id").reindex(pid).path_flow.to_numpy(float)
        y = np.log(b); X, knots, no, nd = design(zo, zd, km)
        coef = lsqr(X, y, atol=1e-10, btol=1e-10, iter_lim=20000)[0]
        fit = X @ coef; r = y - fit; res[split] = r; gs[split] = coef[no + nd:]
        rows.append(dict(panel=panel, split=split, R2=1 - r.var() / y.var(), sd=r.std(), skew=pd.Series(r).skew(), kurt=pd.Series(r).kurt(),
                         corr_r_km=np.corrcoef(r, km)[0, 1], g=np.round(gs[split] - gs[split][0], 2).tolist()[:8]))
    for a, bb in (("train", "validation"), ("validation", "private"), ("train", "private")):
        rows[-1][f"rcorr_{a[:2]}{bb[:2]}"] = np.corrcoef(res[a], res[bb])[0, 1]
pd.set_option("display.width", 300); pd.set_option("display.max_colwidth", 80)
print(pd.DataFrame(rows).round(3).to_string())
