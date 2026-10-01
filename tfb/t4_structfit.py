"""Fit f = b * exp(X theta) to segment counts; report how well low-dim corrections explain counts."""
import sys, numpy as np, pandas as pd
from pathlib import Path
from scipy.optimize import least_squares
from scipy import sparse
REL = Path(sys.argv[1])
PANELS = "D7_I10_E D7_I10_W D7_I210_E D7_I210_W D7_I405_N D7_I405_S D12_I5_N D12_I5_S D12_I405_N D12_I405_S".split()
def setup(panel, split):
    net = REL/"corridors"/panel/"network"
    ps = pd.read_csv(net/"path_set.csv", dtype=str); inc = pd.read_csv(net/"path_link_incidence.csv", dtype=str)
    pid = ps.path_id.tolist(); lid = inc.link_id.drop_duplicates().tolist(); li = {x: i for i, x in enumerate(lid)}; pi = {x: i for i, x in enumerate(pid)}
    A = sparse.csr_matrix((np.ones(len(inc)), (inc.link_id.map(li), inc.path_id.map(pi))), shape=(len(lid), len(pid))); A.data[:] = 1
    cdf = pd.read_csv(REL/"task4"/panel/split/"synthetic_link_counts.csv", dtype={"link_id": str})
    # collapse identical rows of A (same segment) -> unique constraints
    Ac = A[[li[x] for x in cdf.link_id]].toarray(); c = cdf["count"].to_numpy(float)
    _, u = np.unique(Ac, axis=0, return_index=True); Ac, c = Ac[u], c[u]
    b = pd.read_csv(REL/"task4"/panel/split/"synthetic_weak_prior.csv", dtype={"path_id": str}).set_index("path_id").reindex(pid).path_flow.to_numpy(float)
    zo = ps.origin_zone.str[1:].astype(int).to_numpy(); zd = ps.destination_zone.str[1:].astype(int).to_numpy(); nz = max(zo.max(), zd.max()) + 1
    L = ps.link_seq.str.count(";").to_numpy() + 1
    return Ac, c, b, zo / nz, zd / nz, L / L.max(), (zd - zo) / nz
def fit(Ac, c, b, X):
    def res(th): return (Ac @ (b * np.exp(X @ th)) - c) / c.mean()
    r = least_squares(res, np.zeros(X.shape[1]), method="lm")
    f = b * np.exp(X @ r.x); rel = np.abs(Ac @ f - c).sum() / c.sum()
    return r.x, rel, f
rows = []
for panel in PANELS:
    for split in ("train", "validation", "private"):
        Ac, c, b, po, pd_, pl, dz = setup(panel, split)
        one = np.ones_like(b)
        models = {"scale": np.c_[one], "+len": np.c_[one, pl], "+len+po+pd": np.c_[one, pl, po, pd_], "+quad": np.c_[one, pl, po, pd_, pl**2, po**2, pd_**2, po*pd_]}
        r = dict(panel=panel, split=split, n_con=len(c), prior_L1=np.abs(Ac @ b - c).sum() / c.sum())
        for k, X in models.items():
            th, rel, f = fit(Ac, c, b, X); r[k] = rel
            if k == "+len+po+pd": r["theta"] = np.round(th, 2).tolist()
        rows.append(r)
pd.set_option("display.width", 250); print(pd.DataFrame(rows).round(4).to_string())
