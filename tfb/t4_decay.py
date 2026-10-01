"""Which distance variable explains the prior's distortion? Fit f = b*exp(a + g(dist)) to segment counts."""
import sys, numpy as np, pandas as pd
from pathlib import Path
from scipy.optimize import least_squares
from scipy import sparse
REL = Path(sys.argv[1])
PANELS = "D7_I10_E D7_I10_W D7_I210_E D7_I210_W D7_I405_N D7_I405_S D12_I5_N D12_I5_S D12_I405_N D12_I405_S".split()


def setup(panel, split):
    net = REL/"corridors"/panel/"network"
    ps = pd.read_csv(net/"path_set.csv", dtype=str); inc = pd.read_csv(net/"path_link_incidence.csv", dtype=str)
    links = pd.read_csv(net/"links.csv", dtype={"link_id": str}).drop_duplicates("link_id").set_index("link_id")
    pid = ps.path_id.tolist(); lid = inc.link_id.drop_duplicates().tolist(); li = {x: i for i, x in enumerate(lid)}; pi = {x: i for i, x in enumerate(pid)}
    A = sparse.csr_matrix((np.ones(len(inc)), (inc.link_id.map(li), inc.path_id.map(pi))), shape=(len(lid), len(pid))); A.data[:] = 1
    cdf = pd.read_csv(REL/"task4"/panel/split/"synthetic_link_counts.csv", dtype={"link_id": str})
    Ac = A[[li[x] for x in cdf.link_id]].toarray(); c = cdf["count"].to_numpy(float)
    _, u = np.unique(Ac, axis=0, return_index=True); Ac, c = Ac[u], c[u]
    b = pd.read_csv(REL/"task4"/panel/split/"synthetic_weak_prior.csv", dtype={"path_id": str}).set_index("path_id").reindex(pid).path_flow.to_numpy(float)
    km = ps.link_seq.str.split(";").apply(lambda s: pd.to_numeric(links.length_km.reindex(s), errors="coerce").fillna(0).sum()).to_numpy()
    zo = ps.origin_zone.str[1:].astype(int).to_numpy(); zd = ps.destination_zone.str[1:].astype(int).to_numpy()
    return Ac, c, b, km, (zd - zo).astype(float), zo, zd


def fit(Ac, c, b, X):
    r = least_squares(lambda th: (Ac @ (b * np.exp(X @ th)) - c) / c.mean(), np.zeros(X.shape[1]), method="lm")
    f = b * np.exp(X @ r.x); return r.x, np.abs(Ac @ f - c).sum() / c.sum()


rows = []
for panel in PANELS:
    for split in ("train", "validation", "private"):
        Ac, c, b, km, dz, zo, zd = setup(panel, split); one = np.ones_like(b); nz = zd.max() + 1
        r = dict(panel=panel, split=split, km_max=km.max().round(1))
        for name, X in {"km": np.c_[one, km], "km2": np.c_[one, km, km**2], "dz": np.c_[one, dz], "logkm": np.c_[one, np.log(km + 0.1)],
                        "km+pos": np.c_[one, km, zo / nz]}.items():
            th, e = fit(Ac, c, b, X); r[name] = round(e, 4); r["th_" + name] = np.round(th, 4).tolist()
        rows.append(r)
pd.set_option("display.width", 300); pd.set_option("display.max_colwidth", 60)
print(pd.DataFrame(rows).to_string())
