"""Task 4 forensics: structure of A, prior b, counts c. Read-only analysis."""
import sys, numpy as np, pandas as pd
from pathlib import Path
from scipy import sparse
REL = Path(sys.argv[1] if len(sys.argv) > 1 else "D:/Desktop/Andrea/kaggle/data/small")
PANELS = "D7_I10_E D7_I10_W D7_I210_E D7_I210_W D7_I405_N D7_I405_S D12_I5_N D12_I5_S D12_I405_N D12_I405_S".split()

def load(panel):
    net = REL / "corridors" / panel / "network"
    paths = pd.read_csv(net / "path_set.csv", dtype=str)
    inc = pd.read_csv(net / "path_link_incidence.csv", dtype=str)
    pid = paths.path_id.tolist(); lid = inc.link_id.drop_duplicates().tolist()
    pi = {x: i for i, x in enumerate(pid)}; li = {x: i for i, x in enumerate(lid)}
    A = sparse.csr_matrix((np.ones(len(inc)), (inc.link_id.map(li), inc.path_id.map(pi))), shape=(len(lid), len(pid)))
    A.data[:] = 1.0
    return paths, inc, pid, lid, li, A

if __name__ == "__main__":
    p0 = PANELS[0]
    paths, inc, pid, lid, li, A = load(p0)
    print("path_set cols:", list(paths.columns)); print(paths.head(3).to_string())
    print("incidence cols:", list(inc.columns)); print(inc.head(3).to_string())
    for s in ("train", "validation", "private"):
        b = pd.read_csv(REL / "task4" / p0 / s / "synthetic_weak_prior.csv"); c = pd.read_csv(REL / "task4" / p0 / s / "synthetic_link_counts.csv")
        print(s, "prior cols", list(b.columns), "counts cols", list(c.columns)); print(b.head(3).to_string()); print(c.head(3).to_string())
    rows = []
    for panel in PANELS:
        paths, inc, pid, lid, li, A = load(panel)
        for s in ("train", "validation", "private"):
            bdf = pd.read_csv(REL / "task4" / panel / s / "synthetic_weak_prior.csv", dtype={"path_id": str})
            cdf = pd.read_csv(REL / "task4" / panel / s / "synthetic_link_counts.csv", dtype={"link_id": str})
            b = bdf.set_index("path_id").reindex(pid).path_flow.to_numpy(float)
            r = np.array([li[x] for x in cdf.link_id if x in li]); c = cdf.set_index("link_id").reindex([lid[i] for i in r])["count"].to_numpy(float)
            Am = A[r]; d = Am @ b - c
            frac = np.abs(b - np.round(b))
            rows.append(dict(panel=panel, split=s, paths=len(pid), links=len(lid), counted=len(r), sum_b=b.sum(),
                             b_min=b.min(), b_q01=np.quantile(b, .01), b_med=np.median(b), b_max=b.max(),
                             b_int_frac=float((frac < 1e-9).mean()), c_int_frac=float((np.abs(c - np.round(c)) < 1e-9).mean()),
                             deficit_mean_rel=float(d.sum() / c.sum()), deficit_neg_frac=float((d < 0).mean()),
                             pathlen_mean=float(np.asarray(A.sum(0)).mean())))
    pd.set_option("display.width", 250); print(pd.DataFrame(rows).round(4).to_string())
