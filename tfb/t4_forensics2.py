import sys, numpy as np, pandas as pd
sys.path.insert(0, "tfb"); from t4_forensics import load, PANELS
from pathlib import Path
REL = Path(sys.argv[1])
import t4_forensics; t4_forensics.REL = REL
pd.set_option("display.width", 250)
for panel in ["D7_I10_E", "D12_I405_S"]:
    paths, inc, pid, lid, li, A = load(panel)
    ltype = pd.Series(lid).str.extract(r"^([A-Z]+)")[0].to_numpy()
    print("=====", panel, "link types (all):", pd.Series(ltype).value_counts().to_dict())
    for s in ("validation",):
        b = pd.read_csv(REL/"task4"/panel/s/"synthetic_weak_prior.csv", dtype={"path_id": str}).set_index("path_id").reindex(pid).path_flow.to_numpy(float)
        cdf = pd.read_csv(REL/"task4"/panel/s/"synthetic_link_counts.csv", dtype={"link_id": str})
        r = np.array([li[x] for x in cdf.link_id]); c = cdf["count"].to_numpy(float)
        Ab = A[r] @ b; t = ltype[r]
        df = pd.DataFrame(dict(link=cdf.link_id, type=t, c=c, Ab=Ab, ratio=Ab/c))
        print(df.groupby("type").ratio.describe().round(3))
        print("counted link types:", pd.Series(t).value_counts().to_dict())
        # zones and ramps
        print(paths.origin_zone.nunique(), "origins", paths.destination_zone.nunique(), "dests")
        # row/col sums vs ramp counts
        on = df[df.type == "RON"]; off = df[df.type == "ROFF"]
        print("on-ramp ratio sample:", on.ratio.round(2).head(10).tolist()); print("off-ramp ratio sample:", off.ratio.round(2).head(10).tolist())
        ml = df[df.type.str.startswith("L")]
        print(ml.head(20).to_string())
