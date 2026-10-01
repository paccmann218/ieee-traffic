import sys, numpy as np, pandas as pd
from pathlib import Path
REL = Path(sys.argv[1]); panel = sys.argv[2] if len(sys.argv) > 2 else "D7_I10_E"; split = sys.argv[3] if len(sys.argv) > 3 else "validation"
pd.set_option("display.width", 250)
net = REL/"corridors"/panel/"network"
print(pd.read_csv(net/"ramp_attachment_map.csv").head(5).to_string())
print(pd.read_csv(net/"lwr_mainline_topology.csv").head(3).to_string())
files = sorted((REL/"corridors"/panel/split/"ramp_states").glob("**/*.parquet")); print(len(files), "ramp files", files[0])
r = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True); print(r.dtypes); print(r.head(3).to_string())
ts = pd.to_datetime(r.timestamp, utc=True); pm = (ts.dt.hour >= 15) & (ts.dt.hour < 19)
g = r[pm & r.is_score_eligible.astype(bool)].groupby(["ramp_link_id", "ramp_type"]).flow_vph.agg(["mean", "count"]).reset_index()
print(g.head(10).to_string()); print(len(g), "ramps")
g.to_csv(f"/tmp/ramp_pm_{panel}_{split}.csv", index=False)
# consistency: segment count differences vs ramp flows, following path order of zone Z0 -> Z last
paths = pd.read_csv(net/"path_set.csv", dtype=str)
inc = pd.read_csv(net/"path_link_incidence.csv", dtype=str)
print(inc[inc.link_id.str.startswith("R")].link_id.drop_duplicates().head(10).tolist())
