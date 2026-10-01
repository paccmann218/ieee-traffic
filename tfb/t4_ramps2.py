import sys, numpy as np, pandas as pd
from pathlib import Path
REL = Path(sys.argv[1]); panel = sys.argv[2]; split = sys.argv[3]
pd.set_option("display.width", 250)
net = REL/"corridors"/panel/"network"
topo = pd.read_csv(net/"lwr_mainline_topology.csv", dtype=str)
order = topo.sort_values("order_index", key=lambda s: s.astype(int)).link_id.tolist()
c = pd.read_csv(REL/"task4"/panel/split/"synthetic_link_counts.csv", dtype={"link_id": str}).set_index("link_id")["count"]
# mainline PM mean flow per link from masked layer (eligible observed)
d = "mainline_states" if split == "train" else "mainline_states_masked"
files = sorted((REL/"corridors"/panel/split/d).glob("**/*.parquet"))
m = pd.concat([pd.read_parquet(f, columns=["timestamp","link_id","station_id","flow_vph","speed_kmh","is_score_eligible"]) for f in files])
ts = pd.to_datetime(m.timestamp, utc=True); pm = (ts.dt.hour >= 15) & (ts.dt.hour < 19)
g = m[pm & m.is_score_eligible.astype(bool) & m.flow_vph.notna()].groupby("link_id").flow_vph.mean()
allday = m[m.is_score_eligible.astype(bool) & m.flow_vph.notna()].groupby("link_id").flow_vph.mean()
inc = pd.read_csv(net/"path_link_incidence.csv", dtype=str); ps = pd.read_csv(net/"path_set.csv", dtype=str)
# zone attachment: for each zone Zk, first mainline link of paths originating at Zk
first = ps.assign(first=ps.link_seq.str.split(";").str[0]).groupby("origin_zone")["first"].first()
last = ps.assign(last=ps.link_seq.str.split(";").str[-1]).groupby("destination_zone")["last"].first()
rows = []
for i, l in enumerate(order):
    rows.append(dict(i=i, link=l, count=c.get(l, np.nan), pm_obs=g.get(l, np.nan), allday=allday.get(l, np.nan),
                     orig_here=",".join(first[first == l].index), dest_here=",".join(last[last == l].index),
                     onr=topo.set_index("link_id").loc[l, "on_ramp_link_ids"], offr=topo.set_index("link_id").loc[l, "off_ramp_link_ids"]))
df = pd.DataFrame(rows)
print(df.head(60).to_string())
x = df.dropna(subset=["count", "pm_obs"]); print("count/pm_obs ratio:", (x["count"]/x.pm_obs).describe().round(4).to_dict())
