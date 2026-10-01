import sys, numpy as np, pandas as pd
from pathlib import Path
REL = Path(sys.argv[1]); panel = sys.argv[2] if len(sys.argv) > 2 else "D12_I5_N"
net = REL/"corridors"/panel/"network"
topo = pd.read_csv(net/"lwr_mainline_topology.csv", dtype=str)
order = topo.sort_values("order_index", key=lambda s: s.astype(int)).link_id.tolist()
hasramp = topo.set_index("link_id")[["on_ramp_link_ids", "off_ramp_link_ids"]].notna().any(axis=1)
files = sorted((REL/"corridors"/panel/"train"/"mainline_states").glob("**/*.parquet"))[:20]
t = pd.concat([pd.read_parquet(f, columns=["timestamp","station_id","link_id","speed_kmh","flow_vph","pct_observed"]) for f in files])
print("stations per link:", t.groupby("link_id").station_id.nunique().value_counts().to_dict(), "links in data", t.link_id.nunique(), "topo links", len(order))
g = t.groupby(["timestamp","link_id"]).agg(v=("speed_kmh","mean"), q=("flow_vph","mean"), p=("pct_observed","min")).reset_index()
Q = g.pivot(index="timestamp", columns="link_id", values="q"); V = g.pivot(index="timestamp", columns="link_id", values="v"); P = g.pivot(index="timestamp", columns="link_id", values="p")
cols = [l for l in order if l in Q.columns]; Q, V, P = Q[cols], V[cols], P[cols]
ok = (P >= 75)
Qo = Q.where(ok); Vo = V.where(ok)
# neighbor differences
dq_space = (Qo.values[:, 1:] - Qo.values[:, :-1]); dq_time = Qo.values[1:, :] - Qo.values[:-1, :]
nr = ~hasramp.reindex(cols).fillna(False).values[1:]   # downstream link has no ramp
print("flow diff adjacent links (no ramp) sd:", np.nanstd(dq_space[:, nr]).round(2), " same link consecutive t sd:", np.nanstd(dq_time).round(2))
dv_space = Vo.values[:, 1:] - Vo.values[:, :-1]; dv_time = Vo.values[1:] - Vo.values[:-1]
print("speed diff adjacent sd:", np.nanstd(dv_space).round(3), " time sd:", np.nanstd(dv_time).round(3))
# exact equality check
print("frac exactly equal adjacent flows:", np.nanmean(np.isclose(dq_space, 0, atol=1e-6)).round(4))
# how many distinct links have identical flows (link shares same station?)
print(Q.iloc[100:103, :8].round(2).to_string()); print(V.iloc[100:103, :8].round(2).to_string())
