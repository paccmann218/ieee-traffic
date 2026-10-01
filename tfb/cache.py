"""Dense per-panel caches. build(panel, split) -> npz with arrays [D, 288, L] in corridor (topology) order.

Channels: speed, flow, occ (observed layer where available), mspeed/mflow/mocc (masked layer), pct, elig, target.
Stations: one per link in this release (asserted). Ramps: [D, 288, R] flow/pct/elig with ramp meta.
"""
from __future__ import annotations
import os, sys, json, numpy as np, pandas as pd
from pathlib import Path
REL = Path(os.environ.get("TFB_REL", str(Path.home() / "tfb/data/kaggle_public")))
OUT = Path(os.environ.get("TFB_CACHE", str(Path.home() / "tfb/cache")))
PANELS = "D7_I10_E D7_I10_W D7_I210_E D7_I210_W D7_I405_N D7_I405_S D12_I5_N D12_I5_S D12_I405_N D12_I405_S".split()

def order(panel):
    topo = pd.read_csv(REL/"corridors"/panel/"network"/"lwr_mainline_topology.csv", dtype=str)
    return topo.sort_values("order_index", key=lambda s: s.astype(int)).link_id.tolist()

def _read(files, cols):
    return pd.concat([pd.read_parquet(f, columns=cols) for f in files], ignore_index=True) if files else None

def build(panel, split):
    links = order(panel); li = {x: i for i, x in enumerate(links)}; L = len(links)
    pdir = REL/"corridors"/panel/split
    mk = _read(sorted((pdir/"mainline_states_masked").glob("**/*.parquet")),
               ["date","timestamp","station_id","link_id","speed_kmh","flow_vph","occupancy","pct_observed","is_score_eligible","mask_regime"] )
    assert mk.groupby("link_id").station_id.nunique().max() == 1, "multi-station links"
    dates = sorted(mk.date.unique()); di = {d: i for i, d in enumerate(dates)}; D = len(dates)
    ts = mk.timestamp.str
    slot = (ts[11:13].astype(int) * 12 + ts[14:16].astype(int) // 5).to_numpy()
    d_ = mk.date.map(di).to_numpy(); l_ = mk.link_id.map(li).to_numpy()
    def grid(vals, fill=np.nan, dtype=np.float32):
        a = np.full((D, 288, L), fill, dtype); a[d_, slot, l_] = vals; return a
    z = dict(links=np.array(links), dates=np.array(dates))
    z["mspeed"] = grid(mk.speed_kmh.to_numpy(np.float32)); z["mflow"] = grid(mk.flow_vph.to_numpy(np.float32)); z["mocc"] = grid(mk.occupancy.to_numpy(np.float32))
    z["pct"] = grid(mk.pct_observed.to_numpy(np.int16), 0, np.int16); z["elig"] = grid(mk.is_score_eligible.to_numpy(np.int8), 0, np.int8)
    reg = mk.groupby("date").mask_regime.first(); z["regime"] = np.array([reg[d] for d in dates])
    # Task 1 targets from the template
    tp = REL/"task1"/panel/split/"sample_submission_state.csv"
    t = pd.read_csv(tp, usecols=["timestamp","station_id","link_id"], dtype=str)
    tgt = np.zeros((D, 288, L), np.int8)
    tt = t.timestamp.str; tgt[t.timestamp.str[:10].map(di).to_numpy(), (tt[11:13].astype(int)*12 + tt[14:16].astype(int)//5).to_numpy(), t.link_id.map(li).to_numpy()] = 1
    z["target"] = tgt
    if split == "train":
        tr = _read(sorted((pdir/"mainline_states").glob("**/*.parquet")), ["date","timestamp","link_id","speed_kmh","flow_vph","occupancy"])
        ts2 = tr.timestamp.str; s2 = (ts2[11:13].astype(int)*12 + ts2[14:16].astype(int)//5).to_numpy(); d2 = tr.date.map(di).to_numpy(); l2 = tr.link_id.map(li).to_numpy()
        for k, c in (("speed","speed_kmh"),("flow","flow_vph"),("occ","occupancy")):
            a = np.full((D, 288, L), np.nan, np.float32); a[d2, s2, l2] = tr[c].to_numpy(np.float32); z[k] = a
    # ramps
    rp = _read(sorted((pdir/"ramp_states").glob("**/*.parquet")), ["date","timestamp","ramp_link_id","ramp_type","flow_vph","pct_observed","is_score_eligible"])
    rids = sorted(rp.ramp_link_id.unique()); ri = {x: i for i, x in enumerate(rids)}; R = len(rids)
    ts3 = rp.timestamp.str; s3 = (ts3[11:13].astype(int)*12 + ts3[14:16].astype(int)//5).to_numpy(); d3 = rp.date.map(di).to_numpy(); r3 = rp.ramp_link_id.map(ri).to_numpy()
    a = np.full((D, 288, R), np.nan, np.float32); a[d3, s3, r3] = rp.flow_vph.to_numpy(np.float32); z["rflow"] = a
    a = np.zeros((D, 288, R), np.int8); a[d3, s3, r3] = rp.is_score_eligible.to_numpy(np.int8); z["relig"] = a
    rt = rp.groupby("ramp_link_id").ramp_type.first(); z["ramp_ids"] = np.array(rids); z["ramp_type"] = np.array([rt[r] for r in rids])
    OUT.mkdir(parents=True, exist_ok=True); np.savez(OUT/f"{panel}_{split}.npz", **z)
    print(panel, split, "D", D, "L", L, "R", R, "targets", int(tgt.sum()), flush=True)

def load(panel, split):
    return dict(np.load(OUT/f"{panel}_{split}.npz", allow_pickle=False))

if __name__ == "__main__":
    panels = sys.argv[1:] or PANELS
    for p in panels:
        for s in ("train", "validation", "private"): build(p, s)
