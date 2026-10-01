"""Digital-twin feasibility: is the generator an LWR/CTM on the released triangular FD?

Check 1 (FD): per observed eligible cell (train, observed layer), compare speed with the triangular-FD speed implied by
         the cell's own density k=q/v:  v_FD(k) = v_f (k<=k_c) else w*(k_jam-k)/k,  w = cap/(k_jam-k_c).
         Report median |v - v_FD| in free flow and in congestion, and the share of congested cells within 5 km/h.
         If the generator uses this FD, congested points sit on the branch up to measurement noise.
Check 2 (conservation between consecutive sensored links with no ramps and no latent links in between):
         |q_i - q_{i+1}| sd vs the per-link noise level -> are 5-min flows conserved up to noise?
Check 3 (capacity): distribution of max flow per link vs released capacity (x1.0 or x1.2 multiplier?) on train vs validation.
"""
import sys, numpy as np, pandas as pd
sys.path.insert(0, "tfb")
import cache, score_local as sl

panels = sys.argv[1:] or ["D12_I5_N", "D7_I405_N", "D7_I10_W"]
for panel in panels:
    z = cache.load(panel, "train"); links = list(z["links"]); p = sl.net_params(panel, links)
    vf = p.free_speed_kmh.to_numpy(); cap = p.capacity_vph.to_numpy(); kc = p.critical_density.to_numpy(); kj = p.k_jam.to_numpy()
    w = cap / np.maximum(kj - kc, 1e-6)
    el = z["elig"].astype(bool); v = np.where(el, z["speed"], np.nan); q = np.where(el, z["flow"], np.nan)
    k = q / np.maximum(v, 1.0)
    vFD = np.where(k <= kc, vf, w * np.maximum(kj - k, 0) / np.maximum(k, 1e-6))
    err = v - vFD; cong = k > kc
    print(f"== {panel}: v_f med {np.median(vf):.1f}, w med {np.median(w):.1f} km/h, k_jam/lane med {np.median(kj / p.lanes.to_numpy()):.0f}")
    print(f"  free-flow cells: median|v-vFD| {np.nanmedian(np.abs(err[~cong])):.2f}  sd {np.nanstd(err[~cong]):.2f}   n={np.isfinite(err[~cong]).sum()}")
    print(f"  congested cells: median|v-vFD| {np.nanmedian(np.abs(err[cong])):.2f}  sd {np.nanstd(err[cong]):.2f}  within5 {np.nanmean(np.abs(err[cong]) <= 5):.3f}  n={np.isfinite(err[cong]).sum()}")
    # congested: fit empirical v vs k to see the real congested branch
    kk = k[cong & np.isfinite(v)]; vv = v[cong & np.isfinite(v)]; ll = np.broadcast_to(np.arange(len(links)), k.shape)[cong & np.isfinite(v)]
    if len(kk) > 100:
        qq = kk * vv; lanes = p.lanes.to_numpy()[ll]
        A = np.c_[np.ones_like(kk), kk / lanes]; coef = np.linalg.lstsq(A, qq / lanes, rcond=None)[0]
        print(f"  empirical congested branch per lane: q = {coef[0]:.0f} + {coef[1]:.1f}*k  (w_emp={-coef[1]:.1f} km/h, k_jam_emp={-coef[0]/coef[1]:.0f}/lane)")
    # check 2: adjacent sensored links, both without ramps
    topo = pd.read_csv(cache.REL/"corridors"/panel/"network"/"lwr_mainline_topology.csv", dtype=str).drop_duplicates("link_id").set_index("link_id").reindex(links)
    noramp = topo.on_ramp_link_ids.isna().to_numpy() & topo.off_ramp_link_ids.isna().to_numpy()
    nxt = topo.outgoing_link_ids.fillna("").str.split(";").apply(lambda s: [x for x in s if x in set(links)])
    li = {l: i for i, l in enumerate(links)}; diffs = []
    for i, l in enumerate(links):
        if not noramp[i] or len(nxt.iloc[i]) != 1: continue
        j = li[nxt.iloc[i][0]]
        if not noramp[j]: continue
        diffs.append((q[:, :, i] - q[:, :, j]).ravel())
    if diffs:
        dd = np.concatenate(diffs); print(f"  adjacent no-ramp flow diff: sd {np.nanstd(dd):.1f} vph, mean {np.nanmean(dd):.1f} (pairs={len(diffs)})")
    # check 3: capacity use
    for split in ("train", "validation", "private"):
        zz = cache.load(panel, split); qq = np.where(zz["elig"].astype(bool), zz["mflow"], np.nan)
        r = np.nanquantile(qq, 0.999, axis=(0, 1)) / cap
        print(f"  {split}: q99.9/capacity median {np.nanmedian(r):.3f}  max {np.nanmax(r):.3f}")
