"""CRITIQUE.md section 0.5: validate tfb/score_local.T3 against the official repo/src/task3/score_task3.py, and measure the
physics ceiling with the right controls (the earlier 'ideal = truth + fresh noise' experiment was the wrong perturbation).

make_flux(): writes a --boundary-flux file projected onto the observations (inflow = dN_obs/dt - on + off, outflow = 0), which is
what the load_boundary_flux docstring says the organizer flux is ("a participant who reproduces what was published satisfies the
conservation identity exactly"). With that file the official scorer and T3 must agree on any field.

Fields written at the Task 1 train targets of the Dec-Feb train months (official hash mask, R1/R2/R3 by day):
  exact        the observations themselves (official S_LWR must be 1)
  exact+noise  observations + a fresh draw at the estimated per-link noise sd (the old 'ideal' control)
  denoise3     3-slot centred mean of the observations of the same link (an oracle-flavoured conditional-mean proxy)
  tinterp      time interpolation of the released masked layer (a real predictor)
  *_fdk        the same field moved onto the released triangle at constant density k=q/v (N and S_LWR unchanged)
Noise sd per link: DER_SNR-style robust estimator 1.4826*MAD(second difference)/sqrt(6) on the Dec-Feb observations.
Usage: t3_validate.py PANEL [PANEL ...]
"""
import sys, os, subprocess, numpy as np, pandas as pd
from pathlib import Path
sys.path.insert(0, "tfb"); sys.path.insert(0, "repo/src"); sys.path.insert(0, "repo/src/task3")
import cache
from score_local import T3, net_params
import score_task3 as st3

REL = cache.REL; STAGE = Path.home()/"tfb/stage_t3"; OUT = Path.home()/"tfb/work/t3val"; OUT.mkdir(parents=True, exist_ok=True)
MONTHS = ("2030-12", "2031-01", "2031-02"); rng = np.random.default_rng(0)


def subset(z, m):
    D = len(z["dates"]); return {k: (v[m] if isinstance(v, np.ndarray) and v.ndim >= 1 and v.shape[0] == D and k not in ("links", "ramp_ids", "ramp_type") else v) for k, v in z.items()}


def stage(panel):
    src = REL/"corridors"/panel; dst = STAGE/"corridors"/panel
    for sub in ("mainline_states", "ramp_states"):
        for mo in MONTHS:
            d = dst/"train"/sub/f"year_month={mo}"; d.mkdir(parents=True, exist_ok=True)
            for f in sorted((src/"train"/sub/f"year_month={mo}").glob("*.parquet")):
                if not (d/f.name).exists(): os.symlink(f, d/f.name)
    if not (dst/"network").exists(): os.symlink(src/"network", dst/"network")
    t = pd.read_csv(REL/"task1"/panel/"train"/"sample_submission_state.csv", dtype=str)
    t = t[t.timestamp.str[:7].isin(MONTHS)]; (STAGE/"task1"/panel/"train").mkdir(parents=True, exist_ok=True)
    t.to_csv(STAGE/"task1"/panel/"train"/"sample_submission_state.csv", index=False); return t


def noise_sd(x):
    """[D,288,L] -> per-link robust noise sd from second differences within a day."""
    d2 = x[:, 2:] - 2 * x[:, 1:-1] + x[:, :-2]
    med = np.nanmedian(d2, axis=(0, 1)); mad = np.nanmedian(np.abs(d2 - med), axis=(0, 1)); return 1.4826 * mad / np.sqrt(6)


def tinterp(mv, ok):
    out = np.empty_like(mv)
    for d in range(mv.shape[0]):
        out[d] = pd.DataFrame(np.where(ok[d], mv[d], np.nan)).interpolate(limit_direction="both").to_numpy()
    return out


def fd_k(v, q, p):
    """Move (v, q) onto the released triangle at constant k = q/max(v,1) (so N = k*L is unchanged)."""
    lanes = p.lanes.to_numpy(); vf = p.free_speed_kmh.to_numpy(); cap = p.capacity_vph.to_numpy(); kc = p.critical_density.to_numpy(); kj = p.k_jam.to_numpy()
    k = q / np.maximum(v, 1.0); kl = k / lanes
    qfd = np.where(kl <= kc / lanes, vf * kl, (cap / lanes) / np.maximum((kj - kc) / lanes, 1e-9) * np.maximum(kj / lanes - kl, 0)) * lanes
    ok = (k > 1e-6) & np.isfinite(qfd) & (qfd > 0) & (qfd / np.maximum(k, 1e-6) >= 1.0)
    return np.where(ok, qfd / np.maximum(k, 1e-6), v), np.where(ok, qfd, q)


def make_flux(panel, truth_sub):
    panel_dir = STAGE/"corridors"/panel; params = st3.network_parameters(panel_dir).set_index("link_id")
    order = st3.link_order(panel_dir, "train"); ramps = st3.load_ramp_flows(panel_dir, "train"); parts = []
    for r in st3.REGIMES:
        f, miss = st3.derive_physics(panel, STAGE, "train", truth_sub, r, order, params, ramps, None); assert miss == 0
        f = f.sort_values(["link_id", "timestamp"]).copy(); g = f.groupby("link_id")
        dN = g.accumulation_N.shift(-1) - f.accumulation_N; dt = (g.timestamp.shift(-1) - f.timestamp).dt.total_seconds() / 3600.0
        v = (dt > 0) & (dt <= 5.1 / 60.0)
        f["inflow_vph"] = np.where(v, dN / dt.where(v, 1.0) - f.on_ramp_flow_vph + f.off_ramp_flow_vph, f.flow_vph)
        f["outflow_vph"] = np.where(v, 0.0, f.flow_vph); parts.append(f[["timestamp", "link_id", "inflow_vph", "outflow_vph"]])
    fl = pd.concat(parts, ignore_index=True); path = OUT/f"flux_{panel}.parquet"; fl.to_parquet(path, index=False); return path


def official(panel, sub_csv, flux, tag):
    out = OUT/f"off_{panel}_{tag}.csv"
    r = subprocess.run([sys.executable, "repo/src/task3/score_task3.py", "--state-submission", str(sub_csv), "--split", "train", "--boundary-flux", str(flux),
                        "--release-root", str(STAGE), "--panel", panel, "--output", str(out)], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-2000:]
    d = pd.read_csv(out); d = d[d.level == "panel_regime"]; return d.set_index("regime")[["S_FD", "S_LWR", "S_physics"]]


def run(panel):
    z0 = cache.load(panel, "train"); m = np.array([str(d)[:7] in MONTHS for d in z0["dates"]]); z = subset(z0, m)
    tpl = stage(panel); links = list(z["links"]); li = {l: i for i, l in enumerate(links)}; di = {d: i for i, d in enumerate(z["dates"])}
    p = net_params(panel, links); lanes = p.lanes.to_numpy()
    V, Q = z["speed"], z["flow"]; el = z["elig"].astype(bool) & np.isfinite(V) & np.isfinite(Q)
    sv, sq = noise_sd(np.where(el, V, np.nan)), noise_sd(np.where(el, Q / lanes, np.nan))
    print(f"{panel}: noise sd speed median {np.nanmedian(sv):.2f} km/h [p10 {np.nanpercentile(sv, 10):.2f}, p90 {np.nanpercentile(sv, 90):.2f}]; "
          f"flow median {np.nanmedian(sq):.1f} vph/lane [p10 {np.nanpercentile(sq, 10):.1f}, p90 {np.nanpercentile(sq, 90):.1f}]", flush=True)
    okm = z["elig"].astype(bool) & np.isfinite(z["mspeed"]) & np.isfinite(z["mflow"]) & ~z["target"].astype(bool)
    Vn = np.where(el, V, np.nan); Qn = np.where(el, Q, np.nan)
    with np.errstate(all="ignore"):
        den = lambda x: np.nanmean(np.stack([np.roll(x, 1, 1), x, np.roll(x, -1, 1)]), 0)
    F = {"exact": (V, Q),
         "exact+noise": (V + rng.normal(0, 1, V.shape) * sv, Q + rng.normal(0, 1, Q.shape) * sq * lanes),
         "denoise3": (np.where(np.isfinite(den(Vn)), den(Vn), V), np.where(np.isfinite(den(Qn)), den(Qn), Q)),
         "tinterp": (tinterp(z["mspeed"], okm), tinterp(z["mflow"], okm))}
    for nm in ("exact", "exact+noise", "denoise3", "tinterp"): F[nm + "_fdk"] = fd_k(*F[nm], p)
    t3 = T3(panel, z); tg = z["target"].astype(bool) & el; rows = []
    d_ = tpl.timestamp.str[:10].map(di).to_numpy(); s_ = (tpl.timestamp.str[11:13].astype(int) * 12 + tpl.timestamp.str[14:16].astype(int) // 5).to_numpy(); l_ = tpl.link_id.map(li).to_numpy()
    flux = None
    for nm, (pv, pq) in F.items():
        pv = np.clip(pv, 1.0, 160.0); pq = np.maximum(pq, 0.0)
        loc = t3.score(z, pv, pq)
        sub = tpl[["panel", "timestamp", "station_id", "link_id", "mask_regime"]].copy(); sub["speed_kmh"] = pv[d_, s_, l_]; sub["flow_vph"] = pq[d_, s_, l_]
        sub = sub.fillna({"speed_kmh": 60.0, "flow_vph": 1000.0}); path = OUT/f"sub_{panel}_{nm}.csv"; sub.to_csv(path, index=False)
        if flux is None:
            assert nm == "exact"; flux = make_flux(panel, sub)
        off = official(panel, path, flux, nm)
        rm_v = np.sqrt(np.mean((pv - V)[tg] ** 2)); rm_q = np.sqrt(np.mean(((pq - Q) / lanes)[tg] ** 2))
        for r in ("R1", "R2", "R3"):
            rows.append(dict(panel=panel, field=nm, regime=r, loc_FD=loc[r]["S_FD"], off_FD=off.loc[r, "S_FD"], loc_LWR=loc[r]["S_LWR"], off_LWR=off.loc[r, "S_LWR"],
                             loc_phys=loc[r]["S_phys"], off_phys=off.loc[r, "S_physics"], rmse_v=rm_v, rmse_q_lane=rm_q))
        print(nm, "local", round(loc["S_physics"], 4), "official", round(off.S_physics.mean(), 4), "rmse v", round(rm_v, 2), "q/lane", round(rm_q, 1), flush=True)
    return pd.DataFrame(rows)


if __name__ == "__main__":
    r = pd.concat([run(p) for p in sys.argv[1:]], ignore_index=True); r.to_csv(OUT/"t3_validate.csv", index=False)
    g = r.groupby("field")[["loc_FD", "off_FD", "loc_LWR", "off_LWR", "loc_phys", "off_phys", "rmse_v", "rmse_q_lane"]].mean().round(4)
    print(g.to_string()); print("max |local - official| S_physics:", (r.loc_phys - r.off_phys).abs().max().round(5))
