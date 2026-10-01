"""Digital twin: Cell Transmission Model (Godunov LWR) on the released triangular FD, for Task 2 forecasting.

Rules-safe: uses only the published history T-60..T-5 (slots s0-12..s0-1) of the window's own day, ramp flows at the
same slots, and static network data. Nothing is learned from train, so its score on train windows is an honest
estimate for an independently generated scenario.

Model per corridor (links in topology order = travel direction):
  k_i density (veh/km, all lanes), v_f,i, cap_i (vph), k_jam,i from fd_parameters; w_i = cap_i/(k_jam_i - k_c,i).
  Sending  S_i = min(v_f,i k_i, cap_i)          Receiving R_i = min(cap_i, w_i (k_jam,i - k_i))
  Interface flow between i and i+1: F = min(S_i (1-beta_i), R_{i+1} - ... )  (off-ramp split beta_i at the end of i,
  on-ramp inflow r_{i+1} enters link i+1 with priority to mainline proportionally).
  Upstream boundary: inflow = min(demand, R_0). Downstream boundary: free outflow S_last.
Calibration from history (last 3 slots, forward-filled):
  initial k = q/v at slot s0-1; demand = upstream-link flow; on-ramp flows; beta_i = off-ramp flow / mainline flow;
  active bottleneck: if link i is queued and link i+1 is not, capacity at the i->i+1 interface = observed flow at i+1
  (+ its off-ramp), i.e. the measured discharge rate (captures incidents / capacity drops without memorising locations).
"""
from __future__ import annotations
import sys, numpy as np, pandas as pd
sys.path.insert(0, "tfb")
import cache, score_local as sl

REL = cache.REL


class Corridor:
    def __init__(self, panel, z):
        self.panel = panel; links = list(z["links"]); self.links = links; L = len(links)
        p = sl.net_params(panel, links)
        topo = pd.read_csv(REL/"corridors"/panel/"network"/"lwr_mainline_topology.csv", dtype=str).drop_duplicates("link_id").set_index("link_id").reindex(links)
        self.len = pd.to_numeric(topo.length_km, errors="coerce").fillna(p.length_km).clip(lower=0.05).to_numpy()
        self.vf = p.free_speed_kmh.to_numpy(); self.cap = p.capacity_vph.to_numpy(); self.kj = p.k_jam.to_numpy()
        self.kc = self.cap / self.vf; self.w = self.cap / np.maximum(self.kj - self.kc, 1e-6)
        self.vcut = 0.6 * self.vf
        amap = pd.read_csv(REL/"corridors"/panel/"network"/"ramp_attachment_map.csv", dtype=str)
        rid = list(z["ramp_ids"]); ri = {x: i for i, x in enumerate(rid)}; li = {x: i for i, x in enumerate(links)}
        self.on = [[] for _ in range(L)]; self.off = [[] for _ in range(L)]
        for row in amap.itertuples():
            if row.ramp_link_id in ri and row.nearest_mainline_link_id in li:
                r = ri[row.ramp_link_id]; l = li[row.nearest_mainline_link_id]
                (self.on if str(z["ramp_type"][r]).upper() in ("OR", "ON") else self.off)[l].append(r)
        self.dt_h = 0.9 * np.min(self.len / self.vf)          # CFL
        self.nsub = int(np.ceil((5 / 60) / self.dt_h)); self.dt_h = (5 / 60) / self.nsub

    def ramp_sum(self, rf, idx_lists):
        out = np.zeros(len(idx_lists))
        for i, lst in enumerate(idx_lists):
            if lst: out[i] = np.nansum(rf[lst]) if np.isfinite(rf[lst]).any() else np.nan
        return out

    def calibrate(self, v_hist, q_hist, rf_hist):
        """v_hist, q_hist: [12, L] visible history (NaN = missing); rf_hist: [12, R] eligible ramp flows."""
        def last_ffill(a, n=3):
            b = pd.DataFrame(a).ffill().to_numpy()
            return np.nanmean(b[-n:], axis=0)
        v = last_ffill(v_hist, 1); q = last_ffill(q_hist, 1)
        k = q / np.maximum(v, 1.0)
        k = pd.Series(k).interpolate(limit_direction="both").to_numpy()      # spatial fill of links never observed
        k = np.where(np.isfinite(k), k, self.kc * 0.5)
        qm = pd.Series(last_ffill(q_hist, 3)).interpolate(limit_direction="both").to_numpy()
        vm = pd.Series(last_ffill(v_hist, 3)).interpolate(limit_direction="both").to_numpy()
        rf = pd.DataFrame(rf_hist).ffill().to_numpy()[-3:].mean(0) if rf_hist.shape[1] else np.zeros(0)
        on = np.nan_to_num(self.ramp_sum(rf, self.on)); off = self.ramp_sum(rf, self.off)
        beta = np.clip(np.nan_to_num(off / np.maximum(qm, 1.0)), 0, 0.5)
        demand = qm[0]
        # active bottlenecks: queued link followed by a free-flow link -> interface capacity = observed discharge
        queued = vm <= self.vcut
        icap = np.minimum(self.cap[:-1], self.cap[1:]).astype(float)           # interface capacity i -> i+1
        for i in range(len(k) - 1):
            if queued[i] and not queued[i + 1] and np.isfinite(qm[i + 1]):
                icap[i] = max(min(icap[i], qm[i + 1] - on[i + 1] + 0.0), 0.3 * self.cap[i])
        return dict(k=k, on=on, beta=beta, demand=demand, icap=icap)

    def run(self, st, nslots=6, trend=None):
        trend = np.ones(nslots) if trend is None else trend
        k = st["k"].copy(); L = len(k); vf, cap, kj, w = self.vf, self.cap, self.kj, self.w
        speeds = np.zeros((nslots, L))
        for s in range(nslots):
            vacc = np.zeros(L)
            for _ in range(self.nsub):
                S = np.minimum(vf * k, cap); R = np.minimum(cap, w * np.maximum(kj - k, 0))
                send_main = S[:-1] * (1 - st["beta"][:-1])
                # on-ramp of link i+1 shares the receiving capacity with the mainline (mainline first, ramp gets rest)
                recv = np.minimum(R[1:], st["icap"])
                Fm = np.minimum(send_main, recv)
                on = st["on"] * trend[s]; dem = st["demand"] * trend[s]
                Fr = np.minimum(on[1:], np.maximum(R[1:] - Fm, 0))
                Fin0 = min(dem, R[0]) + min(on[0], max(R[0] - dem, 0))
                inflow = np.r_[Fin0, Fm + Fr]
                outflow = np.r_[Fm / np.maximum(1 - st["beta"][:-1], 1e-6), S[-1]]      # includes off-ramp share of link i
                k = np.clip(k + self.dt_h / self.len * (inflow - outflow), 0, kj)
                vacc += np.where(k <= self.kc, vf, w * np.maximum(kj - k, 0) / np.maximum(k, 1e-6))
            speeds[s] = vacc / self.nsub
        return speeds


def slot_of(ts): return int(ts[11:13]) * 12 + int(ts[14:16]) // 5


def fill_holes(q):
    """Latent queues are contiguous: mark a link queued if both neighbours are queued."""
    q = q.copy(); q[..., 1:-1] |= q[..., :-2] & q[..., 2:]; return q


def forecast(corr, z, d, s0, prof_q=None):
    el = z["elig"][d].astype(bool)
    sp = z["speed"][d] if "speed" in z else z["mspeed"][d]; fl = z["flow"][d] if "flow" in z else z["mflow"][d]
    v = np.where(el, sp, np.nan)[s0 - 12:s0]; q = np.where(el, fl, np.nan)[s0 - 12:s0]
    rf = np.where(z["relig"][d].astype(bool), z["rflow"][d], np.nan)[s0 - 12:s0]
    st = corr.calibrate(v, q, rf); trend = None
    if prof_q is not None:
        base = np.nanmean(prof_q[s0 - 3:s0].sum(1)); trend = np.clip(prof_q[s0 + 1:s0 + 7].sum(1) / max(base, 1.0), 0.5, 1.5)
    vs = corr.run(st, trend=trend)
    return fill_holes(vs <= corr.vcut[None, :]), vs


def eval_train_windows():
    import t1_base as tb
    from scipy.ndimage import median_filter
    rows = []
    for panel in cache.PANELS[:8]:
        z = cache.load(panel, "train"); corr = Corridor(panel, z); dates = list(z["dates"]); PQ = tb.profile(z, ch="flow")
        wi = pd.read_csv(REL/"task2"/panel/"train"/"window_index.csv")
        for w in wi.itertuples():
            d = dates.index(w.date); s0 = slot_of(w.forecast_origin)
            pred0, _ = forecast(corr, z, d, s0); pred, _ = forecast(corr, z, d, s0, PQ[d])
            y = np.nan_to_num(z["speed"][d, s0 + 1:s0 + 7], nan=1e9) <= corr.vcut[None, :]
            sm = median_filter(np.nan_to_num(z["speed"][d], nan=1e9), size=(3, 1), mode="nearest")
            ylat = fill_holes(sm[s0 + 1:s0 + 7] <= corr.vcut[None, :])
            k = z["elig"][d, s0 + 1:s0 + 7].astype(bool) & np.isfinite(z["speed"][d, s0 + 1:s0 + 7])
            pers = np.repeat((np.nan_to_num(pd.DataFrame(np.where(z["elig"][d].astype(bool), z["speed"][d], np.nan)[s0 - 12:s0]).ffill().to_numpy()[-1], nan=1e9) <= corr.vcut)[None], 6, 0)
            r = dict(panel=panel, cond=w.condition[6:], nsub=corr.nsub)
            for lab_name, yy in (("", y), ("_lat", ylat)):
                for nm, s in (("ctm", pred0), ("ctm_trend", pred), ("persTm5", pers), ("persTm5_fill", fill_holes(pers))):
                    u = ((s | yy) & k).sum(); r[nm + lab_name] = 1.0 if u == 0 else (s & yy & k).sum() / u
            rows.append(r)
        print(panel, "nsub", corr.nsub, flush=True)
    r = pd.DataFrame(rows); cols = [c for c in r.columns if c not in ("panel", "cond", "nsub")]
    print(r.groupby("cond")[cols].mean().round(4).T)
    r.to_csv(cache.Path.home()/"tfb/work/twin_train_windows.csv", index=False)


if __name__ == "__main__":
    eval_train_windows()
