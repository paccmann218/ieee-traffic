"""Fast local scorers on dense caches (train split, where truth is released).

score_t1: official Task 1 rule (per regime RMSE speed, RMSE per-lane flow, 0.54/0.46 blend, mean over regimes).
score_t3: official Task 3 rule with the organizer boundary flux reconstructed from the truth field.
    The legacy builder's project_onto_conservation shows the organizer flux makes the observed (truth) field
    conserve exactly, so rhs == dN_true on every scored transition. Hence
        S_LWR = 1 - sum|dN_sub - dN_true| / sum|dN_true|
    over transitions that are consecutive (dt = 5 min, both cells eligible), on observable links, with valid ramps.
    S_FD is computed on all eligible cells of the regime (targets carry our values), per lane, triangular FD.
Validate once against repo/src/task3/score_task3.py with a written --boundary-flux file (see make_flux()).
"""
from __future__ import annotations
import numpy as np, pandas as pd
from pathlib import Path
import cache

REL = cache.REL
REGIMES = ("R1", "R2", "R3")


def net_params(panel, links):
    """Mirror score_task3.network_parameters on the fd_parameters authority, in cache link order."""
    n = pd.read_csv(REL/"corridors"/panel/"network"/"links.csv", dtype={"link_id": str}).drop_duplicates("link_id").set_index("link_id")
    f = pd.read_csv(REL/"corridors"/panel/"network"/"fd_parameters.csv", dtype={"link_id": str}).groupby("link_id")[["free_speed_kmh", "capacity_vph", "lanes", "k_jam"]].mean()
    out = pd.DataFrame(index=pd.Index(links, name="link_id"))
    out["length_km"] = pd.to_numeric(n.length_km.reindex(links), errors="coerce").fillna(1.0)
    for c in ("free_speed_kmh", "capacity_vph", "lanes"):
        out[c] = f[c].reindex(links).fillna(pd.to_numeric(n[c].reindex(links), errors="coerce"))
    out["lanes"] = out.lanes.fillna(1.0).clip(lower=1.0)
    out["critical_density"] = out.capacity_vph / out.free_speed_kmh.clip(lower=1)
    out["k_jam"] = f.k_jam.reindex(links).fillna(100.0 * out.lanes)
    out["k_jam"] = np.maximum(out.k_jam, out.critical_density * 1.05)
    return out


def lanes_t1(panel, links):
    """score_task1 lane rule: fd_parameters lanes by link (links.csv fallback), valid 1..16, default 1."""
    lanes = {}
    n = pd.read_csv(REL/"corridors"/panel/"network"/"links.csv", dtype={"link_id": str})
    f = pd.read_csv(REL/"corridors"/panel/"network"/"fd_parameters.csv", dtype={"link_id": str})
    for fr in (n, f):
        v = pd.to_numeric(fr.lanes, errors="coerce"); ok = v.notna() & (v >= 1) & (v <= 16)
        lanes.update(zip(fr.link_id[ok].astype(str), v[ok].astype(float)))
    return np.array([lanes.get(l, 1.0) for l in links])


def score_t1(z, pv, pq, lanes):
    """z: train cache; pv/pq: predictions [D,288,L] (only target cells used)."""
    tgt = z["target"].astype(bool) & np.isfinite(z["speed"]) & np.isfinite(z["flow"])
    out = {}
    for r in REGIMES:
        m = tgt & (z["regime"] == r)[:, None, None]
        if not m.any(): out[r] = dict(S=0.0); continue
        ds = np.nan_to_num(pv[m], nan=0.0) - z["speed"][m]
        dq = (np.nan_to_num(pq[m], nan=0.0) - z["flow"][m]) / np.broadcast_to(lanes, m.shape)[m]
        rv, rq = np.sqrt(np.mean(ds**2)), np.sqrt(np.mean(dq**2))
        sv, sq = max(0, 1 - rv / 25), max(0, 1 - rq / 600)
        out[r] = dict(rmse_v=rv, rmse_q=rq, S=0.54 * sv + 0.46 * sq, n=int(m.sum()))
    out["S_state"] = float(np.mean([out[r]["S"] for r in REGIMES]))
    return out


class T3:
    """Precomputed topology / ramp validity for a panel+split; call .score(z, pv, pq)."""

    def __init__(self, panel, z):
        self.panel = panel; links = list(z["links"]); self.links = links; L = len(links)
        self.p = net_params(panel, links)
        topo = pd.read_csv(REL/"corridors"/panel/"network"/"lwr_mainline_topology.csv", dtype=str).drop_duplicates("link_id").set_index("link_id").reindex(links)
        on_att = topo.on_ramp_link_ids.fillna("").str.len().gt(0).to_numpy()
        off_att = topo.off_ramp_link_ids.fillna("").str.len().gt(0).to_numpy()
        topo_ids = set(topo.index)
        # observable: all mainline neighbours in topo must be present in the submission field (all cached links are)
        obs = []
        for l in links:
            nb = set()
            for c in ("incoming_link_ids", "outgoing_link_ids"):
                nb |= {v.strip() for v in str(topo.loc[l, c]).split(";") if v.strip() and v.strip() != "nan"}
            obs.append((nb & topo_ids).issubset(set(links)))
        self.observable = np.array(obs)
        # ramp validity per (d, t, link): any valid attached ramp of that type (evaluator aggregates max over ramps)
        amap = pd.read_csv(REL/"corridors"/panel/"network"/"ramp_attachment_map.csv", dtype=str)
        rid = list(z["ramp_ids"]); ri = {x: i for i, x in enumerate(rid)}; li = {x: i for i, x in enumerate(links)}
        D = z["elig"].shape[0]
        on_valid = np.zeros((D, 288, L), bool); off_valid = np.zeros((D, 288, L), bool)
        rt = z["ramp_type"]; relig = z["relig"].astype(bool) & np.isfinite(z["rflow"])
        for row in amap.itertuples():
            if row.ramp_link_id not in ri or row.nearest_mainline_link_id not in li: continue
            r, l = ri[row.ramp_link_id], li[row.nearest_mainline_link_id]
            if str(rt[r]).upper() in ("OR", "ON"): on_valid[:, :, l] |= relig[:, :, r]
            elif str(rt[r]).upper() in ("FR", "OFF"): off_valid[:, :, l] |= relig[:, :, r]
        self.ramp_ok = (on_valid | ~on_att[None, None, :]) & (off_valid | ~off_att[None, None, :])

    def field(self, z, pv, pq, regime):
        dm = z["regime"] == regime
        el = z["elig"][dm].astype(bool) & np.isfinite(z["speed"][dm]) & np.isfinite(z["flow"][dm])
        tg = z["target"][dm].astype(bool) & el
        V = np.where(tg, pv[dm], z["speed"][dm]); Q = np.where(tg, pq[dm], z["flow"][dm])
        V = np.where(el, V, np.nan); Q = np.where(el, Q, np.nan)
        return dm, el, tg, V, Q

    def score(self, z, pv, pq):
        p = self.p; Lk = p.length_km.to_numpy(); lanes = p.lanes.to_numpy()
        vf = p.free_speed_kmh.to_numpy(); cap = p.capacity_vph.to_numpy(); kc = p.critical_density.to_numpy(); kj = p.k_jam.to_numpy()
        dates = z["dates"]; out = {}
        for r in REGIMES:
            dm, el, tg, V, Q = self.field(z, pv, pq, r)
            Vt = np.where(el, z["speed"][dm], np.nan); Qt = np.where(el, z["flow"][dm], np.nan)
            # S_FD on all eligible cells
            k = Q / np.maximum(V, 1.0); ql = Q / lanes; kl = k / lanes
            qfd = np.where(kl <= kc / lanes, vf * kl, (cap / lanes) / np.maximum((kj - kc) / lanes, 1e-9) * np.maximum(kj / lanes - kl, 0))
            ok = el & np.isfinite(ql) & np.isfinite(qfd)
            s_fd = max(0.0, 1 - np.abs(ql - qfd)[ok].sum() / (np.abs(ql[ok]).sum() + 1e-9))
            # conservation transitions: flatten days of this regime in date order, link across midnight when dates consecutive
            N = k * Lk; Nt = (Qt / np.maximum(Vt, 1.0)) * Lk
            d_idx = np.flatnonzero(dm)
            cons = np.r_[(np.diff(pd.to_datetime(dates[d_idx]).values).astype("timedelta64[D]").astype(int) == 1), False]
            Nf = N.reshape(-1, N.shape[-1]); Ntf = Nt.reshape(-1, N.shape[-1]); elf = el.reshape(-1, N.shape[-1])
            rok = self.ramp_ok[dm].reshape(-1, N.shape[-1])
            nxt_ok = np.ones(Nf.shape[0], bool); nxt_ok[-1] = False
            day_end = np.arange(Nf.shape[0]) % 288 == 287
            nxt_ok[day_end] = np.repeat(cons, 1)[: day_end.sum()]
            valid = elf[:-1] & elf[1:] & nxt_ok[:-1, None] & rok[:-1] & self.observable[None, :]
            dN = Nf[1:] - Nf[:-1]; dNt = Ntf[1:] - Ntf[:-1]
            num = np.abs(dN - dNt)[valid].sum(); den = np.abs(dNt)[valid].sum()
            s_lwr = max(0.0, 1 - min(1.0, num / (den + 1e-9)))
            out[r] = dict(S_FD=s_fd, S_LWR=s_lwr, S_phys=(s_fd + 2 * s_lwr) / 3)
        out["S_physics"] = float(np.mean([out[r]["S_phys"] for r in REGIMES]))
        return out
