"""Where does an onset queue form? Non-learned legal rules on the 40 released TRAIN onset windows (nothing fitted, so the
windows are admissible as an evaluation; the frequency prior is computed leave-the-window-day-out).
Signals from the legal history [T-60, T) only, eligible cells forward-filled:
  r_last   last speed / v_cut (low = near the queue threshold)      r_min   hour minimum of speed / v_cut
  qcap     last flow / capacity (high = near capacity)              dr      r_last - r(T-30) (negative = slowing)
  r_down   r_last of the downstream link                            freq    train queue rate at T+30, weekday type, leave-day-out
Rules predict the top-k links at h=6 only (the sampler regularity); IoU over the 6-step horizon on eligible cells.
Also reports where the true h=6 links sit in each ranking, and the true queue size.
"""
import sys, numpy as np, pandas as pd
sys.path.insert(0, "tfb")
import cache
import t2_persist as tp
from scipy.ndimage import median_filter
from t2 import statics, daytype
REL = cache.REL; pd.set_option("display.width", 250)


def iou(s, y, k):
    u = ((s | y) & k).sum(); return 1.0 if u == 0 else (s & y & k).sum() / u


def downstream_index(panel, links):
    topo = pd.read_csv(REL/"corridors"/panel/"network"/"lwr_mainline_topology.csv", dtype=str).drop_duplicates("link_id").set_index("link_id")
    li = {l: i for i, l in enumerate(links)}; out = np.full(len(links), -1)
    for i, l in enumerate(links):
        if l in topo.index:
            nxt = [x.strip() for x in str(topo.loc[l, "outgoing_link_ids"]).split(";") if x.strip() in li]
            if nxt: out[i] = li[nxt[0]]
    return out


def main():
    rows, ranks = [], []
    for panel in tp.Q_PANELS:
        z = cache.load(panel, "train"); links = list(z["links"]); st = statics(panel, links); vcut = st["vcut"]; dates = list(z["dates"]); dows = daytype(z["dates"])
        fd = pd.read_csv(REL/"corridors"/panel/"network"/"fd_parameters.csv", dtype={"link_id": str}).groupby("link_id").capacity_vph.mean().reindex(links).to_numpy()
        dn = downstream_index(panel, links); sp, fl, el_all = z["speed"], z["flow"], z["elig"].astype(bool)
        qd = (np.nan_to_num(sp, nan=1e9) <= vcut[None, None, :]) & el_all; vd = el_all & np.isfinite(sp)
        wt_all = (np.asarray(dows) >= 5).astype(int)
        S = {w: qd[wt_all == w].sum(0).astype(float) for w in (0, 1)}; N = {w: vd[wt_all == w].sum(0).astype(float) for w in (0, 1)}
        wi = pd.read_csv(REL/"task2"/panel/"train"/"window_index.csv"); wi = wi[wi.condition == "queue_onset"]
        for w in wi.itertuples():
            d = dates.index(w.date); s0 = tp.slot_of(w.forecast_origin); el = el_all[d]; wt = wt_all[d]; h = slice(s0 - 12, s0)
            v = pd.DataFrame(np.where(el[h], sp[d, h], np.nan)).ffill().to_numpy(); q = pd.DataFrame(np.where(el[h], fl[d, h], np.nan)).ffill().to_numpy()
            r = v / vcut[None, :]; r_last = np.nan_to_num(r[-1], nan=9.0); r_min = np.nan_to_num(np.nanmin(np.where(np.isfinite(r), r, np.nan), 0), nan=9.0)
            qcap = np.nan_to_num(q[-1] / fd, nan=0.0); dr = np.nan_to_num(r[-1] - r[-7], nan=0.0)
            r_down = np.where(dn >= 0, r_last[np.maximum(dn, 0)], 9.0)
            freq = (S[wt][s0 + 6] - qd[d, s0 + 6]) / np.maximum(N[wt][s0 + 6] - vd[d, s0 + 6], 1)
            y = np.nan_to_num(sp[d, s0 + 1:s0 + 7], nan=1e9) <= vcut[None, :]
            ylat = tp.fill(median_filter(np.nan_to_num(sp[d], nan=1e9), size=(3, 1), mode="nearest")[s0 + 1:s0 + 7] <= vcut[None, :])
            k = el[s0 + 1:s0 + 7] & np.isfinite(sp[d, s0 + 1:s0 + 7])
            score = {"r_last": -r_last, "r_min": -r_min, "qcap": qcap, "dr": -dr, "r_down": -r_down, "freq": freq,
                     "phys_rank": -(pd.Series(r_last).rank().to_numpy() + pd.Series(-qcap).rank().to_numpy())}
            rec = dict(panel=panel, w=w.window_id, n_true_h6=int((y[5] & k[5]).sum()), n_true_any=int((y & k).sum()), first_h=int(np.argmax(y.any(1))) + 1 if y.any() else 0,
                       r_last_true=float(np.mean(r_last[y[5] & k[5]])) if (y[5] & k[5]).any() else np.nan, r_last_all=float(np.median(r_last[r_last < 9])))
            true6 = np.flatnonzero(y[5] & k[5])
            for nm, sc in score.items():
                order = np.argsort(-sc)
                pos = np.empty(len(sc)); pos[order] = np.arange(len(sc))
                if len(true6): ranks.append(dict(signal=nm, best_rank=int(pos[true6].min()) + 1, pct=float(pos[true6].min() / len(sc))))
                for kk in (1, 2, 3, 5):
                    s = np.zeros((6, len(sc)), bool); s[5, order[:kk]] = True
                    rec[f"{nm}@{kk}|noisy"] = iou(s, y, k); rec[f"{nm}@{kk}|latent"] = iou(s, ylat, k)
            rows.append(rec)
    r = pd.DataFrame(rows); rk = pd.DataFrame(ranks)
    print("true queue at h=6: median links", r.n_true_h6.median(), "| any-step cells", r.n_true_any.median(), "| first step", r.first_h.value_counts().to_dict())
    print("speed ratio at T-5: true h=6 links", round(r.r_last_true.mean(), 3), "vs all links (median)", round(r.r_last_all.mean(), 3))
    cols = sorted({c.split("|")[0] for c in r.columns if "|" in c}); out = []
    for c in cols:
        out.append(dict(rule=c, noisy=r[f"{c}|noisy"].mean(), se_noisy=r[f"{c}|noisy"].std() / np.sqrt(len(r)), latent=r[f"{c}|latent"].mean(),
                        d_vs_freq_latent=(r[f"{c}|latent"] - r[f"freq@{c.split('@')[1]}|latent"]).mean()))
    o = pd.DataFrame(out).sort_values("latent", ascending=False).round(4); print(o.to_string(index=False))
    print("best rank of a true h=6 link in each ranking (1 = top):"); print(rk.groupby("signal").best_rank.describe()[["mean", "25%", "50%", "75%"]].round(1).to_string())
    r.to_csv(cache.Path.home()/"tfb/work/audit/onset_rules.csv", index=False) if hasattr(cache, "Path") else None


if __name__ == "__main__":
    main()
