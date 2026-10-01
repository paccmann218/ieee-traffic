"""Onset location priors (non-learned), evaluated on the 40 released TRAIN onset windows with every prior computed
LEAVE-ONE-MONTH-OUT (all June 2030 days excluded; the 40 windows sit on 9 June dates, so the effective n is < 40).
  marginal   P(queued at T+30 | link, weekday type)                              (U3's rule)
  tiebreak   marginal, ties broken by the +-30 min smoothed marginal, then the all-day marginal (U3 defect fix: some
             validation windows have < 3 links with a positive point frequency)
  onsetcond  P(link queued at s | the corridor had no eligible queued cell in the 12 slots of a would-be onset history
             ending 6 slots before s), weekday type: the frequency of being part of a NEW queue at that slot
  qdown      last legal flow of the link / capacity of its downstream link (queues form upstream of a bottleneck)
  qup        last legal flow of the upstream link / own capacity
All rules pick top-3 at h=6. Also exported: rank_keys() used by build_next.onset_prior_rows.
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


def neighbours(panel, links):
    topo = pd.read_csv(REL/"corridors"/panel/"network"/"lwr_mainline_topology.csv", dtype=str).drop_duplicates("link_id").set_index("link_id")
    li = {l: i for i, l in enumerate(links)}; up = np.full(len(links), -1); dn = np.full(len(links), -1)
    for i, l in enumerate(links):
        if l not in topo.index: continue
        a = [x.strip() for x in str(topo.loc[l, "incoming_link_ids"]).split(";") if x.strip() in li]
        b = [x.strip() for x in str(topo.loc[l, "outgoing_link_ids"]).split(";") if x.strip() in li]
        if a: up[i] = li[a[0]]
        if b: dn[i] = li[b[0]]
    return up, dn


def tables(z, vcut, day_mask):
    """Queue indicators and marginal / onset-conditional counts over the days in day_mask."""
    sp = z["speed"]; el = z["elig"].astype(bool)
    qd = (np.nan_to_num(sp, nan=1e9) <= vcut) & el; vd = el & np.isfinite(sp)
    anyq = qd.any(2)                                            # [D, 288] corridor has a queued eligible cell
    # quiet[d, s] = no queued cell in slots s-18 .. s-7 (history of a window whose origin is s-6)
    cs = np.concatenate([np.zeros((anyq.shape[0], 1)), np.cumsum(anyq, 1)], 1)
    quiet = np.zeros_like(anyq)
    for s in range(18, 288):
        quiet[:, s] = (cs[:, s - 6] - cs[:, s - 18]) == 0
    return qd, vd, quiet


def rank_keys(qd, vd, quiet, m, s6):
    """Lexicographic keys (primary first) for the marginal prior with the tie-break chain; m = day mask."""
    f = qd[m, s6].sum(0) / np.maximum(vd[m, s6].sum(0), 1)
    lo, hi = max(s6 - 6, 0), min(s6 + 7, 288)
    fs = qd[m, lo:hi].sum((0, 1)) / np.maximum(vd[m, lo:hi].sum((0, 1)), 1)
    fd = qd[m].sum((0, 1)) / np.maximum(vd[m].sum((0, 1)), 1)
    return f, fs, fd


def order_by(keys):
    return np.lexsort(tuple(-k for k in keys[::-1]))


def main():
    rows = []
    for panel in tp.Q_PANELS:
        z = cache.load(panel, "train"); links = list(z["links"]); L = len(links); vcut = statics(panel, links)["vcut"]; dates = list(z["dates"])
        wt = (np.asarray(daytype(z["dates"])) >= 5).astype(int); mon = np.array([str(d)[:7] for d in dates])
        cap = pd.read_csv(REL/"corridors"/panel/"network"/"fd_parameters.csv", dtype={"link_id": str}).groupby("link_id").capacity_vph.mean().reindex(links).to_numpy()
        up, dn = neighbours(panel, links); qd, vd, quiet = tables(z, vcut, None); sp, fl, el = z["speed"], z["flow"], z["elig"].astype(bool)
        wi = pd.read_csv(REL/"task2"/panel/"train"/"window_index.csv"); wi = wi[wi.condition == "queue_onset"]
        for w in wi.itertuples():
            d = dates.index(w.date); s0 = tp.slot_of(w.forecast_origin); s6 = s0 + 6; m = (mon != mon[d]) & (wt == wt[d])
            y = np.nan_to_num(sp[d, s0 + 1:s0 + 7], nan=1e9) <= vcut; k = el[d, s0 + 1:s0 + 7] & np.isfinite(sp[d, s0 + 1:s0 + 7])
            ylat = tp.fill(median_filter(np.nan_to_num(sp[d], nan=1e9), size=(3, 1), mode="nearest")[s0 + 1:s0 + 7] <= vcut)
            f, fs, fd = rank_keys(qd, vd, quiet, m, s6)
            mq = m & quiet[:, s6]; fo = qd[mq, s6].sum(0) / max(mq.sum(), 1)
            h = slice(s0 - 12, s0); q = pd.DataFrame(np.where(el[d, h], fl[d, h], np.nan)).ffill().to_numpy()[-1]; q = np.nan_to_num(q, nan=0.0)
            qdown = np.where(dn >= 0, q / cap[np.maximum(dn, 0)], 0.0); qup = np.where(up >= 0, q[np.maximum(up, 0)] / cap, 0.0)
            R = {"marginal": np.argsort(-f), "tiebreak": order_by([f, fs, fd]), "onsetcond": order_by([fo, f, fs, fd]),
                 "qdown": np.argsort(-qdown), "qup": np.argsort(-qup)}
            rec = dict(panel=panel, date=w.date, n_at_risk=int(mq.sum()))
            for nm, o in R.items():
                s = np.zeros((6, L), bool); s[5, o[:3]] = True; rec[nm + "|noisy"] = iou(s, y, k); rec[nm + "|lat"] = iou(s, ylat, k)
            rows.append(rec)
    r = pd.DataFrame(rows); out = []
    for nm in ["marginal", "tiebreak", "onsetcond", "qdown", "qup"]:
        dn_ = r[nm + "|noisy"] - r["marginal|noisy"]; dl = r[nm + "|lat"] - r["marginal|lat"]
        # date-clustered SE (40 windows on 9 dates)
        g = pd.DataFrame({"date": r.date, "dl": dl}).groupby("date").dl.mean()
        out.append(dict(rule=nm, noisy=r[nm + "|noisy"].mean(), d_noisy=dn_.mean(), latent=r[nm + "|lat"].mean(), d_lat=dl.mean(),
                        se_lat_window=dl.std() / np.sqrt(len(dl)), se_lat_date=g.std() / np.sqrt(len(g))))
    print(pd.DataFrame(out).round(4).to_string(index=False)); print("at-risk days per window (onsetcond): median", r.n_at_risk.median())


if __name__ == "__main__":
    main()
