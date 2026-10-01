"""Physics growth decoder for ongoing Task 2 windows (rules-safe: history T-60..T-5 only, nothing learned).

For each queue region visible at T-5 (contiguous queued links, holes filled; links ordered in travel direction, so
upstream = lower index):
  tail velocity  (a) empirical: least-squares slope of the tail position over the last 6 history slots
                 (b) kinematic wave: w = (q_up - q_q) / (k_up - k_q) using the free-flow link just upstream of the tail
                     and the queued tail link (flows averaged over the last 3 slots), converted to links/slot with lengths
  head: stays at the bottleneck (most downstream queued link) unless the region is shrinking from downstream.
Prediction for h=1..6 (i.e. T+5..T+30 = 2..7 slots after the last history slot): region [tail(h), head(h)].
Evaluated on the 80 organizer-selected TRAIN windows: since nothing is fitted on train, this is an honest estimate of
transfer to a new scenario. Compared with persistence (T-5), with noisy labels and a denoised-truth proxy.
"""
import sys, numpy as np, pandas as pd
sys.path.insert(0, "tfb")
import cache, score_local as sl
from scipy.ndimage import median_filter
REL = cache.REL


def slot_of(ts): return int(ts[11:13]) * 12 + int(ts[14:16]) // 5


def fill(q):
    q = q.copy(); q[..., 1:-1] |= q[..., :-2] & q[..., 2:]; return q


def regions(q):
    out = []; i = 0; L = len(q)
    while i < L:
        if q[i]:
            j = i
            while j + 1 < L and q[j + 1]: j += 1
            out.append((i, j)); i = j + 1
        else: i += 1
    return out


def predict(v_hist, q_hist, vcut, lengths, mode="kw", gain=1.0, head_mode="fixed"):
    """v_hist, q_hist: [12, L] (NaN missing). Returns [6, L] bool."""
    vf_ = pd.DataFrame(v_hist).ffill().to_numpy(); qf_ = pd.DataFrame(q_hist).ffill().to_numpy()
    Q = fill(np.nan_to_num(vf_, nan=1e9) <= vcut[None, :])
    L = Q.shape[1]; pred = np.zeros((6, L), bool); last = Q[-1]
    pos = np.cumsum(lengths) - lengths / 2
    for (a, b) in regions(last):
        # empirical tail trajectory: for each of the last 6 slots, the most upstream queued link of the region overlapping [a, b]
        tails = []
        for t in range(6, 12):
            r = [rg for rg in regions(Q[t]) if rg[1] >= a - 3 and rg[0] <= b + 3]
            tails.append(min(rg[0] for rg in r) if r else np.nan)
        tails = np.array(tails, float); ok = np.isfinite(tails)
        v_emp = np.polyfit(np.arange(6)[ok], tails[ok], 1)[0] if ok.sum() >= 3 else 0.0   # links per slot (negative = growing upstream)
        v_kw = 0.0
        if a - 1 >= 0 and np.isfinite(vf_[-1, a - 1]) and np.isfinite(vf_[-1, a]):
            qu, vu = np.nanmean(qf_[-3:, a - 1]), np.nanmean(vf_[-3:, a - 1]); qq, vq = np.nanmean(qf_[-3:, a]), np.nanmean(vf_[-3:, a])
            ku, kq = qu / max(vu, 1), qq / max(vq, 1)
            if abs(ku - kq) > 1e-6:
                w_kmh = (qu - qq) / (ku - kq)                       # shock speed (km/h), negative = moving upstream
                km_per_slot = w_kmh * 5 / 60; v_kw = km_per_slot / max(np.mean(lengths[max(a - 3, 0):a + 1]), 0.05)
        vel = {"kw": v_kw, "emp": v_emp, "avg": 0.5 * (v_kw + v_emp), "zero": 0.0}[mode] * gain
        head = b
        if head_mode == "emp":
            heads = []
            for t in range(6, 12):
                r = [rg for rg in regions(Q[t]) if rg[1] >= a - 3 and rg[0] <= b + 3]
                heads.append(max(rg[1] for rg in r) if r else np.nan)
            hh = np.array(heads, float); okh = np.isfinite(hh)
            vh = np.polyfit(np.arange(6)[okh], hh[okh], 1)[0] if okh.sum() >= 3 else 0.0
        for h in range(6):
            dt = h + 2                                                # slots after the last history slot (T-5)
            ta = int(np.clip(round(a + vel * dt), 0, L - 1))
            hb = b if head_mode == "fixed" else int(np.clip(round(b + min(vh, 0) * dt), 0, L - 1))
            if hb >= ta: pred[h, ta:hb + 1] = True
    return pred


def main():
    rows = []
    for panel in cache.PANELS[:8]:
        z = cache.load(panel, "train"); links = list(z["links"]); p = sl.net_params(panel, links)
        vcut = 0.6 * p.free_speed_kmh.to_numpy(); lengths = p.length_km.to_numpy(); dates = list(z["dates"])
        wi = pd.read_csv(REL/"task2"/panel/"train"/"window_index.csv"); wi = wi[wi.condition == "queue_ongoing"]
        for w in wi.itertuples():
            d = dates.index(w.date); s0 = slot_of(w.forecast_origin); el = z["elig"][d].astype(bool)
            vh = np.where(el, z["speed"][d], np.nan)[s0 - 12:s0]; qh = np.where(el, z["flow"][d], np.nan)[s0 - 12:s0]
            y = np.nan_to_num(z["speed"][d, s0 + 1:s0 + 7], nan=1e9) <= vcut[None, :]
            ylat = fill(median_filter(np.nan_to_num(z["speed"][d], nan=1e9), size=(3, 1), mode="nearest")[s0 + 1:s0 + 7] <= vcut[None, :])
            k = el[s0 + 1:s0 + 7] & np.isfinite(z["speed"][d, s0 + 1:s0 + 7])
            r = dict(panel=panel, w=w.window_id)
            cands = {"pers": predict(vh, qh, vcut, lengths, mode="zero")}
            for mode in ("kw", "emp", "avg"):
                for g in (0.5, 1.0):
                    cands[f"{mode}_g{g}"] = predict(vh, qh, vcut, lengths, mode=mode, gain=g)
            cands["emp_heademp"] = predict(vh, qh, vcut, lengths, mode="emp", head_mode="emp")
            for lab, yy in (("", y), ("_lat", ylat)):
                for nm, s in cands.items():
                    u = ((s | yy) & k).sum(); r[nm + lab] = 1.0 if u == 0 else (s & yy & k).sum() / u
            rows.append(r)
    r = pd.DataFrame(rows); cols = [c for c in r.columns if c not in ("panel", "w")]
    m = r[cols].mean().round(4); print(pd.DataFrame({"noisy": m[[c for c in cols if not c.endswith("_lat")]].values,
                                                     "latent": m[[c for c in cols if c.endswith("_lat")]].values},
                                                    index=[c for c in cols if not c.endswith("_lat")]))
    base = r["pers_lat"]
    for c in [c for c in cols if c.endswith("_lat") and c != "pers_lat"]:
        d_ = r[c] - base; print(f"{c:22s} paired vs pers_lat: {d_.mean():+.4f} +- {d_.std() / np.sqrt(len(d_)):.4f}")


if __name__ == "__main__":
    main()
