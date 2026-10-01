"""Physics proxy with a LATENT (denoised) conservation reference.

Hypothesis: the organizer boundary flux makes the noise-free state conserve, so rhs = dN_latent, and a perfect
(observed) answer scores S_LWR ~0.945 (docs: perfect physics 0.96 with S_FD ~0.99).
We approximate the latent field by smoothing the observed truth and CALIBRATE the smoothing so that the observed truth
scores S_LWR ~0.945. Admissibility: the proxy must then give time-interpolation and our GBDT physics ~0.65 (LB-derived).
Test months Dec-Feb, CV predictions from t1_train cv (work/t1feat/cv_preds.parquet).
"""
import sys, numpy as np, pandas as pd
sys.path.insert(0, "tfb")
import cache, score_local as sl, t1_base as tb
from scipy.ndimage import uniform_filter1d, median_filter
from pathlib import Path

te = pd.read_parquet(Path.home()/"tfb/work/t1feat/cv_preds.parquet")


def latent(x, mode, k):
    f = np.where(np.isfinite(x), x, np.nan); filled = pd.DataFrame(f.reshape(-1, f.shape[-1])).interpolate(limit_direction="both").to_numpy().reshape(f.shape)
    if mode == "mean": return uniform_filter1d(filled, k, axis=1, mode="nearest")
    if mode == "median": return median_filter(filled, size=(1, k, 1), mode="nearest")
    if mode == "st": return uniform_filter1d(uniform_filter1d(filled, k, axis=1, mode="nearest"), 3, axis=2, mode="nearest")
    return filled


class T3L(sl.T3):
    """Same transitions/filters as score_local.T3 but rhs = dN of a latent field (Nl) instead of the observed truth."""
    def score_lat(self, z, pv, pq, Vl, Ql):
        p = self.p; Lk = p.length_km.to_numpy(); out = {}
        dates = z["dates"]
        for r in sl.REGIMES:
            dm, el, tg, V, Q = self.field(z, pv, pq, r)
            N = Q / np.maximum(V, 1.0) * Lk; Nl = np.where(el, Ql[dm] / np.maximum(Vl[dm], 1.0), np.nan) * Lk
            d_idx = np.flatnonzero(dm); cons = np.r_[(np.diff(pd.to_datetime(dates[d_idx]).values).astype("timedelta64[D]").astype(int) == 1), False]
            Nf = N.reshape(-1, N.shape[-1]); Nlf = Nl.reshape(-1, N.shape[-1]); elf = el.reshape(-1, N.shape[-1]); rok = self.ramp_ok[dm].reshape(-1, N.shape[-1])
            nxt = np.ones(Nf.shape[0], bool); nxt[-1] = False; de = np.arange(Nf.shape[0]) % 288 == 287; nxt[de] = cons[: de.sum()]
            valid = elf[:-1] & elf[1:] & nxt[:-1, None] & rok[:-1] & self.observable[None, :]
            dN = Nf[1:] - Nf[:-1]; dNl = Nlf[1:] - Nlf[:-1]
            out[r] = max(0.0, 1 - min(1.0, np.abs(dN - dNl)[valid].sum() / (np.abs(dNl)[valid].sum() + 1e-9)))
        return float(np.mean(list(out.values())))


rows = []
panels = sys.argv[1:] or ["D12_I5_N", "D7_I405_N", "D7_I10_E"]
for panel in panels:
    pi = cache.PANELS.index(panel); z = cache.load(panel, "train")
    mon = pd.to_datetime(z["dates"]).month.to_numpy(); keep = np.isin(mon, [12, 1, 2])
    zz = {k: (v[keep] if isinstance(v, np.ndarray) and v.ndim >= 1 and v.shape[0] == len(mon) else v) for k, v in z.items()}
    t3 = T3L(panel, z); t3.ramp_ok = t3.ramp_ok[keep]
    sub = te[te.panel == pi]; dmap = {d: i for i, d in enumerate(np.flatnonzero(keep))}; dd = sub.d.map(dmap).to_numpy()
    GV = np.full(zz["speed"].shape, np.nan, np.float32); GQ = GV.copy(); GV[dd, sub.t, sub.l] = sub.pv; GQ[dd, sub.t, sub.l] = sub.pq
    IV, IQ = tb.time_interp(zz, "speed"), tb.time_interp(zz, "flow")
    for mode, k in (("none", 1), ("mean", 3), ("median", 3), ("mean", 5), ("st", 3)):
        Vl, Ql = latent(zz["speed"], mode, k), latent(zz["flow"], mode, k)
        r = dict(panel=panel, latent=f"{mode}{k}")
        for name, (a, b) in (("truth", (zz["speed"], zz["flow"])), ("tinterp", (IV, IQ)), ("gbdt", (GV, GQ)), ("latent_pred", (Vl, Ql))):
            r[name] = t3.score_lat(zz, a, b, Vl, Ql)
        rows.append(r); print(r, flush=True)
r = pd.DataFrame(rows); print(r.groupby("latent")[["truth", "tinterp", "gbdt", "latent_pred"]].mean().round(4))
