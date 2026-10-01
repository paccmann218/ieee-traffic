"""Distribution shift between organizer-selected ongoing windows and generated ongoing windows (train)."""
import sys, numpy as np, pandas as pd
sys.path.insert(0, "tfb")
import cache, t2

def wstats(df):
    g = df.assign(qT=(df.r_lag0 <= 1.0)).groupby("wid")
    k = df.known.to_numpy(); y = df.y.to_numpy() == 1; s = (df.r_lag0 <= 1.0).to_numpy()
    tmp = pd.DataFrame({"wid": df.wid, "i": s & y & k, "u": (s | y) & k, "y": y & k, "qT": s & (df.h == 1).to_numpy()})
    a = tmp.groupby("wid").sum()
    return pd.DataFrame({"pers_iou": a.i / a.u.clip(lower=1), "n_true": a.y, "n_qT": a.qT,
                         "slot": g.slot.first(), "corr_age": g.corr_age.first(), "nq_d6": g.nq_d6.first(), "dow": g.dow.first()})

gen = t2.load_cond("ongoing", 0.02); G = wstats(gen)
rows = []
for pi, panel in enumerate(t2.Q_PANELS):
    z = cache.load(panel, "train"); st = t2.statics(panel, z["links"]); qd, okd, we, S, N = t2.queue_freq(z, st["vcut"]); dates = list(z["dates"]); dows = t2.daytype(z["dates"])
    wi = pd.read_csv(cache.REL/"task2"/panel/"train"/"window_index.csv")
    for w in wi[wi.condition == "queue_ongoing"].itertuples():
        d = dates.index(w.date); s0 = int(w.forecast_origin[11:13]) * 12 + int(w.forecast_origin[14:16]) // 5
        mv, mq, mo = t2.visible(z, d, True)
        X = t2.features(mv, mq, mo, st, s0, dows[d], np.zeros((6, mv.shape[1])), pi)
        X["y"] = (np.nan_to_num(z["speed"][d, s0 + 1:s0 + 7], nan=1e9) <= st["vcut"]).reshape(-1).astype(int)
        X["known"] = np.isfinite(z["speed"][d, s0 + 1:s0 + 7]).reshape(-1) & z["elig"][d, s0 + 1:s0 + 7].reshape(-1).astype(bool)
        X["wid"] = w.window_id; rows.append(X)
Sx = wstats(pd.concat(rows, ignore_index=True))
q = [0.1, 0.25, 0.5, 0.75, 0.9]
print("GENERATED\n", G.quantile(q).round(2).to_string()); print("SELECTED\n", Sx.quantile(q).round(2).to_string())
print("slot hist gen:", np.histogram(G.slot, bins=range(0, 289, 24))[0]); print("slot hist sel:", np.histogram(Sx.slot, bins=range(0, 289, 24))[0])
