"""Analyse organizer-selected Task 2 train windows against hypotheses of the selector."""
import sys, numpy as np, pandas as pd
sys.path.insert(0, "tfb"); import cache
from pathlib import Path
REL = cache.REL
Q_PANELS = cache.PANELS[:8]
pd.set_option("display.width", 250)


def vcut_fd(panel, links):
    fd = pd.read_csv(REL/"corridors"/panel/"network"/"fd_parameters.csv", dtype={"link_id": str}).groupby("link_id").free_speed_kmh.mean()
    return 0.6 * fd.reindex(links).to_numpy()


def slot_of(ts): return int(ts[11:13]) * 12 + int(ts[14:16]) // 5


rows = []
for panel in Q_PANELS:
    z = cache.load(panel, "train"); links = z["links"]; dates = list(z["dates"]); vc = vcut_fd(panel, links)
    sp = z["speed"]; el = z["elig"].astype(bool)
    wi = pd.read_csv(REL/"task2"/panel/"train"/"window_index.csv")
    for w in wi.itertuples():
        d = dates.index(w.date); s0 = slot_of(w.forecast_origin)
        hist = slice(s0 - 12, s0); hor = slice(s0 + 1, s0 + 7)
        cov = el[d, hist].mean()
        q = sp[d] <= vc[None, :]; qe = q & el[d]
        hq_vis = qe[hist].any(); hq_last = qe[s0 - 1].any(); q_at_T = q[s0].any()
        lab = q[hor]; first_h = int(np.argmax(lab.any(1))) + 1 if lab.any() else 0
        pers = np.repeat(q[s0 - 1][None], 6, 0); pers_iou = (pers & lab).sum() / max(1, (pers | lab).sum())
        persT = np.repeat(q[s0][None], 6, 0); persT_iou = (persT & lab).sum() / max(1, (persT | lab).sum())
        rows.append(dict(panel=panel, wid=w.window_id[-3:], cond=w.condition[6:], date=w.date, origin=w.forecast_origin[11:16], s0=s0,
                         cov_idx=w.history_coverage, cov_mine=round(cov, 4), hq_vis=hq_vis, hq_last=hq_last, q_at_T=q_at_T,
                         n_lab=int(lab.sum()), first_h=first_h, pers_iou=round(pers_iou, 3), persT_iou=round(persT_iou, 3)))
df = pd.DataFrame(rows); (Path.home()/"tfb/work").mkdir(exist_ok=True); df.to_csv(Path.home()/"tfb/work/t2_train_windows.csv", index=False)
print(df.to_string())
print(df.groupby("cond")[["hq_vis", "hq_last", "q_at_T"]].mean())
print("coverage match:", np.mean(np.isclose(df.cov_idx, df.cov_mine, atol=1e-3)))
