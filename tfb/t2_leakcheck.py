"""Research only: how much does post-buffer information (T+95..) reveal the horizon? Compare on shipped train windows.

Methods (queue state per link over h=1..6):
  persist_Tm1 : visible state at T-5 repeated          (the organizer baseline)
  persist_T   : state at T (slot s0) repeated          (causal)
  interp_post : link queued if queued at T AND at T+95 (first slot after the 60-min buffer), OR queued at both T and h-slot
                -> uses future data (NOT causal); only to understand what top-LB scores could be built on.
Also: check in validation masked layer which slots after the horizon are blanked.
"""
import sys, numpy as np, pandas as pd
sys.path.insert(0, "tfb"); import cache
from t2_simsel import vcut_fd, slot_of
REL = cache.REL
rows = []
for panel in cache.PANELS[:8]:
    z = cache.load(panel, "train"); vc = vcut_fd(panel, z["links"]); dates = list(z["dates"]); el = z["elig"].astype(bool)
    wi = pd.read_csv(REL/"task2"/panel/"train"/"window_index.csv")
    for w in wi.itertuples():
        d = dates.index(w.date); s0 = slot_of(w.forecast_origin)
        q = np.nan_to_num(z["speed"][d], nan=1e9) <= vc; lab = q[s0 + 1:s0 + 7]; k = el[d, s0 + 1:s0 + 7]
        def iou(pred):
            inter = (pred & lab & k).sum(); uni = ((pred | lab) & k).sum(); return 1.0 if uni == 0 else inter / uni
        pT1 = np.repeat((q[s0 - 1] & el[d, s0 - 1])[None], 6, 0); pT = np.repeat(q[s0][None], 6, 0)
        post = s0 + 19 if s0 + 19 < 288 else 287
        both = np.repeat((q[s0] & q[post])[None], 6, 0)
        lin = np.stack([q[s0] if h < 3 else (q[s0] & q[post]) | (q[post] & (h >= 4)) for h in range(6)])
        rows.append(dict(panel=panel, cond=w.condition[6:], pers_Tm1=iou(pT1), pers_T=iou(pT), both=iou(both), lin=iou(lin)))
r = pd.DataFrame(rows); print(r.groupby("cond")[["pers_Tm1", "pers_T", "both", "lin"]].mean().round(3))
# blanking extent after horizon on validation
p = "D7_I10_E"; z = cache.load(p, "validation"); dates = list(z["dates"]); wi = pd.read_csv(REL/"task2"/p/"validation"/"window_index.csv")
w = wi.iloc[5]; d = dates.index(w.date); s0 = slot_of(w.forecast_origin)
el = z["elig"][d].astype(bool); vis = np.isfinite(z["mspeed"][d]) & el
print("visible share of eligible cells by slot offset from T:", {o: round(vis[s0 + o][el[s0 + o]].mean(), 2) for o in range(-2, 24) if s0 + o < 288})
