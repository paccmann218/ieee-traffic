"""Find the ongoing acceptance condition: features that are True at the actual origin s0 and False at s0-1
(for windows whose s0-1 is not blocked by spacing from another selected window)."""
import sys, numpy as np, pandas as pd
sys.path.insert(0, "tfb"); import cache
from t2_simsel import vcut_fd, slot_of
REL = cache.REL


def feats(q, el, cov_el, c):
    """Boolean/numeric candidate conditions for origin slot c."""
    qe = q & el; lab = q[c + 1:c + 7]; k = el[c + 1:c + 7]
    f = {}
    for j in (1, 2, 3):
        f[f"vis_q_Tm{j}"] = qe[c - j].any(); f[f"all_q_Tm{j}"] = q[c - j].any()
    f["all_q_T"] = q[c].any(); f["vis_q_T"] = qe[c].any()
    f["vis_q_2consec"] = (qe[c - 1] & qe[c - 2]).any(); f["vis_q_last2_any"] = qe[c - 1].any() and qe[c - 2].any()
    f["all_q_T_and_Tm1"] = q[c].any() and q[c - 1].any()
    f["hq_vis_count_ge2slots"] = qe[c - 12:c].any(1).sum() >= 2; f["hq_vis_count_ge3slots"] = qe[c - 12:c].any(1).sum() >= 3
    f["lab_h1"] = lab[0].any(); f["lab_all_h"] = lab.any(1).all(); f["lab_h1_elig"] = (lab[0] & k[0]).any(); f["lab_all_h_elig"] = (lab & k).any(1).all()
    for ref_name, ref in (("Tm1v", qe[c - 1]), ("Tm1", q[c - 1]), ("T", q[c]), ("Tv", qe[c])):
        pers = np.repeat(ref[None], 6, 0)
        for sc_name, m in (("all", np.ones_like(k)), ("el", k)):
            u = ((pers | lab) & m).sum(); i_ = (pers & lab & m).sum(); iou = 1.0 if u == 0 else i_ / u
            f[f"piou_{ref_name}_{sc_name}<=.9"] = iou <= 0.9
    f["cov>=.7"] = cov_el[c - 12:c].mean() >= 0.7
    f["cov13>=.7"] = cov_el[c - 12:c + 1].mean() >= 0.7
    f["fut_cov=1"] = True
    return f


rows = []
for panel in cache.PANELS[:8]:
    z = cache.load(panel, "train"); vc = vcut_fd(panel, z["links"]); dates = list(z["dates"])
    wi = pd.read_csv(REL/"task2"/panel/"train"/"window_index.csv")
    allsel = [dates.index(r.date) * 288 + slot_of(r.forecast_origin) for r in wi.itertuples()]
    for r in wi[wi.condition == "queue_ongoing"].itertuples():
        d = dates.index(r.date); s0 = slot_of(r.forecast_origin); g = d * 288 + s0
        others = [x for x in allsel if x != g]
        blocked_prev = any(abs(x - (g - 1)) < 72 for x in others if x < g)
        q = np.nan_to_num(z["speed"][d], nan=1e9) <= vc[None, :]; el = z["elig"][d].astype(bool)
        f0 = feats(q, el, el, s0); f1 = feats(q, el, el, s0 - 1)
        rows.append(dict(panel=panel, s0=s0, blocked_prev=blocked_prev, **{k: (f0[k], f1[k]) for k in f0}))
df = pd.DataFrame(rows)
free = df[~df.blocked_prev]
print("ongoing windows:", len(df), "with s0-1 not blocked by spacing:", len(free))
res = []
for k in [c for c in df.columns if c not in ("panel", "s0", "blocked_prev")]:
    at_s0 = df[k].map(lambda t: t[0]).mean(); flip = free[k].map(lambda t: t[0] and not t[1]).mean()
    res.append(dict(cond=k, true_at_s0_all=round(at_s0, 3), true_s0_false_prev_free=round(flip, 3)))
print(pd.DataFrame(res).sort_values("true_s0_false_prev_free", ascending=False).to_string(index=False))
