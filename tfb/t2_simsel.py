"""Simulate the organizer window selector on train and compare with the shipped window_index.

Hypothesis H: per (panel, condition), scan origins chronologically (day, slot); a candidate qualifies if
  coverage(hist T-12..T-1, eligible share) >= 0.7
  onset:   no visible queue in hist (eligible observed speed <= vcut), queue somewhere in horizon T+1..T+6
  ongoing: visible queue at T-1 (last hist slot), queue in horizon, persistence IoU (state at T-1 repeated) <= 0.9
  and origin >= previous selected origin + 360 min. Keep the first 5.
Labels use the observed train layer (all cells) as a proxy of the latent state.
Also reports, for each selected ongoing window, why the previous slot (T-5) did not qualify.
"""
import sys, numpy as np, pandas as pd
sys.path.insert(0, "tfb"); import cache
REL = cache.REL


def vcut_fd(panel, links):
    fd = pd.read_csv(REL/"corridors"/panel/"network"/"fd_parameters.csv", dtype={"link_id": str}).groupby("link_id").free_speed_kmh.mean()
    return 0.6 * fd.reindex(links).to_numpy()


def slot_of(ts): return int(ts[11:13]) * 12 + int(ts[14:16]) // 5

VARIANT = dict(qsrc="elig_Tm1", psrc="elig_Tm1", pers_scope="all")


def _state(q, qe, s0, src):
    return {"elig_Tm1": qe[s0 - 1], "all_Tm1": q[s0 - 1], "all_T": q[s0], "elig_T": qe[s0]}[src]


def denoise(sp, k=3, mode="mean"):
    """Temporal smoothing per link within each day (latent-state proxy)."""
    from scipy.ndimage import uniform_filter1d, median_filter
    x = np.where(np.isfinite(sp), sp, np.nan)
    filled = np.where(np.isfinite(x), x, np.nanmean(x, axis=1, keepdims=True))
    if mode == "median": return median_filter(filled, size=(1, k, 1), mode="nearest")
    return uniform_filter1d(filled, size=k, axis=1, mode="nearest")


def conds(sp, el, vc, d, s0):
    hist = slice(s0 - 12, s0); hor = slice(s0 + 1, s0 + 7)
    cov = el[d, hist].mean()
    q = sp[d] <= vc[None, :]; qe = q & el[d]
    lab = q[hor]; ps = _state(q, qe, s0, VARIANT["psrc"]); pers = np.repeat(ps[None], 6, 0)
    if VARIANT["pers_scope"] == "elig":  # IoU only over eligible horizon cells
        m = el[d, hor]; piou = (pers & lab & m).sum() / max(1, ((pers | lab) & m).sum())
    else:
        piou = (pers & lab).sum() / max(1, (pers | lab).sum())
    return dict(cov=cov, hq=qe[hist].any(), qlast=_state(q, qe, s0, VARIANT["qsrc"]).any(), qhor=lab.any(), piou=piou)

def contract_cond(sp, el, vc, d, s0):
    """competition_contract.json:137 — ongoing iff visible history (T-60..T-5) has >=2 queued observations with >=2 on one link."""
    hist = slice(s0 - 12, s0); hor = slice(s0 + 1, s0 + 7)
    q = sp[d] <= vc[None, :]; qe = (q & el[d])[hist]
    ongoing = qe.sum() >= 2 and (qe.sum(0) >= 2).any()
    return dict(cov=el[d, hist].mean(), ongoing=ongoing, qhor=q[hor].any())


PERS = "vis_Tm1"


def simulate_contract(z, vc, spacing=72, first=12, last=281):
    """Single chronological scan; classify each candidate by the contract rule; shared spacing; 5 per condition."""
    sp = z["speed"]; el = z["elig"].astype(bool); D = sp.shape[0]
    sel = {"onset": [], "ongoing": []}; taken = []
    for d in range(D):
        for s0 in range(first, last + 1):
            g = d * 288 + s0
            if taken and min(abs(g - t) for t in taken) < spacing: continue
            c = contract_cond(sp, el, vc, d, s0)
            if c["cov"] < 0.7 or not c["qhor"]: continue
            k = "ongoing" if c["ongoing"] else "onset"
            if len(sel[k]) >= 5: continue
            if k == "ongoing":
                q = sp[d] <= vc[None, :]; lab = q[s0 + 1:s0 + 7]
                ref = {"vis_Tm1": (q & el[d])[s0 - 1], "all_Tm1": q[s0 - 1], "all_T": q[s0], "none": None}[PERS]
                if ref is not None:
                    pers = np.repeat(ref[None], 6, 0)
                    if (pers & lab).sum() / max(1, (pers | lab).sum()) > 0.9: continue
            sel[k].append((str(z["dates"][d]), s0)); taken.append(g)
            if len(sel["onset"]) >= 5 and len(sel["ongoing"]) >= 5: return sel
    return sel


def simulate_contract_2pass(z, vc, spacing=72, first=12, last=281, hist_elig=True):
    """Pass 1: onset windows (contract: not ongoing) chronologically; pass 2: ongoing windows, spaced from everything."""
    sp = z["speed"]; el = z["elig"].astype(bool) if hist_elig else np.ones_like(z["elig"], bool); D = sp.shape[0]
    sel = {"onset": [], "ongoing": []}; taken = []
    for k in ("onset", "ongoing"):
        for d in range(D):
            for s0 in range(first, last + 1):
                if len(sel[k]) >= 5: break
                g = d * 288 + s0
                if taken and min(abs(g - t) for t in taken) < spacing: continue
                c = contract_cond(sp, el, vc, d, s0)
                if el[d, s0 - 12:s0].mean() < 0.7 if hist_elig else z["elig"][d, s0 - 12:s0].astype(bool).mean() < 0.7: continue
                if not c["qhor"] or (k == "ongoing") != c["ongoing"]: continue
                sel[k].append((str(z["dates"][d]), s0)); taken.append(g)
    return sel


def simulate(panel, cond, z, vc, spacing=72, first=12, last=281, taken=()):
    """taken: global slots (d*288+s0) of windows already selected in other conditions; keep >= spacing from them."""
    sp = z["speed"]; el = z["elig"].astype(bool); D = sp.shape[0]; sel = []; prev = -10**9
    taken = np.array(list(taken), dtype=np.int64)
    for d in range(D):
        for s0 in range(first, last + 1):
            g = d * 288 + s0
            if g - prev < spacing: continue
            if len(taken) and np.min(np.abs(taken - g)) < spacing: continue
            c = conds(sp, el, vc, d, s0)
            if c["cov"] < 0.7 or not c["qhor"]: continue
            if cond == "onset" and c["hq"]: continue
            if cond == "ongoing" and (not c["qlast"] or c["piou"] > 0.9): continue
            sel.append((str(z["dates"][d]), s0)); prev = g
            if len(sel) == 5: return sel
    return sel

import itertools
if "--contract2" in sys.argv:
    for he in (True, False):
        tot = {"onset": 0, "ongoing": 0}
        for panel in cache.PANELS[:8]:
            z = cache.load(panel, "train"); vc = vcut_fd(panel, z["links"])
            wi = pd.read_csv(REL/"task2"/panel/"train"/"window_index.csv"); sel = simulate_contract_2pass(z, vc, hist_elig=he)
            for cond in ("onset", "ongoing"):
                act = [(r.date, slot_of(r.forecast_origin)) for r in wi[wi.condition == "queue_" + cond].itertuples()]
                tot[cond] += sum(a == b for a, b in zip(act, sel[cond]))
                if cond == "ongoing" and panel == "D12_I5_S": print("  act", act, "\n  sim", sel[cond])
        print("2pass hist_elig", he, "TOTAL", tot, flush=True)
    sys.exit()
if "--contract" in sys.argv:
  for PERS in ("none", "all_Tm1", "all_T", "vis_Tm1"):
    tot = {"onset": 0, "ongoing": 0}
    for panel in cache.PANELS[:8]:
        z = cache.load(panel, "train"); vc = vcut_fd(panel, z["links"])
        wi = pd.read_csv(REL/"task2"/panel/"train"/"window_index.csv"); sel = simulate_contract(z, vc)
        for cond in ("onset", "ongoing"):
            act = [(r.date, slot_of(r.forecast_origin)) for r in wi[wi.condition == "queue_" + cond].itertuples()]
            n = sum(a == b for a, b in zip(act, sel[cond])); tot[cond] += n
    print("PERS", PERS, "TOTAL exact matches:", tot, flush=True)
  sys.exit()
if "--latent" in sys.argv:
    for mode, k in (("mean", 3), ("median", 3), ("mean", 5), ("median", 5)):
        for qs, ps_ in (("elig_Tm1", "all_T"), ("all_T", "all_T"), ("elig_Tm1", "elig_Tm1"), ("all_Tm1", "all_Tm1")):
            VARIANT.update(qsrc=qs, psrc=ps_, pers_scope="all"); tot_on = tot_og = 0
            for panel in cache.PANELS[:8]:
                z = dict(cache.load(panel, "train")); z["speed"] = denoise(z["speed"], k, mode)
                vc = vcut_fd(panel, z["links"]); dates = list(z["dates"])
                wi = pd.read_csv(REL/"task2"/panel/"train"/"window_index.csv")
                act_on = [(r.date, slot_of(r.forecast_origin)) for r in wi[wi.condition == "queue_onset"].itertuples()]
                act_og = [(r.date, slot_of(r.forecast_origin)) for r in wi[wi.condition == "queue_ongoing"].itertuples()]
                sim_on = simulate(panel, "onset", z, vc)
                taken = [dates.index(a) * 288 + s for a, s in sim_on]
                sim_og = simulate(panel, "ongoing", z, vc, taken=taken)
                tot_on += sum(a == b for a, b in zip(act_on, sim_on)); tot_og += sum(a == b for a, b in zip(act_og, sim_og))
            print(mode, k, qs, ps_, "onset", tot_on, "/40 ongoing", tot_og, "/40", flush=True)
    sys.exit()
if "--grid" in sys.argv:
    Z = {p: (cache.load(p, "train"),) for p in cache.PANELS[:8]}
    for qs, ps_, sc in itertools.product(["elig_Tm1", "all_Tm1", "all_T", "elig_T"], ["elig_Tm1", "all_Tm1", "all_T", "elig_T"], ["all", "elig"]):
        VARIANT.update(qsrc=qs, psrc=ps_, pers_scope=sc); tot = 0; near = 0
        for panel in cache.PANELS[:8]:
            z = Z[panel][0]; vc = vcut_fd(panel, z["links"]); dates = list(z["dates"])
            wi = pd.read_csv(REL/"task2"/panel/"train"/"window_index.csv")
            taken = [dates.index(r.date) * 288 + slot_of(r.forecast_origin) for r in wi[wi.condition == "queue_onset"].itertuples()]
            act = [(r.date, slot_of(r.forecast_origin)) for r in wi[wi.condition == "queue_ongoing"].itertuples()]
            sim = simulate(panel, "ongoing", z, vc, taken=taken)
            tot += sum(a == b for a, b in zip(act, sim))
        print(qs, ps_, sc, "exact", tot, "/40", flush=True)
    sys.exit()

rows = []
for panel in cache.PANELS[:8]:
    z = cache.load(panel, "train"); vc = vcut_fd(panel, z["links"])
    wi = pd.read_csv(REL/"task2"/panel/"train"/"window_index.csv")
    dates = list(z["dates"]); taken = []
    for cond in ("onset", "ongoing"):
        act = [(r.date, slot_of(r.forecast_origin)) for r in wi[wi.condition == "queue_" + cond].itertuples()]
        sim = simulate(panel, cond, z, vc, taken=taken if "--shared" in sys.argv else ())
        taken += [dates.index(a) * 288 + s for a, s in act]  # use actual onset origins as taken
        rows.append(dict(panel=panel, cond=cond, match=sum(a == b for a, b in zip(act, sim)), actual=act, sim=sim))
        print(panel, cond, "match", rows[-1]["match"], "/5\n  act", act, "\n  sim", sim, flush=True)
