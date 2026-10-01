"""CRITIQUE.md section 0.8/0.9 measurements (diagnostics only; nothing here is fitted or submitted).
 1 persist   full t2_persist table, noisy + latent labels, paired vs organizer AND vs ffill_inel, plus DIAGNOSTIC slot-T
             rows (the leak), so "the leak was the fill" can be tested as a paired number on the same 40 train windows
 2 missing   per split x condition: share of links whose T-5 cell is missing/ineligible in the released layer, share of links
             where ffill_inel disagrees with organizer, share of links queued under each rule
 3 coverage  index history_coverage vs eligible fraction on [s0-12, s0) and on [s0-11, s0] (which slots the history holds)
 4 onsetmem  U1 onset links at h=6 vs the top-k links of the train frequency prior and vs the k slowest links at T-5
"""
import sys, numpy as np, pandas as pd
from pathlib import Path
sys.path.insert(0, "tfb")
import cache
import t2_persist as tp
from scipy.ndimage import median_filter
from t2 import statics, queue_freq, daytype
REL = cache.REL; Q = tp.Q_PANELS; pd.set_option("display.width", 250)


def iou(s, y, k):
    u = ((s | y) & k).sum(); return 1.0 if u == 0 else (s & y & k).sum() / u


def persist():
    rows = []
    for panel in Q:
        z = cache.load(panel, "train"); vcut = statics(panel, z["links"])["vcut"]; dates = list(z["dates"])
        wi = pd.read_csv(REL/"task2"/panel/"train"/"window_index.csv"); wi = wi[wi.condition == "queue_ongoing"]
        for w in wi.itertuples():
            d = dates.index(w.date); s0 = tp.slot_of(w.forecast_origin); el = z["elig"][d].astype(bool); sp = z["speed"][d]
            h = slice(s0 - 12, s0); hT = slice(s0 - 12, s0 + 1)
            v_el = np.where(el[h], sp[h], np.nan); v_all = sp[h].copy()
            y = np.nan_to_num(sp[s0 + 1:s0 + 7], nan=1e9) <= vcut[None, :]
            ylat = tp.fill(median_filter(np.nan_to_num(sp, nan=1e9), size=(3, 1), mode="nearest")[s0 + 1:s0 + 7] <= vcut[None, :])
            k = el[s0 + 1:s0 + 7] & np.isfinite(sp[s0 + 1:s0 + 7])
            V = tp.variants(v_el, v_all, vcut)
            # DIAGNOSTIC ONLY (illegal input): slot T, eligible-only, forward-filled through T (what the 0.82669 probe did)
            vT = pd.DataFrame(np.where(el[hT], sp[hT], np.nan)).ffill().to_numpy()[-1]
            V["DIAG_persT_ffill"] = np.nan_to_num(vT, nan=1e9) <= vcut
            V["DIAG_persT_only"] = np.nan_to_num(np.where(el[s0], sp[s0], np.nan), nan=1e9) <= vcut
            r = dict(panel=panel, w=w.window_id)
            for nm, q1 in V.items():
                s = np.repeat(q1[None], 6, 0)
                r[f"{nm}|noisy"] = iou(s, y, k); r[f"{nm}|latent"] = iou(s, ylat, k)
            rows.append(r)
    r = pd.DataFrame(rows); names = sorted({c.split("|")[0] for c in r.columns if "|" in c}); out = []
    for nm in names:
        row = dict(variant=nm)
        for lab in ("noisy", "latent"):
            row[lab] = r[f"{nm}|{lab}"].mean()
            for ref in ("organizer", "ffill_inel"):
                dl = r[f"{nm}|{lab}"] - r[f"{ref}|{lab}"]; row[f"d_{ref[:4]}_{lab}"] = dl.mean(); row[f"se_{ref[:4]}_{lab}"] = dl.std() / np.sqrt(len(dl))
        out.append(row)
    o = pd.DataFrame(out).round(4); print("=== 1. persistence variants, 40 train ongoing windows (d_orga = paired vs organizer, d_ffil = paired vs ffill_inel)")
    print(o.to_string(index=False)); return o


def missing():
    print("=== 2. last-slot missingness (released layer) and rule disagreement, per split x condition")
    rows = []
    for split in ("train", "validation", "private"):
        for panel in Q:
            z = cache.load(panel, split); vcut = statics(panel, z["links"])["vcut"]; dates = list(z["dates"])
            sp = z["mspeed"] if "mspeed" in z else z["speed"]
            for w in pd.read_csv(REL/"task2"/panel/split/"window_index.csv").itertuples():
                d = dates.index(w.date); s0 = tp.slot_of(w.forecast_origin); el = z["elig"][d].astype(bool); h = slice(s0 - 12, s0)
                v_el = np.where(el[h], sp[d, h], np.nan); v_all = sp[d, h].copy(); V = tp.variants(v_el, v_all, vcut)
                last_ok = np.isfinite(sp[d, s0 - 1]) & el[s0 - 1]
                rows.append(dict(split=split, cond=w.condition[6:], miss_last=1 - last_ok.mean(), miss_last_raw=1 - np.isfinite(sp[d, s0 - 1]).mean(),
                                 cov_hist=(np.isfinite(sp[d, h]) & el[h]).mean(), q_org=V["organizer"].mean(), q_ffi=V["ffill_inel"].mean(),
                                 n_org=V["organizer"].sum(), n_ffi=V["ffill_inel"].sum(), n_flip=(V["organizer"] != V["ffill_inel"]).sum()))
    r = pd.DataFrame(rows); g = r.groupby(["split", "cond"]).agg(n=("miss_last", "size"), miss_last=("miss_last", "mean"), miss_last_raw=("miss_last_raw", "mean"),
                                                               cov_hist=("cov_hist", "mean"), n_org=("n_org", "mean"), n_ffi=("n_ffi", "mean"), n_flip=("n_flip", "mean"))
    print(g.round(4).to_string()); return r


def coverage():
    print("=== 3. index history_coverage vs eligible fraction of the released history")
    rows = []
    for split in ("train", "validation", "private"):
        for panel in Q:
            z = cache.load(panel, split); dates = list(z["dates"]); sp = z["mspeed"] if "mspeed" in z else z["speed"]
            for w in pd.read_csv(REL/"task2"/panel/split/"window_index.csv").itertuples():
                d = dates.index(w.date); s0 = tp.slot_of(w.forecast_origin); el = z["elig"][d].astype(bool)
                a = (np.isfinite(sp[d, s0 - 12:s0]) & el[s0 - 12:s0]).mean(); b = (np.isfinite(sp[d, s0 - 11:s0 + 1]) & el[s0 - 11:s0 + 1]).mean()
                e = el[s0 - 12:s0].mean(); rows.append(dict(split=split, idx=w.history_coverage, fin_el_Tm60_Tm5=a, fin_el_Tm55_T=b, el_only_Tm60_Tm5=e))
    r = pd.DataFrame(rows)
    for c in ("fin_el_Tm60_Tm5", "fin_el_Tm55_T", "el_only_Tm60_Tm5"):
        print(f"  match(|idx-{c}|<=1e-3): " + ", ".join(f"{s} {np.mean(np.isclose(g.idx, g[c], atol=1e-3)):.3f}" for s, g in r.groupby("split")),
              f"| mean abs diff {np.mean(np.abs(r.idx - r[c])):.4f}")
    return r


def onsetmem():
    print("=== 4. U1 onset links (h=6) vs train frequency prior top-k and vs the k slowest links at the last released slot")
    q = pd.read_csv(Path.home()/"tfb/work/u1u2/queue_U1.csv", dtype={"window_id": str, "link_id": str}); rows = []
    for split in ("validation", "private"):
        for panel in Q:
            ztr = cache.load(panel, "train"); st = statics(panel, ztr["links"]); _, _, _, S, N = queue_freq(ztr, st["vcut"]); links = [str(l) for l in ztr["links"]]
            z = cache.load(panel, split); dates = list(z["dates"]); dows = daytype(z["dates"]); vcut = st["vcut"]
            wi = pd.read_csv(REL/"task2"/panel/split/"window_index.csv"); wi = wi[wi.condition == "queue_onset"]
            for w in wi.itertuples():
                d = dates.index(w.date); s0 = tp.slot_of(w.forecast_origin); wt = int(dows[d] >= 5)
                qq = q[q.window_id == w.window_id]; ts6 = sorted(qq.timestamp.unique())[-1]
                pred = set(qq[(qq.timestamp == ts6) & (qq.queue_pred == 1)].link_id); kk = len(pred)
                f = S[wt][s0 + 6] / np.maximum(N[wt][s0 + 6], 1); topf = {links[i] for i in np.argsort(-f)[:kk]}
                el = z["elig"][d].astype(bool); h = slice(s0 - 12, s0)
                ff = pd.DataFrame(np.where(el[h], z["mspeed"][d, h], np.nan)).ffill().to_numpy()[-1]
                ratio = np.nan_to_num(ff / vcut, nan=9.0); slow = {links[i] for i in np.argsort(ratio)[:kk]}
                rows.append(dict(split=split, panel=panel, k=kk, in_freq_topk=len(pred & topf) / max(kk, 1), in_slow_topk=len(pred & slow) / max(kk, 1),
                                 freq_of_pred=np.mean([f[links.index(l)] for l in pred]) if kk else np.nan, max_freq=f.max(),
                                 ratio_of_pred=np.mean([ratio[links.index(l)] for l in pred]) if kk else np.nan))
    r = pd.DataFrame(rows); print(r.groupby("split")[["k", "in_freq_topk", "in_slow_topk", "freq_of_pred", "max_freq", "ratio_of_pred"]].mean().round(3).to_string())
    print("  per panel:"); print(r.groupby("panel")[["k", "in_freq_topk", "in_slow_topk", "ratio_of_pred"]].mean().round(3).to_string()); return r


if __name__ == "__main__":
    W = Path.home()/"tfb/work/audit"; W.mkdir(parents=True, exist_ok=True)
    for nm in (sys.argv[1:] or ["persist", "missing", "coverage", "onsetmem"]):
        globals()[nm]().to_csv(W/f"{nm}.csv", index=False)
