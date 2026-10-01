"""Task 2 non-learned legal persistence variants (history [T-60, T) only; never slot T).

Variants for the ongoing queue state carried through the 6 horizon steps:
  organizer        eligible observation at the last history slot T-5 (the official baseline's rule)
  ffill            last eligible observation per link within the 12-slot history
  ffill_inel       last released observation per link, including ineligible (pct_observed<75) values
  *_fill           single-link hole filling (latent queues are contiguous)
  mean2            mean of the last two (forward-filled) eligible history slots, a light denoise
Evaluation: the 80 released TRAIN windows (ongoing condition), labels from the unmasked train layer (noisy) and a
latent proxy (3-slot temporal median + hole fill). Admissible for non-learned methods only (nothing is fitted).
Usage: t2_persist.py eval | (import) build_rows(variant, split)
"""
import sys, numpy as np, pandas as pd
sys.path.insert(0, "tfb")
import cache
from scipy.ndimage import median_filter
from t2 import statics

REL = cache.REL
Q_PANELS = cache.PANELS[:8]


def slot_of(ts): return int(ts[11:13]) * 12 + int(ts[14:16]) // 5


def fill(q):
    q = q.copy(); q[..., 1:-1] |= q[..., :-2] & q[..., 2:]; return q


def variants(v_el, v_all, vcut):
    """v_el: [12, L] eligible-only history speeds; v_all: [12, L] all released history values. -> name -> [L] bool."""
    out = {}
    out["organizer"] = np.nan_to_num(v_el[-1], nan=1e9) <= vcut
    ff = pd.DataFrame(v_el).ffill().to_numpy(); ffa = pd.DataFrame(v_all).ffill().to_numpy()
    out["ffill"] = np.nan_to_num(ff[-1], nan=1e9) <= vcut
    out["ffill_inel"] = np.nan_to_num(ffa[-1], nan=1e9) <= vcut
    out["ffill_fill"] = fill(out["ffill"]); out["ffill_inel_fill"] = fill(out["ffill_inel"])
    with np.errstate(all="ignore"):
        m2 = np.nanmean(ff[-2:], 0)
    out["mean2"] = np.nan_to_num(m2, nan=1e9) <= vcut; out["mean2_fill"] = fill(out["mean2"])
    return out


def evaluate():
    rows = []
    for panel in Q_PANELS:
        z = cache.load(panel, "train"); st = statics(panel, z["links"]); vcut = st["vcut"]; dates = list(z["dates"])
        wi = pd.read_csv(REL/"task2"/panel/"train"/"window_index.csv"); wi = wi[wi.condition == "queue_ongoing"]
        for w in wi.itertuples():
            d = dates.index(w.date); s0 = slot_of(w.forecast_origin); el = z["elig"][d].astype(bool)
            h = slice(s0 - 12, s0)                       # history only
            v_el = np.where(el[h], z["speed"][d, h], np.nan); v_all = z["speed"][d, h].copy()
            y = np.nan_to_num(z["speed"][d, s0 + 1:s0 + 7], nan=1e9) <= vcut[None, :]
            ylat = fill(median_filter(np.nan_to_num(z["speed"][d], nan=1e9), size=(3, 1), mode="nearest")[s0 + 1:s0 + 7] <= vcut[None, :])
            k = el[s0 + 1:s0 + 7] & np.isfinite(z["speed"][d, s0 + 1:s0 + 7])
            r = dict(panel=panel, w=w.window_id)
            for nm, q1 in variants(v_el, v_all, vcut).items():
                s = np.repeat(q1[None], 6, 0)
                for lab, yy in (("noisy", y), ("latent", ylat)):
                    u = ((s | yy) & k).sum(); r[f"{nm}|{lab}"] = 1.0 if u == 0 else (s & yy & k).sum() / u
            rows.append(r)
    r = pd.DataFrame(rows); names = sorted({c.split("|")[0] for c in r.columns if "|" in c})
    out = []
    for nm in names:
        row = dict(variant=nm)
        for lab in ("noisy", "latent"):
            dlt = r[f"{nm}|{lab}"] - r[f"organizer|{lab}"]
            row[lab] = r[f"{nm}|{lab}"].mean(); row[f"d_{lab}"] = dlt.mean(); row[f"se_{lab}"] = dlt.std() / np.sqrt(len(dlt))
        out.append(row)
    o = pd.DataFrame(out).round(4); print(o.to_string(index=False), "\n n windows =", len(r), flush=True)
    return o


def build_rows(variant, split):
    """Template-keyed ongoing rows (window_id, timestamp, link_id, queue_pred) for one split, legal history only."""
    out = []
    for panel in Q_PANELS:
        ztr = cache.load(panel, "train"); st = statics(panel, ztr["links"]); vcut = st["vcut"]; links = list(ztr["links"])
        z = cache.load(panel, split); assert list(z["links"]) == links; dates = list(z["dates"])
        wi = pd.read_csv(REL/"task2"/panel/split/"window_index.csv"); wi = wi[wi.condition == "queue_ongoing"]
        for w in wi.itertuples():
            d = dates.index(w.date); s0 = slot_of(w.forecast_origin); el = z["elig"][d].astype(bool); h = slice(s0 - 12, s0)
            assert 12 <= s0 and s0 + 6 < 288, f"window {w.window_id} crosses midnight: timestamp keys would not match"
            v_el = np.where(el[h], z["mspeed"][d, h], np.nan); v_all = z["mspeed"][d, h].copy()
            q1 = variants(v_el, v_all, vcut)[variant]
            ts = [f"{w.date}T{(s0 + k) // 12:02d}:{(s0 + k) % 12 * 5:02d}:00Z" for k in range(1, 7)]
            out.append(pd.DataFrame({"window_id": w.window_id, "timestamp": np.repeat(ts, len(links)),
                                     "link_id": np.tile(links, 6), "queue_pred": np.tile(q1.astype(int), 6)}))
    return pd.concat(out, ignore_index=True)


if __name__ == "__main__":
    evaluate()
