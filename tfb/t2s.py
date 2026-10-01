"""Task 2, rules-safe rebuild.

Contract compliance: every input uses only the published history T-60..T-5 (12 slots) plus earlier same-day data
visible in the masked layer, and train-only statistics. Slot T is NEVER read (organizer: re-reading the state at T
"is not part of the Task 2 participant contract").

Consistency: train rows are built exactly as validation/private look: masked layer for the day (Task 1 targets
blank), history window republished complete, nothing at or after T.

Window sampling (competition_contract.json:137): ongoing iff the visible history holds >= 2 queued observations with
>= 2 on one link; onset otherwise. Ongoing training windows are the first 1..6 slots after that trigger first holds
in a queue episode (validation windows are young queues). Onset training windows: first queue at h=6.

Proxy (critic plan): pseudo-windows on the VALIDATION and PRIVATE masked layers, scored on observed eligible
horizon cells. Methods must beat rules-safe persistence (T-5) on both months.

Usage: t2s.py build | proxy | predict
"""
from __future__ import annotations
import sys, time, numpy as np, pandas as pd, lightgbm as lgb
from pathlib import Path
sys.path.insert(0, "tfb")
import cache
from t2 import statics, queue_freq, ffill, shift_l, dist_to, daytype, pick_set, iou_by_window, Q_PANELS, REL, PARAMS

W = Path.home()/"tfb/work/t2s"; W.mkdir(parents=True, exist_ok=True)
NB = (1, 2, 3, 5, 8)
META = ["y", "known", "wid", "date", "split"]


import os
INCL_INELIG = os.environ.get("T2S_INELIG", "0") == "1"   # released ineligible cells still carry values (noisier evidence)
if INCL_INELIG:
    W = Path.home()/"tfb/work/t2s_inel"; W.mkdir(parents=True, exist_ok=True)


def day_layer(z, d):
    el = z["elig"][d].astype(bool)
    if INCL_INELIG:
        return z["mspeed"][d].copy(), z["mflow"][d].copy(), z["mocc"][d].copy(), el
    return (np.where(el, z["mspeed"][d], np.nan), np.where(el, z["mflow"][d], np.nan), np.where(el, z["mocc"][d], np.nan), el)


def with_history(z, d, s0, mv, mq, mo, el):
    """Republish the 12-slot history from the observed layer (train only; val/private histories are already complete)."""
    if "speed" not in z: return mv, mq, mo
    h = slice(s0 - 12, s0); mv, mq, mo = mv.copy(), mq.copy(), mo.copy()
    keep = np.ones_like(el[h]) if INCL_INELIG else el[h]
    mv[h] = np.where(keep, z["speed"][d, h], np.nan); mq[h] = np.where(keep, z["flow"][d, h], np.nan); mo[h] = np.where(keep, z["occ"][d, h], np.nan)
    return mv, mq, mo


def day_ctx(mv, vcut, s0):
    """Causal day context from slots < s0, forward-filled (blank != free flow)."""
    r = ffill(mv[:s0] / vcut); q = np.nan_to_num(r, nan=9) <= 1.0; t = np.arange(s0)
    first_link = np.where(q.any(0), q.argmax(0), -1)
    lastfree = np.where((~q).any(0), s0 - 1 - np.argmax((~q)[::-1], 0), -1)
    nq = q.sum(1); anyq = nq > 0; cfirst = t[anyq][0] if anyq.any() else -1; L = q.shape[1]
    return dict(since_first_q=np.where(first_link >= 0, s0 - first_link, -1).astype(np.float32),
                run_q=np.where(q[-1], s0 - 1 - lastfree, 0).astype(np.float32), frac_q_today=q.mean(0).astype(np.float32),
                corr_age=np.full(L, s0 - cfirst if cfirst >= 0 else -1, np.float32), nq_peak=np.full(L, nq.max(), np.float32),
                nq_d6=np.full(L, nq[-1] - (nq[-7] if s0 >= 7 else 0), np.float32))


def features(mv, mq, mo, st, s0, dow, freq6, pidx, ident=True):
    """History strictly s0-12..s0-1 (T-60..T-5)."""
    L = mv.shape[1]
    hv = ffill(mv[s0 - 12:s0] / st["vcut"]); hq = ffill(mq[s0 - 12:s0] / st["cap"]); ho = ffill(mo[s0 - 12:s0])
    f = {}
    for g in (0, 1, 2, 3, 5, 8, 11): f[f"r_lag{g}"] = hv[11 - g]
    f["r_min"] = np.nanmin(hv, 0); f["r_mean"] = np.nanmean(hv, 0); f["r_sl3"] = hv[11] - hv[8]; f["r_sl11"] = hv[11] - hv[0]
    f["qc0"] = hq[11]; f["qc3"] = hq[8]; f["qc_max"] = np.nanmax(hq, 0); f["qc_sl"] = hq[11] - hq[5]; f["occ0"] = ho[11]; f["occ_sl"] = ho[11] - ho[8]
    f["obs_cov"] = np.isfinite(mv[s0 - 12:s0]).mean(0)
    last = hv[11]
    for k in NB:
        f[f"nb+{k}"] = shift_l(last, k); f[f"nb-{k}"] = shift_l(last, -k); f[f"nbq+{k}"] = shift_l(hq[11], k); f[f"nbq-{k}"] = shift_l(hq[11], -k)
        f[f"nbsl+{k}"] = shift_l(hv[11] - hv[8], k); f[f"nbsl-{k}"] = shift_l(hv[11] - hv[8], -k)
    queued = np.nan_to_num(last, nan=9) <= 1.0; slow = np.nan_to_num(last, nan=9) <= 1.3
    f["d_q_dn"] = dist_to(queued, 1); f["d_q_up"] = dist_to(queued, -1); f["d_s_dn"] = dist_to(slow, 1); f["d_s_up"] = dist_to(slow, -1)
    hq_all = np.nan_to_num(hv, nan=9) <= 1.0
    f["n_queued"] = np.full(L, queued.sum(), np.float32); f["n_q_hist"] = np.full(L, hq_all.sum(), np.float32); f["link_q_hist"] = hq_all.sum(0).astype(np.float32)
    f["min_r_corr"] = np.full(L, np.nanmin(last))
    f.update(day_ctx(mv, st["vcut"], s0))
    f["length"], f["lanes"], f["on"], f["off"] = st["length"], st["lanes"], st["on"], st["off"]
    if ident:
        f["pos"] = np.arange(L) / L; f["slot"] = np.full(L, s0); f["dow"] = np.full(L, dow); f["panel"] = np.full(L, pidx)
    base = pd.DataFrame({k: np.asarray(v, np.float32) for k, v in f.items()}); base["link"] = np.arange(L)
    parts = []
    for h in range(6):
        b = base.copy(); b["h"] = h + 1
        if freq6 is not None:
            b["freq"] = freq6[h]; b["freq_nb"] = np.nanmax(np.stack([shift_l(freq6[h], k) for k in (-2, -1, 0, 1, 2)]), 0)
        parts.append(b)
    return pd.concat(parts, ignore_index=True)


def ongoing_trigger(qe_hist):
    return qe_hist.sum() >= 2 and (qe_hist.sum(0) >= 2).any()


def windows_for_day(z, d, vc, el, labels):
    """Yield (cond, s0) candidate windows following the contract. labels: [288, L] bool queue labels for the horizon."""
    mv = np.where(el, z["mspeed"][d] if "speed" not in z else z["speed"][d], np.nan)
    qv = np.nan_to_num(mv, nan=1e9) <= vc; lab_any = labels.any(1)
    trig_prev = False; age = 0
    for s0 in range(13, 282):
        qe = (qv & el)[s0 - 12:s0]
        if el[s0 - 12:s0].mean() < 0.7: trig_prev = False; continue
        if not lab_any[s0 + 1:s0 + 7].any(): trig_prev = ongoing_trigger(qe); age = age + 1 if trig_prev else 0; continue
        trig = ongoing_trigger(qe)
        age = age + 1 if trig else 0
        if trig and 1 <= age <= 6: yield "ongoing", s0
        elif not trig and not qe.any() and lab_any[s0 + 6] and not lab_any[s0 + 1:s0 + 6].any(): yield "onset", s0


def build(split="train"):
    masked_hist = split == "trainm"
    if masked_hist: split = "train"
    """Training (split=train) or proxy (split=validation/private) tables with the SAME construction."""
    out = []
    for pi, panel in enumerate(Q_PANELS):
        t0 = time.time(); ztr = cache.load(panel, "train"); st = statics(panel, ztr["links"]); qd, okd, we, S, N = queue_freq(ztr, st["vcut"])
        z = ztr if split == "train" else cache.load(panel, split); dows = daytype(z["dates"]); n = 0
        for d in range(len(z["dates"])):
            mv0, mq0, mo0, el = day_layer(z, d)
            if split == "train":
                sp = z["speed"][d]; labels = np.nan_to_num(sp, nan=1e9) <= st["vcut"]; known_all = el & np.isfinite(sp)
            else:  # proxy: labels only where the masked layer is observed (Task 1 targets / other windows' blanks are unknown)
                sp = np.where(el, z["mspeed"][d], np.nan); labels = np.nan_to_num(sp, nan=1e9) <= st["vcut"]; known_all = el & np.isfinite(sp)
            for cond, s0 in windows_for_day(z, d, st["vcut"], el, labels):
                if split == "train" and cond == "ongoing" and np.random.default_rng(d * 1000 + s0).random() > 0.5: continue
                mv, mq, mo = with_history(z, d, s0, mv0, mq0, mo0, el) if (split == "train" and not masked_hist) else (mv0, mq0, mo0)
                wt = int(dows[d] >= 5)
                if split == "train":
                    f6 = (S[wt][s0 + 1:s0 + 7] - qd[d][s0 + 1:s0 + 7]) / np.maximum(N[wt][s0 + 1:s0 + 7] - okd[d][s0 + 1:s0 + 7], 1)
                else:
                    f6 = S[wt][s0 + 1:s0 + 7] / np.maximum(N[wt][s0 + 1:s0 + 7], 1)
                X = features(mv, mq, mo, st, s0, dows[d], f6, pi)
                X["y"] = labels[s0 + 1:s0 + 7].reshape(-1).astype(np.int8); X["known"] = known_all[s0 + 1:s0 + 7].reshape(-1)
                X["cond"] = 0 if cond == "onset" else 1; X["wid"] = f"{split}|{panel}|{z['dates'][d]}|{s0}"; X["date"] = z["dates"][d]; X["split"] = split
                if masked_hist: X["split"] = "trainm"
                if cond == "onset": X = X[X.h == 6]
                out.append(X); n += 1
        pd.concat(out, ignore_index=True).to_parquet(W/f"{'trainm' if masked_hist else split}_{panel}.parquet"); out = []
        print(split, panel, n, "windows", f"{time.time()-t0:.0f}s", flush=True)


def load(split, cond, frac=1.0, seed=0):
    parts = []
    for p in Q_PANELS:
        x = pd.read_parquet(W/f"{split}_{p}.parquet"); x = x[x.cond == (0 if cond == "onset" else 1)]
        if frac < 1:
            u = x.wid.unique(); keep = set(u[np.random.default_rng(seed).random(len(u)) < frac]); x = x[x.wid.isin(keep)]
        parts.append(x)
    return pd.concat(parts, ignore_index=True)


IDENT = ["link", "pos", "panel", "slot", "dow", "length", "lanes", "on", "off"]
# same-day data before T-60: contract risk (organizer: use only the released [T-60, T) history) -> never used in fits
DAYCTX = ["since_first_q", "run_q", "frac_q_today", "corr_age", "nq_peak", "nq_d6"]
FREQ = ["freq", "freq_nb"]


def proxy(rounds=400):
    """Train on train tables, score on validation+private pseudo-windows (observed cells only). Compare to T-5 persistence."""
    rows = []
    for cond, frac in (("ongoing", 0.35), ("onset", 1.0)):
        tr = load("train", cond, frac); tr = tr[tr.known]
        tests = {s: load(s, cond) for s in ("validation", "private")}
        feats_all = [c for c in tr.columns if c not in META + ["cond"]]
        for name, drop in (("full", []), ("no_ident", IDENT), ("no_freq", FREQ), ("no_ident_no_freq", IDENT + FREQ)):
            fs = [c for c in feats_all if c not in drop]
            m = lgb.train(PARAMS, lgb.Dataset(tr[fs], tr.y), rounds)
            for s, te in tests.items():
                iou = iou_by_window(te, m.predict(te[fs])); rows.append(dict(cond=cond, split=s, method=name, iou=iou.mean(), n=len(iou)))
                print(cond, s, name, round(iou.mean(), 4), len(iou), flush=True)
        for s, te in tests.items():  # rules-safe persistence (T-5), same scoring
            pers = (te.r_lag0.to_numpy() <= 1.0).astype(float) if cond == "ongoing" else np.zeros(len(te))
            d = te[["wid", "y", "known"]].assign(s=pers > 0.5); k = d.known.to_numpy(); y = d.y.to_numpy() == 1; ss = d.s.to_numpy()
            a = pd.DataFrame({"wid": d.wid, "i": ss & y & k, "u": (ss | y) & k}).groupby("wid").sum()
            v = (a.i / a.u.clip(lower=1)).where(a.u > 0, 1.0).mean(); rows.append(dict(cond=cond, split=s, method="persist_T-5", iou=v, n=len(a)))
            print(cond, s, "persist_T-5", round(v, 4), flush=True)
    r = pd.DataFrame(rows); r.to_csv(W/"proxy_results.csv", index=False)
    print(r.pivot_table(index=["cond", "method"], columns="split", values="iou").round(4))


def backtest(split="validation"):
    """Admissibility check (critic rule c): on pseudo-windows of `split`, the OLD t2.py models (with slot T,
    v1 features, final_*_0.txt) must score BELOW persistence-at-T for ongoing, as on the LB (0.59 < 0.73),
    and onset ~0.64. If the proxy does not reproduce this ordering, it is not a proxy."""
    import t2 as T2
    old = {c: lgb.Booster(model_file=str(Path.home()/f"tfb/work/t2/final_{c}_0.txt")) for c in ("onset", "ongoing")}
    res = {"onset": {"model": [], "persT": []}, "ongoing": {"model": [], "persT": []}}
    for pi, panel in enumerate(Q_PANELS):
        ztr = cache.load(panel, "train"); st = statics(panel, ztr["links"]); _, _, _, S, N = queue_freq(ztr, st["vcut"])
        z = cache.load(panel, split); dows = daytype(z["dates"])
        for d in range(len(z["dates"])):
            mv0, mq0, mo0, el = day_layer(z, d); sp = np.where(el, z["mspeed"][d], np.nan)
            labels = np.nan_to_num(sp, nan=1e9) <= st["vcut"]; known = el & np.isfinite(sp)
            for cond, s0 in windows_for_day(z, d, st["vcut"], el, labels):
                wt = int(dows[d] >= 5); f6 = S[wt][s0 + 1:s0 + 7] / np.maximum(N[wt][s0 + 1:s0 + 7], 1)
                X = T2.features(mv0, mq0, mo0, st, s0, dows[d], f6, pi)
                m = old[cond]; p = m.predict(X[m.feature_name()])
                y = labels[s0 + 1:s0 + 7].reshape(-1); k = known[s0 + 1:s0 + 7].reshape(-1)
                if cond == "onset": p = np.where(X.h.to_numpy() == 6, p, 0.0)
                sel = pick_set(p, k)
                persT = np.tile(np.nan_to_num(ffill(mv0[s0 - 12:s0 + 1] / st["vcut"])[12], nan=9) <= 1.0, 6) if cond == "ongoing" else np.zeros_like(k)
                persTm5 = np.tile(np.nan_to_num(ffill(mv0[s0 - 12:s0] / st["vcut"])[11], nan=9) <= 1.0, 6) if cond == "ongoing" else np.zeros_like(k)
                for nm, s in (("model", sel), ("persT", persT), ("persTm5", persTm5)):
                    u = ((s | y) & k).sum(); res[cond].setdefault(nm, []).append(1.0 if u == 0 else (s & y & k).sum() / u)
        print(panel, "done", flush=True)
    for c in res:
        r = pd.DataFrame(res[c]); print(split, c, "ALL", r.mean().round(4).to_dict(), "n=", len(r), flush=True)
        if c == "ongoing":
            for col in ("persT", "persTm5"):
                sub = r[r[col] <= 0.9]; print(split, c, f"{col}<=0.9", sub.mean().round(4).to_dict(), "n=", len(sub), flush=True)


def episode_backtest(split="validation", shifts=(6, 7, 8)):
    """Score old model vs persistence on pseudo-windows placed `shift` slots BEFORE each real ongoing origin:
    their horizon lies inside the real window's republished (complete) history, so labels are fully observed,
    and they sample the SAME queue episodes the leaderboard scores."""
    import t2 as T2
    m = lgb.Booster(model_file=str(Path.home()/"tfb/work/t2/final_ongoing_0.txt"))
    rows = []
    for pi, panel in enumerate(Q_PANELS):
        ztr = cache.load(panel, "train"); st = statics(panel, ztr["links"]); _, _, _, S, N = queue_freq(ztr, st["vcut"])
        z = cache.load(panel, split); dates = list(z["dates"]); dows = daytype(z["dates"])
        wi = pd.read_csv(REL/"task2"/panel/split/"window_index.csv"); wi = wi[wi.condition == "queue_ongoing"]
        for w in wi.itertuples():
            d = dates.index(w.date); o = int(w.forecast_origin[11:13]) * 12 + int(w.forecast_origin[14:16]) // 5
            mv0, mq0, mo0, el = day_layer(z, d); sp = np.where(el, z["mspeed"][d], np.nan)
            labels = np.nan_to_num(sp, nan=1e9) <= st["vcut"]; known = el & np.isfinite(sp)
            for sh in shifts:
                s0 = o - sh
                if s0 < 13: continue
                wt = int(dows[d] >= 5); f6 = S[wt][s0 + 1:s0 + 7] / np.maximum(N[wt][s0 + 1:s0 + 7], 1)
                X = T2.features(mv0, mq0, mo0, st, s0, dows[d], f6, pi); p = m.predict(X[m.feature_name()])
                y = labels[s0 + 1:s0 + 7].reshape(-1); k = known[s0 + 1:s0 + 7].reshape(-1)
                if not (y & k).any(): continue
                sel = pick_set(p, k)
                pT = np.tile(np.nan_to_num(ffill(mv0[s0 - 12:s0 + 1] / st["vcut"])[12], nan=9) <= 1.0, 6)
                pT5 = np.tile(np.nan_to_num(ffill(mv0[s0 - 12:s0] / st["vcut"])[11], nan=9) <= 1.0, 6)
                r = dict(panel=panel, w=w.window_id, shift=sh, known_share=k.mean())
                for nm, s in (("model", sel), ("persT", pT), ("persTm5", pT5)):
                    u = ((s | y) & k).sum(); r[nm] = 1.0 if u == 0 else (s & y & k).sum() / u
                rows.append(r)
    r = pd.DataFrame(rows); print(split, "episode backtest", r[["model", "persT", "persTm5", "known_share"]].mean().round(4).to_dict(), "n=", len(r), flush=True)
    print(r.groupby("shift")[["model", "persT", "persTm5"]].mean().round(4), flush=True)


def episode_eval_safe(models, split, shifts=(6, 7, 8)):
    """Episode backtest for rules-safe t2s models (history T-60..T-5 only)."""
    rows = []
    for pi, panel in enumerate(Q_PANELS):
        ztr = cache.load(panel, "train"); st = statics(panel, ztr["links"]); _, _, _, S, N = queue_freq(ztr, st["vcut"])
        z = cache.load(panel, split); dates = list(z["dates"]); dows = daytype(z["dates"])
        wi = pd.read_csv(REL/"task2"/panel/split/"window_index.csv"); wi = wi[wi.condition == "queue_ongoing"]
        for w in wi.itertuples():
            d = dates.index(w.date); o = int(w.forecast_origin[11:13]) * 12 + int(w.forecast_origin[14:16]) // 5
            mv0, mq0, mo0, el = day_layer(z, d); sp = np.where(el, z["mspeed"][d], np.nan)
            labels = np.nan_to_num(sp, nan=1e9) <= st["vcut"]; known = el & np.isfinite(sp)
            for sh in shifts:
                s0 = o - sh; wt = int(dows[d] >= 5); f6 = S[wt][s0 + 1:s0 + 7] / np.maximum(N[wt][s0 + 1:s0 + 7], 1)
                X = features(mv0, mq0, mo0, st, s0, dows[d], f6, pi)
                y = labels[s0 + 1:s0 + 7].reshape(-1); k = known[s0 + 1:s0 + 7].reshape(-1)
                if not (y & k).any(): continue
                qv = (np.nan_to_num(mv0, nan=1e9) <= st["vcut"]) & el
                if not ongoing_trigger(qv[s0 - 12:s0]): continue
                r = dict(panel=panel, shift=sh)
                pT5 = np.tile(np.nan_to_num(X.r_lag0.to_numpy()[:len(y) // 6], nan=9) <= 1.0, 6)
                cands = {"persTm5": pT5}
                for nm, (m, feats) in models.items(): cands[nm] = pick_set(m.predict(X[feats]), k)
                for nm, s in cands.items():
                    u = ((s | y) & k).sum(); r[nm] = 1.0 if u == 0 else (s & y & k).sum() / u
                rows.append(r)
    r = pd.DataFrame(rows); cols = [c for c in r.columns if c not in ("panel", "shift")]
    print(split, "episode (rules-safe)", r[cols].mean().round(4).to_dict(), "n=", len(r), flush=True)
    return r


def scenario_test(rounds=400):
    """Does training on train+validation beat train-only on PRIVATE episodes? (and sanity on validation episodes)"""
    variants = {}
    for name, splits, drop in (("tr_full", ["train"], []), ("trm_full", ["trainm"], []),
                               ("trva_full", ["train", "validation"], []), ("trmva_full", ["trainm", "validation"], [])):
        t = pd.concat([load(s, "ongoing", 0.35 if s in ("train", "trainm") else 1.0) for s in splits], ignore_index=True); t = t[t.known]
        feats = [c for c in t.columns if c not in META + ["cond"] + drop]
        variants[name] = (lgb.train(PARAMS, lgb.Dataset(t[feats], t.y), rounds), feats); print("trained", name, flush=True)
    episode_eval_safe(variants, "private")
    episode_eval_safe({k: v for k, v in variants.items() if "va" not in k}, "validation")


def onset_decoders(rounds=400):
    """Compare onset decoders on validation/private onset pseudo-windows (h=6 rows only; train-only model, no day ctx)."""
    tr = load("train", "onset"); tr = tr[tr.known]
    feats = [c for c in tr.columns if c not in META + ["cond"] + DAYCTX]
    m = lgb.train(PARAMS, lgb.Dataset(tr[feats], tr.y), rounds)
    for split in ("validation", "private"):
        te = load(split, "onset"); te = te.assign(p=m.predict(te[feats]))
        res = {}
        for wid, g in te.groupby("wid", sort=False):
            p = g.p.to_numpy(); k = g.known.to_numpy(); y = g.y.to_numpy() == 1; L = len(p)
            def iou(s): u = ((s | y) & k).sum(); return 1.0 if u == 0 else (s & y & k).sum() / u
            cands = {"expIoU": pick_set(p, k)}
            order = np.argsort(-np.where(k, p, -1))
            for mm in (1, 2, 3, 4, 5):
                s = np.zeros(L, bool); s[order[:mm]] = True; cands[f"top{mm}"] = s
            # contiguous block around argmax, grown while neighbour p >= frac * p_max
            for frac in (0.3, 0.5, 0.7):
                s = np.zeros(L, bool); i0 = order[0]; s[i0] = True; lo = hi = i0
                while lo - 1 >= 0 and p[lo - 1] >= frac * p[i0]: lo -= 1; s[lo] = True
                while hi + 1 < L and p[hi + 1] >= frac * p[i0]: hi += 1; s[hi] = True
                cands[f"block{frac}"] = s
            s = pick_set(p, k); s2 = s.copy(); s2[1:-1] |= s[:-2] & s[2:]; cands["expIoU_fill"] = s2
            for nm, s in cands.items(): res.setdefault(nm, []).append(iou(s))
        print(split, "onset decoders:", {k_: round(float(np.mean(v)), 4) for k_, v in res.items()}, "n=", len(res["expIoU"]), flush=True)


def hybrid_test(rounds=400):
    """Persistence(T-5, ffill, hole-filled) backbone + model margins, on the 80 shipped TRAIN ongoing windows,
    with the ongoing model trained on train windows whose dates exclude the shipped ones. Rules-safe features."""
    from scipy.ndimage import median_filter
    tr = load("train", "ongoing", 0.35); tr = tr[tr.known]
    ship_dates = set()
    for p in Q_PANELS:
        wi = pd.read_csv(REL/"task2"/p/"train"/"window_index.csv"); ship_dates |= set(wi.date)
    tr = tr[~tr.date.isin(ship_dates)]
    feats = [c for c in tr.columns if c not in META + ["cond"] + DAYCTX + FREQ]
    m = lgb.train(PARAMS, lgb.Dataset(tr[feats], tr.y), rounds); rows = []
    for pi, panel in enumerate(Q_PANELS):
        z = cache.load(panel, "train"); st = statics(panel, z["links"]); _, _, _, S, N = queue_freq(z, st["vcut"])
        dates = list(z["dates"]); dows = daytype(z["dates"]); wi = pd.read_csv(REL/"task2"/panel/"train"/"window_index.csv")
        for w in wi[wi.condition == "queue_ongoing"].itertuples():
            d = dates.index(w.date); s0 = int(w.forecast_origin[11:13]) * 12 + int(w.forecast_origin[14:16]) // 5
            mv, mq, mo, el = day_layer(z, d); mv, mq, mo = with_history(z, d, s0, mv, mq, mo, el)
            wt = int(dows[d] >= 5); X = features(mv, mq, mo, st, s0, dows[d], S[wt][s0 + 1:s0 + 7] / np.maximum(N[wt][s0 + 1:s0 + 7], 1), pi)
            p = m.predict(X[feats]).reshape(6, -1); L = p.shape[1]
            pers1 = np.nan_to_num(X.r_lag0.to_numpy()[:L], nan=9) <= 1.0
            pers = np.repeat(pers1[None], 6, 0); pf = pers.copy(); pf[:, 1:-1] |= pf[:, :-2] & pf[:, 2:]
            y = np.nan_to_num(z["speed"][d, s0 + 1:s0 + 7], nan=1e9) <= st["vcut"][None, :]
            sm = median_filter(np.nan_to_num(z["speed"][d], nan=1e9), size=(3, 1), mode="nearest")[s0 + 1:s0 + 7] <= st["vcut"][None, :]
            ylat = sm.copy(); ylat[:, 1:-1] |= ylat[:, :-2] & ylat[:, 2:]
            k = el[s0 + 1:s0 + 7] & np.isfinite(z["speed"][d, s0 + 1:s0 + 7])
            cands = {"pers": pers, "pers_fill": pf, "model": pick_set(p.reshape(-1), k.reshape(-1)).reshape(6, -1)}
            for add, rem in ((0.5, 0.0), (0.7, 0.0), (0.5, 0.1), (0.7, 0.1), (0.7, 0.2), (0.8, 0.05)):
                cands[f"hyb_a{add}_r{rem}"] = (pf | (p >= add)) & ~(pf & (p < rem))
            r = {}
            for lab, yy in (("", y), ("_lat", ylat)):
                for nm, s in cands.items():
                    u = ((s | yy) & k).sum(); r[nm + lab] = 1.0 if u == 0 else (s & yy & k).sum() / u
            rows.append(r)
    r = pd.DataFrame(rows); cols = [c for c in r.columns if not c.endswith("_lat")]
    out = pd.DataFrame({"noisy": r[cols].mean(), "latent": r[[c + "_lat" for c in cols]].mean().values}).round(4)
    out["d_lat_vs_pers_fill"] = (r[[c + "_lat" for c in cols]].values - r[["pers_fill_lat"]].values).mean(0).round(4)
    out["se"] = ((r[[c + "_lat" for c in cols]].values - r[["pers_fill_lat"]].values).std(0) / np.sqrt(len(r))).round(4)
    print(out, flush=True)


def add_val_test(rounds=400):
    """Idea 2: does adding the validation month's pseudo-windows to training help on the PRIVATE proxy?"""
    for cond, frac, drop in (("ongoing", 0.35, FREQ), ("onset", 1.0, [])):
        tr = load("train", cond, frac); va = load("validation", cond); pr = load("private", cond)
        feats = [c for c in tr.columns if c not in META + ["cond"] + drop]
        for name, t in (("train_only", tr), ("train+val", pd.concat([tr, va], ignore_index=True))):
            t = t[t.known]; m = lgb.train(PARAMS, lgb.Dataset(t[feats], t.y), rounds)
            print(cond, name, "-> private proxy IoU %.4f" % iou_by_window(pr, m.predict(pr[feats])).mean(), flush=True)


def fit_models(train_splits, rounds=400, tag=""):
    """Final rules-safe models: onset = full features, ongoing = no train-frequency prior."""
    ms = {}
    for cond, frac, drop in (("onset", 1.0, []), ("ongoing", 0.35, FREQ)):
        t = pd.concat([load(s, cond, frac if s == "train" else 1.0) for s in train_splits], ignore_index=True); t = t[t.known]
        feats = [c for c in t.columns if c not in META + ["cond"] + drop + DAYCTX]
        assert not set(feats) & set(DAYCTX)
        m = lgb.train(PARAMS, lgb.Dataset(t[feats], t.y), rounds); m.save_model(str(W/f"final{tag}_{cond}.txt")); ms[cond] = (m, feats)
    return ms


def predict_real(split, ms):
    """Predict the organizer windows of a split with rules-safe features (history T-60..T-5 only)."""
    out = []
    for pi, panel in enumerate(Q_PANELS):
        ztr = cache.load(panel, "train"); st = statics(panel, ztr["links"]); _, _, _, S, N = queue_freq(ztr, st["vcut"]); links = list(ztr["links"])
        z = cache.load(panel, split); dates = list(z["dates"]); dows = daytype(z["dates"])
        wi = pd.read_csv(REL/"task2"/panel/split/"window_index.csv")
        tpl = pd.read_csv(REL/"task2"/panel/split/"sample_submission_queue.csv", dtype=str, usecols=["window_id", "timestamp", "link_id"])
        for w in wi.itertuples():
            d = dates.index(w.date); s0 = int(w.forecast_origin[11:13]) * 12 + int(w.forecast_origin[14:16]) // 5
            mv, mq, mo, el = day_layer(z, d)
            if split == "train": mv, mq, mo = with_history(z, d, s0, mv, mq, mo, el)
            wt = int(dows[d] >= 5)
            X = features(mv, mq, mo, st, s0, dows[d], S[wt][s0 + 1:s0 + 7] / np.maximum(N[wt][s0 + 1:s0 + 7], 1), pi)
            cond = "onset" if w.condition == "queue_onset" else "ongoing"; m, feats = ms[cond]
            p = m.predict(X[feats]); known = z["elig"][d, s0 + 1:s0 + 7].reshape(-1).astype(bool)
            if cond == "onset": p = np.where(X.h.to_numpy() == 6, p, 0.0)
            sel = pick_set(p, known)
            if not (sel & known).any(): sel[np.argmax(np.where(known, p, -1))] = True
            if cond == "ongoing":
                # add-only hybrid (hybrid_test on 80 shipped train windows, latent proxy: 0.804 vs persistence-fill 0.748,
                # +0.055 +- 0.021; never removes persisted cells, which is what hurt the pure model on the LB):
                # persistence of the last history slot (T-5, forward-filled), hole-filled, UNION cells with model p >= 0.5
                L = len(links); pers = np.nan_to_num(X.r_lag0.to_numpy()[:L], nan=9) <= 1.0
                pf = np.repeat(pers[None], 6, 0); pf[:, 1:-1] |= pf[:, :-2] & pf[:, 2:]
                sel = (pf | (p.reshape(6, -1) >= 0.5)).reshape(-1)
            ts = [f"{w.date}T{(s0 + h) // 12:02d}:{(s0 + h) % 12 * 5:02d}:00Z" for h in range(1, 7)]
            r = pd.DataFrame({"window_id": w.window_id, "timestamp": np.repeat(ts, len(links)), "link_id": np.tile(links, 6), "queue_pred": sel.astype(int), "p": p})
            r = tpl[tpl.window_id == w.window_id].merge(r, on=["window_id", "timestamp", "link_id"], how="left")
            assert r.queue_pred.notna().all(); out.append(r)
    return pd.concat(out, ignore_index=True)


def make_submission_queue(private_uses_val=True):
    ms_tr = fit_models(["train"], tag="_tr")
    ms_pr = fit_models(["train", "validation"], tag="_trva") if private_uses_val else ms_tr
    va = predict_real("validation", ms_tr); pr = predict_real("private", ms_pr)
    o = pd.concat([va, pr], ignore_index=True); o.to_csv(W/"queue_safe_full.csv", index=False)
    o[["window_id", "timestamp", "link_id", "queue_pred"]].to_csv(W/"queue_safe.csv", index=False)
    print("wrote", len(o), "share queued", round(o.queue_pred.mean(), 4), flush=True)
    # sanity: the same code path on the shipped TRAIN windows, scored against train labels
    from t2 import score_pred_file
    tr = predict_real("train", ms_tr); tr[["window_id", "timestamp", "link_id", "queue_pred"]].to_csv(W/"queue_safe_train.csv", index=False)
    score_pred_file(W/"queue_safe_train.csv")


if __name__ == "__main__":
    a = sys.argv[1]
    if a == "backtest": backtest("validation"); backtest("private")
    if a == "episode": episode_backtest("validation"); episode_backtest("private"); episode_backtest("train")
    if a == "add_val": add_val_test()
    if a == "hybrid": hybrid_test()
    if a == "onset_dec": onset_decoders()
    if a == "scenario": scenario_test()
    if a == "make_sub": make_submission_queue()
    if a == "build":
        for s in (sys.argv[2:] or ("train", "validation", "private")): build(s)
    elif a == "proxy": proxy()
