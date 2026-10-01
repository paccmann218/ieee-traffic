"""Task 2 queue forecasting.

Selector facts (reproduced on train, see t2_simsel.py): onset windows = earliest origin (chronological, spacing 360 min
shared across conditions) with no visible queue in T-60..T-5 and a queue in T+5..T+30 -> the first queued cell is
at T+30 (host-approved regularity, discussion/742328). Ongoing: visible queue at T-5, queue in horizon,
persistence IoU <= 0.9, history coverage >= 0.7.

Causality: every feature for origin slot s0 uses slots <= s0 of the same day (masked layer for validation/private;
for train the observed layer restricted to what the masked layer would show) plus train-only statistics.

Usage: t2.py build | cv | predict
"""
from __future__ import annotations
import sys, time, numpy as np, pandas as pd, lightgbm as lgb
from pathlib import Path
sys.path.insert(0, "tfb")
import cache

REL = cache.REL; W = Path.home()/"tfb/work/t2"; W.mkdir(parents=True, exist_ok=True)
Q_PANELS = cache.PANELS[:8]
OG, ON = "og2_", "onset2"   # v2 tables: shockwave features + hard ongoing filter at build time
NB = (1, 2, 3, 5, 8)


def statics(panel, links):
    fd = pd.read_csv(REL/"corridors"/panel/"network"/"fd_parameters.csv", dtype={"link_id": str}).groupby("link_id")[["free_speed_kmh", "capacity_vph", "lanes", "length_km"]].mean().reindex(links)
    topo = pd.read_csv(REL/"corridors"/panel/"network"/"lwr_mainline_topology.csv", dtype=str).drop_duplicates("link_id").set_index("link_id").reindex(links)
    return dict(vcut=0.6 * fd.free_speed_kmh.to_numpy(), vf=fd.free_speed_kmh.to_numpy(), cap=fd.capacity_vph.to_numpy(),
                lanes=fd.lanes.to_numpy(), length=fd.length_km.to_numpy(),
                on=topo.on_ramp_link_ids.fillna("").str.len().gt(0).to_numpy().astype(np.float32),
                off=topo.off_ramp_link_ids.fillna("").str.len().gt(0).to_numpy().astype(np.float32))


def daytype(dates):
    return pd.to_datetime(dates).weekday.to_numpy()


def queue_freq(ztr, vcut):
    """P(queue | weekday-type, slot, link) from train observed layer; returns S, N per daytype (wk/we) and per-day arrays for LOO."""
    q = np.nan_to_num(ztr["speed"], nan=1e9) <= vcut; ok = np.isfinite(ztr["speed"])
    we = (daytype(ztr["dates"]) >= 5).astype(int)
    S = np.stack([(q & ok)[we == k].sum(0) for k in (0, 1)]).astype(np.float32); N = np.stack([ok[we == k].sum(0) for k in (0, 1)]).astype(np.float32)
    return (q & ok), ok, we, S, N


def ffill(a):
    idx = np.where(np.isfinite(a), np.arange(a.shape[0])[:, None], 0); np.maximum.accumulate(idx, axis=0, out=idx)
    return a[idx, np.arange(a.shape[1])[None, :]]


def shift_l(x, k):
    out = np.full_like(x, np.nan, dtype=np.float32)
    if k > 0: out[..., :-k] = x[..., k:]
    elif k < 0: out[..., -k:] = x[..., :k]
    else: out = x.astype(np.float32).copy()
    return out


def dist_to(flag, direction):
    L = flag.shape[-1]; out = np.full(flag.shape, 99.0, np.float32)
    rng = range(L - 2, -1, -1) if direction > 0 else range(1, L)
    for i in rng:
        j = i + direction; out[..., i] = np.where(flag[..., j], 1.0, np.minimum(out[..., j] + 1, 99.0))
    return out


def day_ctx(mv, vcut, s0):
    """Causal same-day context at origin s0 from the visible layer mv [288, L] (slots < s0 only)."""
    r = mv[:s0] / vcut; q = np.nan_to_num(r, nan=9) <= 1.0
    anyq = q.any(1); t = np.arange(s0)
    first_link = np.where(q.any(0), q.argmax(0), -1)
    since_first = np.where(first_link >= 0, s0 - first_link, -1).astype(np.float32)
    lastfree = np.where((~q).any(0), s0 - 1 - np.argmax((~q)[::-1], 0), -1); run = np.where(q[-1], s0 - 1 - lastfree, 0).astype(np.float32)
    cfirst = t[anyq][0] if anyq.any() else -1
    nq = q.sum(1)
    return dict(since_first_q=since_first, run_q=run, frac_q_today=q.mean(0).astype(np.float32),
                corr_age=np.full(q.shape[1], s0 - cfirst if cfirst >= 0 else -1, np.float32),
                nq_now=np.full(q.shape[1], nq[-1], np.float32), nq_peak=np.full(q.shape[1], nq.max(), np.float32),
                nq_d6=np.full(q.shape[1], nq[-1] - (nq[-7] if s0 >= 7 else 0), np.float32))


def features(mv, mq, mo, st, s0, dow, freq6, pidx):
    """mv/mq/mo: visible day arrays [288, L]. Returns DataFrame with L*6 rows (h=1..6)."""
    L = mv.shape[1]
    hv = ffill(mv[s0 - 12:s0 + 1] / st["vcut"]); hq = ffill(mq[s0 - 12:s0 + 1] / st["cap"])  # 13 slots incl. T
    f = {}
    for g in (0, 1, 2, 3, 5, 8, 12): f[f"r_lag{g}"] = hv[12 - g]
    f["r_min"] = np.nanmin(hv, 0); f["r_mean"] = np.nanmean(hv, 0); f["r_sl3"] = hv[12] - hv[9]; f["r_sl12"] = hv[12] - hv[0]
    f["qc0"] = hq[12]; f["qc3"] = hq[9]; f["qc_max"] = np.nanmax(hq, 0); f["qc_sl"] = hq[12] - hq[6]
    f["occ0"] = ffill(mo[s0 - 12:s0 + 1])[12]
    f["obs_cov"] = np.isfinite(mv[s0 - 12:s0]).mean(0)
    last = hv[12]
    for k in NB:
        f[f"nb+{k}"] = shift_l(last, k); f[f"nb-{k}"] = shift_l(last, -k); f[f"nbq+{k}"] = shift_l(hq[12], k); f[f"nbq-{k}"] = shift_l(hq[12], -k)
        f[f"nbsl+{k}"] = shift_l(hv[12] - hv[9], k); f[f"nbsl-{k}"] = shift_l(hv[12] - hv[9], -k)
    queued = np.nan_to_num(last, nan=9) <= 1.0; slow = np.nan_to_num(last, nan=9) <= 1.3
    f["d_q_dn"] = dist_to(queued, 1); f["d_q_up"] = dist_to(queued, -1); f["d_s_dn"] = dist_to(slow, 1); f["d_s_up"] = dist_to(slow, -1)
    # shockwave kinematics: where was the nearest queue 3 and 6 slots ago -> propagation velocity (links per slot)
    for lag in (3, 6):
        ql = np.nan_to_num(hv[12 - lag], nan=9) <= 1.0
        f[f"d_q_dn_l{lag}"] = dist_to(ql, 1); f[f"d_q_up_l{lag}"] = dist_to(ql, -1); f[f"qd_l{lag}"] = ql.astype(np.float32)
        f[f"nq_l{lag}"] = np.full(L, ql.sum(), np.float32)
    f["vel_dn"] = (f["d_q_dn_l3"] - f["d_q_dn"]) / 3.0; f["vel_up"] = (f["d_q_up_l3"] - f["d_q_up"]) / 3.0
    f["n_queued"] = np.full(L, queued.sum(), np.float32); f["min_r_corr"] = np.full(L, np.nanmin(last))
    f.update(day_ctx(mv, st["vcut"], s0))
    f["length"], f["lanes"], f["on"], f["off"], f["vf"] = st["length"], st["lanes"], st["on"], st["off"], st["vf"]
    f["pos"] = np.arange(L) / L; f["slot"] = np.full(L, s0); f["dow"] = np.full(L, dow); f["panel"] = np.full(L, pidx)
    base = pd.DataFrame({k: np.asarray(v, np.float32) for k, v in f.items()}); base["link"] = np.arange(L)
    parts = []
    for h in range(6):
        b = base.copy(); b["h"] = h + 1; b["freq"] = freq6[h]
        b["ext_dn"] = b.d_q_dn - (h + 1) * b.vel_dn; b["ext_up"] = b.d_q_up - (h + 1) * b.vel_up
        b["freq_nb"] = np.nanmax(np.stack([shift_l(freq6[h], k) for k in (-2, -1, 0, 1, 2)]), 0); b["freq_T"] = freq6[0] * 0 + freq6[h] - freq6[0]
        parts.append(b)
    return pd.concat(parts, ignore_index=True)


def visible(z, d, train):
    """Visible layer for day d: eligible cells of masked layer; on train reconstruct from observed layer the same way."""
    el = z["elig"][d].astype(bool)
    if train:
        # train masked layer blanks Task 1 targets; validation/private histories are unmasked -> use observed layer on eligible
        return np.where(el, z["speed"][d], np.nan), np.where(el, z["flow"][d], np.nan), np.where(el, z["occ"][d], np.nan)
    return (np.where(el, z["mspeed"][d], np.nan), np.where(el, z["mflow"][d], np.nan), np.where(el, z["mocc"][d], np.nan))


def build_train():
    out_on, out_og = [], []
    for pi, panel in enumerate(Q_PANELS):
        t0 = time.time(); z = cache.load(panel, "train"); st = statics(panel, z["links"]); vc = st["vcut"]
        qd, okd, we, S, N = queue_freq(z, vc); dows = daytype(z["dates"])
        for d in range(len(z["dates"])):
            mv, mq, mo = visible(z, d, True)
            lab_all = np.nan_to_num(z["speed"][d], nan=1e9) <= vc          # proxy of latent label, all cells
            known = np.isfinite(z["speed"][d]) & z["elig"][d].astype(bool)  # scored cells are eligible
            vis_q = np.nan_to_num(mv, nan=1e9) <= vc
            anyq = lab_all.any(1)
            def loo(s): return ((S[we[d]][s] - qd[d][s]) / np.maximum(N[we[d]][s] - okd[d][s], 1))
            # onset events: first queued slot e after >= 18 slots with no queue anywhere (latent proxy)
            for e in np.flatnonzero(anyq):
                if e < 19 or anyq[e - 18:e].any(): continue
                s0 = e - 6
                if np.isfinite(mv[s0 - 12:s0]).mean() < 0.7: continue
                X = features(mv, mq, mo, st, s0, dows[d], loo(slice(s0 + 1, s0 + 7)), pi)
                X = X[X.h == 6].copy(); X["y"] = lab_all[e].astype(np.int8); X["known"] = known[e]
                X["wid"] = f"{panel}|{z['dates'][d]}|{s0}"; X["date"] = z["dates"][d]; out_on.append(X)
            # ongoing candidates every 2 slots
            for s0 in range(13, 281, 2):
                if not vis_q[s0 - 1].any(): continue
                lab = lab_all[s0 + 1:s0 + 7]
                if not lab.any() or np.isfinite(mv[s0 - 12:s0]).mean() < 0.7: continue
                pers = np.repeat(vis_q[s0 - 1][None], 6, 0)
                if (pers & lab).sum() / max(1, (pers | lab).sum()) > 0.9: continue
                X = features(mv, mq, mo, st, s0, dows[d], loo(slice(s0 + 1, s0 + 7)), pi)
                yy = lab.reshape(-1); kk = known[s0 + 1:s0 + 7].reshape(-1); sT = X.r_lag0.to_numpy() <= 1.0
                if (sT & yy & kk).sum() / max(1, ((sT | yy) & kk).sum()) > 0.9: continue   # organizer: persistence-at-T IoU <= 0.9
                X["y"] = yy.astype(np.int8); X["known"] = kk
                X["wid"] = f"{panel}|{z['dates'][d]}|{s0}"; X["date"] = z["dates"][d]; out_og.append(X)
        print(panel, f"{time.time()-t0:.0f}s onset_rows={sum(len(x) for x in out_on)} ongoing_rows={sum(len(x) for x in out_og)}", flush=True)
        pd.concat(out_og, ignore_index=True).to_parquet(W/f"{OG}{panel}.parquet"); out_og = []
    pd.concat(out_on, ignore_index=True).to_parquet(W/f"{ON}.parquet")


META = ["y", "known", "wid", "date"]
PARAMS = dict(objective="binary", learning_rate=0.05, num_leaves=127, min_data_in_leaf=100, feature_fraction=0.8, bagging_fraction=0.8,
              bagging_freq=1, lambda_l2=1.0, num_threads=8, verbose=-1, seed=0, deterministic=True)


def pick_set(p, known=None):
    """Choose the prediction set maximizing approx expected IoU: sort by p, choose k maximizing sum_top / (k + sum_all - sum_top)."""
    p = np.asarray(p, float); m = np.ones_like(p, bool) if known is None else np.asarray(known, bool)
    order = np.argsort(-np.where(m, p, -1)); ps = p[order] * m[order]; tot = ps.sum(); cs = np.cumsum(ps); k = np.arange(1, len(p) + 1)
    e = cs / (k + tot - cs); kbest = int(np.argmax(e)) + 1
    sel = np.zeros(len(p), bool); sel[order[:kbest]] = True; return sel


def iou_by_window(df, p):
    d = df[["wid", "y", "known"]].copy(); d["p"] = p; res = {}
    for w, g in d.groupby("wid", sort=False):
        s = pick_set(g.p.to_numpy(), g.known.to_numpy()); k = g.known.to_numpy(); y = g.y.to_numpy() == 1
        inter = (s & y & k).sum(); uni = ((s | y) & k).sum(); res[w] = 1.0 if uni == 0 else inter / uni
    return pd.Series(res)


def load_cond(cond, frac_windows=1.0, seed=0):
    """Load a training table, sampling whole windows per file so the ongoing table fits in memory."""
    if cond == "onset": files = [W/f"{ON}.parquet"]
    else: files = [W/f"{OG}{p}.parquet" for p in Q_PANELS]
    import pyarrow as pa, pyarrow.parquet as pq, pyarrow.compute as pc
    parts = []
    for f in files:
        if frac_windows >= 1: parts.append(pd.read_parquet(f)); continue
        u = pd.read_parquet(f, columns=["wid"]).wid.unique(); rng = np.random.default_rng(seed)
        keep = pa.array(u[rng.random(len(u)) < frac_windows])
        pf = pq.ParquetFile(f)
        for batch in pf.iter_batches(batch_size=500_000):   # filter batch-wise: never materialize the full file
            t = pa.Table.from_batches([batch]); t = t.filter(pc.is_in(t["wid"], value_set=keep))
            if t.num_rows: parts.append(t.to_pandas())
    df = pd.concat(parts, ignore_index=True)
    if cond == "ongoing" and HARD:
        # organizer selector: persistence of the state AT T must have IoU <= 0.9 against the horizon
        s = (df.r_lag0 <= 1.0).to_numpy(); y = df.y.to_numpy() == 1; k = df.known.to_numpy()
        a = pd.DataFrame({"wid": df.wid, "i": s & y & k, "u": (s | y) & k}).groupby("wid").sum()
        ok = a.index[(a.i / a.u.clip(lower=1)) <= 0.9]
        n0 = df.wid.nunique(); df = df[df.wid.isin(ok)].reset_index(drop=True)
        print(f"hard filter: kept {df.wid.nunique()}/{n0} ongoing windows", flush=True)
    return df


HARD = False  # v2 tables are filtered at build time


def cv(cond, frac_windows=1.0, rounds=400):
    from sklearn.model_selection import GroupKFold
    df = load_cond(cond, frac_windows)
    feats = [c for c in df.columns if c not in META]
    p = np.zeros(len(df)); kn = df.known.to_numpy()
    for tr, va in GroupKFold(5).split(df, groups=df.date):
        tr = tr[kn[tr]]
        m = lgb.train(PARAMS, lgb.Dataset(df.iloc[tr][feats], df.iloc[tr].y, categorical_feature=["panel"]), rounds)
        p[va] = m.predict(df.iloc[va][feats])
    iou = iou_by_window(df, p)
    print(cond, "CV IoU %.4f  windows=%d  rows=%d" % (iou.mean(), len(iou), len(df)), flush=True)
    if cond == "ongoing":
        pers = (df.r_lag0.to_numpy() <= 1.0).astype(float)
        d = df[["wid", "y", "known"]].assign(s=pers > 0.5)
        k = d.known.to_numpy(); y = d.y.to_numpy() == 1; s = d.s.to_numpy()
        g = pd.DataFrame({"wid": d.wid, "i": s & y & k, "u": (s | y) & k}).groupby("wid").sum()
        print("persistence-at-T IoU %.4f" % (g.i / g.u.clip(lower=1)).mean(), flush=True)
        # IoU by horizon-share diagnostics
        byh = pd.DataFrame({"h": df.h, "err": np.abs(p - df.y)}).groupby("h").err.mean().round(4).to_dict(); print("abs err by h", byh)
    return df, p, iou


def fit_final(cond, frac_windows=1.0, rounds=400, seeds=(0,)):
    df = load_cond(cond, frac_windows)
    df = df[df.known]; feats = [c for c in df.columns if c not in META]
    ms = []
    for s in seeds:
        m = lgb.train({**PARAMS, "seed": s}, lgb.Dataset(df[feats], df.y, categorical_feature=["panel"]), rounds); m.save_model(str(W/f"final_{cond}_{s}.txt")); ms.append(m)
    return ms, feats


def predict(splits=("validation", "private"), seeds=(0,), reuse=False, tag=""):
    if reuse:
        models = {}
        for c in ("onset", "ongoing"):
            b = lgb.Booster(model_file=str(W/f"final_{c}_0.txt")); models[c] = ([b], b.feature_name())
    else:
        models = {c: fit_final(c, 1.0 if c == "onset" else 0.12, seeds=seeds) for c in ("onset", "ongoing")}
    out = []
    for pi, panel in enumerate(Q_PANELS):
        ztr = cache.load(panel, "train"); st = statics(panel, ztr["links"]); qd, okd, we, S, N = queue_freq(ztr, st["vcut"])
        links = list(ztr["links"])
        for split in splits:
            z = cache.load(panel, split); dates = list(z["dates"]); dows = daytype(z["dates"])
            assert list(z["links"]) == links
            wi = pd.read_csv(REL/"task2"/panel/split/"window_index.csv")
            tpl = pd.read_csv(REL/"task2"/panel/split/"sample_submission_queue.csv", dtype=str, usecols=["window_id", "timestamp", "link_id"])
            for w in wi.itertuples():
                d = dates.index(w.date); s0 = int(w.forecast_origin[11:13]) * 12 + int(w.forecast_origin[14:16]) // 5
                mv, mq, mo = visible(z, d, False)
                wtype = (dows[d] >= 5).astype(int)
                freq6 = S[wtype][s0 + 1:s0 + 7] / np.maximum(N[wtype][s0 + 1:s0 + 7], 1)
                X = features(mv, mq, mo, st, s0, dows[d], freq6, pi)
                cond = "onset" if w.condition == "queue_onset" else "ongoing"
                ms, feats = models[cond]
                p = np.mean([m.predict(X[feats]) for m in ms], 0)
                known = z["elig"][d, s0 + 1:s0 + 7].reshape(-1).astype(bool)   # quality flags survive blanking
                if cond == "onset":
                    p = np.where(X.h.to_numpy() == 6, p, 0.0)
                sel = pick_set(p, known)
                if not (sel & known).any():  # guarantee: at least one queued cell exists; pick the argmax
                    sel[np.argmax(np.where(known, p, -1))] = True
                ts = [f"{w.date}T{(s0 + h) // 12:02d}:{(s0 + h) % 12 * 5:02d}:00Z" for h in range(1, 7)]
                res = pd.DataFrame({"window_id": w.window_id, "timestamp": np.repeat(ts, len(links)),
                                    "link_id": np.tile(links, 6), "queue_pred": sel.astype(int), "p": p})
                t = tpl[tpl.window_id == w.window_id]
                res = t.merge(res, on=["window_id", "timestamp", "link_id"], how="left")
                assert res.queue_pred.notna().all(), (w.window_id, res.queue_pred.isna().sum())
                out.append(res)
            print(panel, split, "done", flush=True)
    o = pd.concat(out, ignore_index=True); o.queue_pred = o.queue_pred.astype(int)
    o.to_csv(W/f"queue_pred_full{tag}.csv", index=False); o[["window_id", "timestamp", "link_id", "queue_pred"]].to_csv(W/f"queue_submission{tag}.csv", index=False)
    print("wrote", len(o), "rows; share predicted queued:", o.queue_pred.mean().round(4))


def eval_shipped(frac_og=0.08, rounds=400):
    """IoU on the 80 organizer-selected train windows, models trained without those windows' dates."""
    ship = []
    for panel in Q_PANELS:
        wi = pd.read_csv(REL/"task2"/panel/"train"/"window_index.csv"); wi["panel"] = panel; ship.append(wi)
    ship = pd.concat(ship, ignore_index=True); hold = set(ship.date)
    models = {}
    for cond, frac in (("onset", 1.0), ("ongoing", frac_og)):
        df = load_cond(cond, frac); df = df[df.known & ~df.date.isin(hold)]; feats = [c for c in df.columns if c not in META]
        models[cond] = (lgb.train(PARAMS, lgb.Dataset(df[feats], df.y, categorical_feature=["panel"]), rounds), feats); del df
    res = []
    for pi, panel in enumerate(Q_PANELS):
        ztr = cache.load(panel, "train"); st = statics(panel, ztr["links"]); qd, okd, we, S, N = queue_freq(ztr, st["vcut"])
        dates = list(ztr["dates"]); dows = daytype(ztr["dates"])
        for w in ship[ship.panel == panel].itertuples():
            d = dates.index(w.date); s0 = int(w.forecast_origin[11:13]) * 12 + int(w.forecast_origin[14:16]) // 5
            mv, mq, mo = visible(ztr, d, True)
            # mimic the scored split: slot T comes from the masked layer (Task 1 targets blank there)
            mv[s0] = np.where(ztr["elig"][d, s0].astype(bool), ztr["mspeed"][d, s0], np.nan)
            f6 = ((S[we[d]][s0 + 1:s0 + 7] - qd[d][s0 + 1:s0 + 7]) / np.maximum(N[we[d]][s0 + 1:s0 + 7] - okd[d][s0 + 1:s0 + 7], 1))
            X = features(mv, mq, mo, st, s0, dows[d], f6, pi)
            cond = "onset" if w.condition == "queue_onset" else "ongoing"; m, feats = models[cond]
            p = m.predict(X[feats]); known = ztr["elig"][d, s0 + 1:s0 + 7].reshape(-1).astype(bool)
            if cond == "onset": p = np.where(X.h.to_numpy() == 6, p, 0.0)
            sel = pick_set(p, known)
            y = (np.nan_to_num(ztr["speed"][d, s0 + 1:s0 + 7], nan=1e9) <= st["vcut"]).reshape(-1)
            k = known & np.isfinite(ztr["speed"][d, s0 + 1:s0 + 7]).reshape(-1)
            i_, u = (sel & y & k).sum(), ((sel | y) & k).sum()
            pers = np.tile(np.nan_to_num(mv[s0 - 1], nan=1e9) <= st["vcut"], 6)
            res.append(dict(panel=panel, cond=cond, iou=1.0 if u == 0 else i_ / u, pers=((pers & y & k).sum() / max(1, ((pers | y) & k).sum()))))
    r = pd.DataFrame(res); print(r.groupby("cond")[["iou", "pers"]].mean().round(4)); print("S_queue-style mean:", r.groupby(["panel", "cond"]).iou.mean().groupby("panel").mean().mean().round(4))


def score_pred_file(path, split="train"):
    """Score a written queue prediction file (template-keyed) against train labels, exactly like the evaluator."""
    o = pd.read_csv(path, dtype={"window_id": str, "link_id": str}); rows = []
    for panel in Q_PANELS:
        z = cache.load(panel, split); st = statics(panel, z["links"]); li = {l: i for i, l in enumerate(z["links"])}; dates = list(z["dates"])
        wi = pd.read_csv(REL/"task2"/panel/split/"window_index.csv")
        for w in wi.itertuples():
            g = o[o.window_id == w.window_id]
            if g.empty: continue
            d = dates.index(w.date); t = g.timestamp.str[11:13].astype(int) * 12 + g.timestamp.str[14:16].astype(int) // 5
            l = g.link_id.map(li).to_numpy(); t = t.to_numpy()
            y = np.nan_to_num(z["speed"][d, t, l], nan=1e9) <= st["vcut"][l]
            k = z["elig"][d, t, l].astype(bool) & np.isfinite(z["speed"][d, t, l]); s = g.queue_pred.to_numpy() == 1
            u = ((s | y) & k).sum(); rows.append(dict(panel=panel, cond=w.condition, iou=1.0 if u == 0 else (s & y & k).sum() / u))
    r = pd.DataFrame(rows); print(r.groupby("cond").iou.mean().round(4)); return r


def realistic_visible(z, d, s0):
    """What a validation/private window actually shows: masked layer for the whole day (Task 1 targets blank),
    but the 12-slot history republished complete; slot T from the masked layer."""
    el = z["elig"][d].astype(bool)
    mv = np.where(el, z["mspeed"][d], np.nan); mq = np.where(el, z["mflow"][d], np.nan); mo = np.where(el, z["mocc"][d], np.nan)
    if "speed" in z:  # train: republish the history from the observed layer
        h = slice(s0 - 12, s0)
        mv[h] = np.where(el[h], z["speed"][d, h], np.nan); mq[h] = np.where(el[h], z["flow"][d, h], np.nan); mo[h] = np.where(el[h], z["occ"][d, h], np.nan)
    return mv, mq, mo


def eval_realistic():
    """Score the 80 shipped train windows with the saved final models, under both input constructions."""
    models = {c: ([lgb.Booster(model_file=str(W/f"final_{c}_0.txt"))], None) for c in ("onset", "ongoing")}
    feats = {c: models[c][0][0].feature_name() for c in models}
    res = []
    for pi, panel in enumerate(Q_PANELS):
        ztr = cache.load(panel, "train"); st = statics(panel, ztr["links"]); qd, okd, we, S, N = queue_freq(ztr, st["vcut"])
        dates = list(ztr["dates"]); dows = daytype(ztr["dates"])
        wi = pd.read_csv(REL/"task2"/panel/"train"/"window_index.csv")
        for w in wi.itertuples():
            d = dates.index(w.date); s0 = int(w.forecast_origin[11:13]) * 12 + int(w.forecast_origin[14:16]) // 5
            f6 = S[we[d]][s0 + 1:s0 + 7] / np.maximum(N[we[d]][s0 + 1:s0 + 7], 1)
            cond = "onset" if w.condition == "queue_onset" else "ongoing"; m = models[cond][0][0]
            y = (np.nan_to_num(ztr["speed"][d, s0 + 1:s0 + 7], nan=1e9) <= st["vcut"]).reshape(-1)
            known = ztr["elig"][d, s0 + 1:s0 + 7].reshape(-1).astype(bool) & np.isfinite(ztr["speed"][d, s0 + 1:s0 + 7]).reshape(-1)
            out = {}
            for mode in ("complete", "realistic"):
                mv, mq, mo = visible(ztr, d, True) if mode == "complete" else realistic_visible(ztr, d, s0)
                X = features(mv, mq, mo, st, s0, dows[d], f6, pi); p = m.predict(X[feats[cond]])
                if cond == "onset": p = np.where(X.h.to_numpy() == 6, p, 0.0)
                sel = pick_set(p, known); i_, u = (sel & y & known).sum(), ((sel | y) & known).sum()
                out[mode] = 1.0 if u == 0 else i_ / u
            res.append(dict(panel=panel, cond=cond, **out))
    r = pd.DataFrame(res); print(r.groupby("cond")[["complete", "realistic"]].mean().round(4))


IDENT = ["link", "pos", "panel", "length", "lanes", "vf", "on", "off", "slot", "dow"]
FREQ = ["freq", "freq_nb", "freq_T"]


def month_block(cond, frac, variants, rounds=400):
    """Train on Jun-Nov windows, test on Dec-Feb windows (independent-month generalization)."""
    df = load_cond(cond, frac); feats = [c for c in df.columns if c not in META]
    mon = pd.to_datetime(df.date).dt.month.to_numpy(); te = np.isin(mon, [12, 1, 2]); tr = ~te & df.known.to_numpy()
    for name, drop in variants.items():
        fs = [c for c in feats if c not in drop]
        m = lgb.train(PARAMS, lgb.Dataset(df.loc[tr, fs], df.loc[tr, "y"], categorical_feature=["panel"] if "panel" in fs else "auto"), rounds)
        p = np.zeros(len(df)); p[te] = m.predict(df.loc[te, fs])
        sub = df[te]; iou = iou_by_window(sub, p[te])
        print(f"{cond} month-blocked [{name}] IoU {iou.mean():.4f} (n={len(iou)})", flush=True)
    s = (df.loc[te, "r_lag0"] <= 1.0).to_numpy() if cond == "ongoing" else np.zeros(te.sum(), bool)
    sub = df[te][["wid", "y", "known"]]; k = sub.known.to_numpy(); y = sub.y.to_numpy() == 1
    a = pd.DataFrame({"wid": sub.wid, "i": s & y & k, "u": (s | y) & k}).groupby("wid").sum()
    print(f"{cond} month-blocked [persistence-at-T] IoU {(a.i / a.u.clip(lower=1)).where(a.u > 0, 1.0).mean():.4f}", flush=True)


FAMILY = {0: 0, 1: 0, 2: 1, 3: 1, 4: 2, 5: 2, 6: 3, 7: 3}   # panel index -> corridor family


def family_block(cond, frac, variants, rounds=400):
    """Leave-one-corridor-family-out CV: a proxy for generalizing to an independently generated scenario,
    where memorized bottleneck locations/times do not transfer."""
    df = load_cond(cond, frac); feats = [c for c in df.columns if c not in META]
    fam = df.panel.map(FAMILY).to_numpy(); kn = df.known.to_numpy()
    out = {}
    for name, drop in variants.items():
        fs = [c for c in feats if c not in drop]; p = np.zeros(len(df))
        for f in range(4):
            te = fam == f; tr = ~te & kn
            m = lgb.train(PARAMS, lgb.Dataset(df.loc[tr, fs], df.loc[tr, "y"]), rounds)
            p[te] = m.predict(df.loc[te, fs])
        out[name] = iou_by_window(df, p); print(f"{cond} family-blocked [{name}] IoU {out[name].mean():.4f} (n={len(out[name])})", flush=True)
    s = (df.r_lag0 <= 1.0).to_numpy(); k = kn; y = df.y.to_numpy() == 1
    if cond == "onset": s = np.zeros(len(df), bool)
    a = pd.DataFrame({"wid": df.wid, "i": s & y & k, "u": (s | y) & k}).groupby("wid").sum()
    print(f"{cond} [persistence-at-T] IoU {(a.i / a.u.clip(lower=1)).where(a.u > 0, 1.0).mean():.4f}", flush=True)
    return df, out


def persT_rows(split_list=("validation", "private")):
    """Ongoing windows: persistence of the state at T (masked slot T, forward-filled from history). Template-keyed rows."""
    wi = pd.concat([pd.read_csv(REL/"task2"/p/s/"window_index.csv") for p in Q_PANELS for s in split_list])
    out = []
    for p in Q_PANELS:
        ztr = cache.load(p, "train"); st = statics(p, ztr["links"]); links = list(ztr["links"])
        for split in split_list:
            z = cache.load(p, split); dates = list(z["dates"])
            for w in wi[(wi.panel == p) & (wi.split == split) & (wi.condition == "queue_ongoing")].itertuples():
                d = dates.index(w.date); s0 = int(w.forecast_origin[11:13]) * 12 + int(w.forecast_origin[14:16]) // 5
                mv, _, _ = visible(z, d, False); r = ffill(mv[s0 - 12:s0 + 1] / st["vcut"])[12]
                q = (np.nan_to_num(r, nan=9) <= 1.0).astype(int)
                ts = [f"{w.date}T{(s0 + h) // 12:02d}:{(s0 + h) % 12 * 5:02d}:00Z" for h in range(1, 7)]
                out.append(pd.DataFrame({"window_id": w.window_id, "timestamp": np.repeat(ts, len(links)), "link_id": np.tile(links, 6), "qn": np.tile(q, 6)}))
    return pd.concat(out, ignore_index=True)


def auto_build():
    """Unattended: choose onset feature set from family-blocked log, train transferable models, write two queue files."""
    import re
    log = (Path.home()/"tfb/logs/t2_family.log").read_text()
    res = {m.group(1) + "|" + m.group(2): float(m.group(3)) for m in re.finditer(r"(onset|ongoing) family-blocked \[(\w+)\] IoU ([0-9.]+)", log)}
    sets = {"full": [], "no_ident": IDENT, "no_ident_no_freq": IDENT + FREQ}
    best_on = max(("full", "no_ident", "no_ident_no_freq"), key=lambda k: res.get("onset|" + k, -1))
    print("family-blocked results:", res, "-> onset uses", best_on, "; ongoing uses no_ident_no_freq", flush=True)
    models = {}
    for cond, drop in (("onset", sets[best_on]), ("ongoing", sets["no_ident_no_freq"])):
        df = load_cond(cond, 1.0 if cond == "onset" else 0.12); df = df[df.known]
        feats = [c for c in df.columns if c not in META and c not in drop]
        m = lgb.train(PARAMS, lgb.Dataset(df[feats], df.y), 400); m.save_model(str(W/f"auto_{cond}.txt")); models[cond] = ([m], feats); del df
    out = []
    for pi, panel in enumerate(Q_PANELS):
        ztr = cache.load(panel, "train"); st = statics(panel, ztr["links"]); qd, okd, we, S, N = queue_freq(ztr, st["vcut"]); links = list(ztr["links"])
        for split in ("validation", "private"):
            z = cache.load(panel, split); dates = list(z["dates"]); dows = daytype(z["dates"])
            wi = pd.read_csv(REL/"task2"/panel/split/"window_index.csv")
            tpl = pd.read_csv(REL/"task2"/panel/split/"sample_submission_queue.csv", dtype=str, usecols=["window_id", "timestamp", "link_id"])
            for w in wi.itertuples():
                d = dates.index(w.date); s0 = int(w.forecast_origin[11:13]) * 12 + int(w.forecast_origin[14:16]) // 5
                mv, mq, mo = visible(z, d, False); wt = int(dows[d] >= 5)
                X = features(mv, mq, mo, st, s0, dows[d], S[wt][s0 + 1:s0 + 7] / np.maximum(N[wt][s0 + 1:s0 + 7], 1), pi)
                cond = "onset" if w.condition == "queue_onset" else "ongoing"; ms, feats = models[cond]
                p = ms[0].predict(X[feats]); known = z["elig"][d, s0 + 1:s0 + 7].reshape(-1).astype(bool)
                if cond == "onset": p = np.where(X.h.to_numpy() == 6, p, 0.0)
                sel = pick_set(p, known)
                if not (sel & known).any(): sel[np.argmax(np.where(known, p, -1))] = True
                ts = [f"{w.date}T{(s0 + h) // 12:02d}:{(s0 + h) % 12 * 5:02d}:00Z" for h in range(1, 7)]
                res_ = pd.DataFrame({"window_id": w.window_id, "timestamp": np.repeat(ts, len(links)), "link_id": np.tile(links, 6), "queue_pred": sel.astype(int)})
                r = tpl[tpl.window_id == w.window_id].merge(res_, on=["window_id", "timestamp", "link_id"], how="left")
                assert r.queue_pred.notna().all(); out.append(r)
        print(panel, "predicted", flush=True)
    A = pd.concat(out, ignore_index=True); A.to_csv(W/"queue_auto_A.csv", index=False)
    pt = persT_rows(); B = A.merge(pt, on=["window_id", "timestamp", "link_id"], how="left")
    B["queue_pred"] = np.where(B.qn.notna(), B.qn, B.queue_pred).astype(int); B[["window_id", "timestamp", "link_id", "queue_pred"]].to_csv(W/"queue_auto_B.csv", index=False)
    print("wrote A and B", len(A), len(B), flush=True)


def compare(cond="ongoing", frac=0.08, rounds=400, drop_sets=None, param_sets=None):
    """Paired day-grouped CV: full features vs feature subsets dropped / param variants. Mean IoU and paired diff +- se."""
    from sklearn.model_selection import GroupKFold
    df = load_cond(cond, frac); feats = [c for c in df.columns if c not in META]
    if drop_sets is None and param_sets is None:
        drop_sets = {"no_shock": [c for c in feats if c.startswith(("d_q_dn_l", "d_q_up_l", "qd_l", "nq_l", "vel_", "ext_"))]}
    variants = [("full", [], {}, rounds)] + [(n, d, {}, rounds) for n, d in (drop_sets or {}).items()] + [(n, [], pp, r) for n, (pp, r) in (param_sets or {}).items()]
    kn = df.known.to_numpy(); res = {}
    for name, drop, pp, rr in variants:
        fs = [c for c in feats if c not in drop]; p = np.zeros(len(df))
        for tr, va in GroupKFold(5).split(df, groups=df.date):
            tr = tr[kn[tr]]
            m = lgb.train({**PARAMS, **pp}, lgb.Dataset(df.iloc[tr][fs], df.iloc[tr].y, categorical_feature=["panel"]), rr)
            p[va] = m.predict(df.iloc[va][fs])
        res[name] = iou_by_window(df, p); print(name, "IoU %.4f" % res[name].mean(), flush=True)
    for name in res:
        if name == "full": continue
        d = res["full"] - res[name]; print(f"full - {name}: {d.mean():+.4f} +- {d.std() / np.sqrt(len(d)):.4f}  (n={len(d)})")


if __name__ == "__main__":
    if sys.argv[1] == "auto_build": auto_build()
    elif sys.argv[1] == "family_block":
        V = {"full": [], "no_ident": IDENT, "no_ident_no_freq": IDENT + FREQ}
        family_block("ongoing", 0.08, V); family_block("onset", 1.0, V)
    elif sys.argv[1] == "month_block":
        V = {"full": [], "no_ident": IDENT, "no_freq": FREQ, "no_ident_no_freq": IDENT + FREQ}
        month_block("onset", 1.0, V); month_block("ongoing", 0.08, V)
    elif sys.argv[1] == "eval_realistic": eval_realistic()
    elif sys.argv[1] == "predict_train_check":
        predict(splits=("train",), reuse=True, tag="_train"); score_pred_file(W/"queue_submission_train.csv")
    elif sys.argv[1] == "compare": compare()
    elif sys.argv[1] == "compare_params":
        compare(param_sets={"big_1000": (dict(num_leaves=255, learning_rate=0.03, min_data_in_leaf=50), 1000),
                            "small_leaves": (dict(num_leaves=31), 600),
                            "onset_like": (dict(min_data_in_leaf=500, lambda_l2=10.0), 400)})
    elif sys.argv[1] == "predict": predict()
    elif sys.argv[1] == "eval_shipped": eval_shipped()
    elif sys.argv[1] == "build": build_train()
    elif sys.argv[1] == "cv":
        for c in sys.argv[2:] or ("onset", "ongoing"): cv(c, frac_windows=0.06 if c == "ongoing" else 1.0)
