"""Task 1 target-month adaptation, evaluated on the validated re-mask proxy (t1_proxy.remask).

Variants (all scored on the same re-masked cells of validation and private):
  base      : final GBDT (train-fitted), train profile                         [= proxy 'gbdt']
  ownprof   : same model, but the profile features come from the TARGET month's own masked layer (weekday x slot,
              leave-one-day-out over the month; Task 1 is offline so the whole month may be used)
  selfsup   : GBDT retrained on train rows + self-supervised rows from the target month
              (re-mask another 10% of observed cells with a DIFFERENT seed; targets = those cells), own-month profile
Usage: t1_ownmonth.py
"""
import sys, numpy as np, pandas as pd, lightgbm as lgb
from pathlib import Path
sys.path.insert(0, "tfb")
import cache, t1_base as tb, score_local as sl, t1_proxy as tp
import t1_features as tf
tf.V2 = False
from t1_train import DROP, PARAMS
W = Path.home()/"tfb/work/t1"; F = Path.home()/"tfb/work/t1feat"


def own_profile(z, ch):
    """Weekday x slot mean from the month's own visible (eligible, non-blank) cells, leave-one-day-out."""
    x = np.where(z["elig"].astype(bool), z["m" + ch], np.nan)
    wd = pd.to_datetime(z["dates"]).weekday.to_numpy(); out = np.empty_like(x)
    allm = np.nanmean(x, axis=0)
    for d in range(x.shape[0]):
        same = (wd == wd[d]) & (np.arange(x.shape[0]) != d)
        m = np.nanmean(x[same], axis=0) if same.sum() else allm
        out[d] = np.where(np.isfinite(m), m, allm)
    return out


def feats_for(panel, pi, z, pv, pq):
    X, idx = tf.build(panel, z, pv, pq); X = X.drop(columns=DROP); X["panel"] = pi
    return X, idx


def score(z2, X, idx, v, q, lanes):
    PV = np.full(z2["mspeed"].shape, np.nan, np.float32); PQ = PV.copy(); d, t, l = idx[:, 0], idx[:, 1], idx[:, 2]
    PV[d, t, l] = v; PQ[d, t, l] = q
    return sl.score_t1(z2, PV, PQ, lanes)["S_state"]


def main():
    mv = lgb.Booster(model_file=str(W/"final_y_v.txt")); mq = lgb.Booster(model_file=str(W/"final_y_q.txt"))
    feats = pd.read_csv(W/"feats.csv").iloc[:, 0].tolist(); rows = []; ss_rows = []
    # ---- evaluation rows (seed 0 re-mask) + self-supervised training rows (seed 1 re-mask) per target month
    evalsets = {}
    for pi, panel in enumerate(cache.PANELS):
        ztr = cache.load(panel, "train")
        for split in ("validation", "private"):
            z = cache.load(panel, split); z2 = tp.remask(z, 0.1, seed=0); lanes = sl.lanes_t1(panel, z2["links"])
            opv, opq = own_profile(z2, "speed"), own_profile(z2, "flow")
            Xb, idx = feats_for(panel, pi, z2, tb.profile(ztr, z2, "speed"), tb.profile(ztr, z2, "flow"))
            Xo, _ = feats_for(panel, pi, z2, opv, opq)
            tv = z2["speed"][idx[:, 0], idx[:, 1], idx[:, 2]]; tq = z2["flow"][idx[:, 0], idx[:, 1], idx[:, 2]]
            evalsets[(panel, split)] = (z2, Xo, idx, lanes)
            for name, X in (("base", Xb), ("ownprof", Xo)):
                v = X.vi.to_numpy() + mv.predict(X[feats]); q = (X.qi.to_numpy() + mq.predict(X[feats])) * X.lanes.to_numpy()
                rows.append(dict(panel=panel, split=split, variant=name, S_state=score(z2, X, idx, v, q, lanes)))
            # self-supervised rows: re-mask a different 10% of the ORIGINAL month's observed cells, exclude seed-0 eval cells
            z3 = tp.remask(z, 0.1, seed=1); z3["target"] = (z3["target"].astype(bool) & ~z2["target"].astype(bool)).astype(np.int8)
            Xs, ids = feats_for(panel, pi, z3, own_profile(z3, "speed"), own_profile(z3, "flow"))
            Xs["y_v"] = z3["speed"][ids[:, 0], ids[:, 1], ids[:, 2]]; Xs["y_q"] = z3["flow"][ids[:, 0], ids[:, 1], ids[:, 2]] / Xs.lanes
            ss_rows.append(Xs.sample(frac=0.5, random_state=0))
        print(panel, "eval+ss built", flush=True)
    # ---- selfsup model: train rows (30%) + target-month rows
    rng = np.random.default_rng(0); parts = []
    for p in cache.PANELS:
        x = pd.read_parquet(F/f"{p}.parquet"); x = x[np.isfinite(x.y_v) & np.isfinite(x.y_q)]; parts.append(x.iloc[rng.choice(len(x), int(len(x) * 0.3), replace=False)])
    tr = pd.concat(parts + ss_rows, ignore_index=True); tr = tr[np.isfinite(tr.y_v) & np.isfinite(tr.y_q)]
    ms = {}
    for tgt, base in (("y_v", "vi"), ("y_q", "qi")):
        ms[tgt] = lgb.train(PARAMS, lgb.Dataset(tr[feats], (tr[tgt] - tr[base]).to_numpy(np.float32), categorical_feature=["panel"]), 800); print("trained", tgt, flush=True)
    for (panel, split), (z2, Xo, idx, lanes) in evalsets.items():
        v = Xo.vi.to_numpy() + ms["y_v"].predict(Xo[feats]); q = (Xo.qi.to_numpy() + ms["y_q"].predict(Xo[feats])) * Xo.lanes.to_numpy()
        rows.append(dict(panel=panel, split=split, variant="selfsup", S_state=score(z2, Xo, idx, v, q, lanes)))
    r = pd.DataFrame(rows); r.to_csv(W/"ownmonth_results.csv", index=False)
    fam = r.assign(fam=r.panel.str.rsplit("_", n=1).str[0]).groupby(["split", "variant", "fam"]).S_state.mean().groupby(["split", "variant"]).mean()
    print(fam.round(4).unstack(0))


if __name__ == "__main__":
    main()
