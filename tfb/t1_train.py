"""Task 1 GBDT residual model: build features for all train target cells, month-blocked CV, local scoring.

Usage: t1_train.py build            -> work/t1feat/<panel>.parquet (features + d,t,l + truth)
       t1_train.py cv [frac]        -> train on Jun-Nov, eval Dec-Feb; prints S_state / S_physics vs tinterp
"""
import sys, time, numpy as np, pandas as pd, lightgbm as lgb
from pathlib import Path
sys.path.insert(0, "tfb")
import cache, t1_base as tb, t1_features as tf, score_local as sl

import os
W = Path.home()/("tfb/work/" + os.environ.get("T1FEAT", "t1feat")); W.mkdir(parents=True, exist_ok=True)
DROP = [] if os.environ.get("T1_KEEP_INELIG", "0") == "1" else ["iv", "iq"]


def build_panel(panel):
    z = cache.load(panel, "train"); t0 = time.time()
    pv, pq = tb.profile(z, ch="speed"), tb.profile(z, ch="flow")
    X, idx = tf.build(panel, z, pv, pq)
    X = X.drop(columns=DROP)
    X["d"], X["t"], X["l"] = idx[:, 0], idx[:, 1], idx[:, 2]
    X["y_v"] = z["speed"][idx[:, 0], idx[:, 1], idx[:, 2]]; X["y_q"] = z["flow"][idx[:, 0], idx[:, 1], idx[:, 2]] / X.lanes
    X["month"] = pd.to_datetime(z["dates"][idx[:, 0]]).month
    X["panel"] = cache.PANELS.index(panel)
    X.to_parquet(W/f"{panel}.parquet"); print(panel, X.shape, f"{time.time()-t0:.0f}s", flush=True)


PARAMS = dict(objective="regression", learning_rate=0.05, num_leaves=255, min_data_in_leaf=200, feature_fraction=0.8,
              bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, num_threads=8, verbose=-1, seed=0, deterministic=True)


def cv(frac=0.3, rounds=800):
    parts = [pd.read_parquet(W/f"{p}.parquet") for p in cache.PANELS]
    df = pd.concat(parts, ignore_index=True); del parts
    df = df[np.isfinite(df.y_v) & np.isfinite(df.y_q)]
    feats = [c for c in df.columns if c not in ("d", "t", "l", "y_v", "y_q", "month")]
    test = df.month.isin([12, 1, 2]).to_numpy(); tr = np.flatnonzero(~test)
    rng = np.random.default_rng(0); tr = rng.choice(tr, int(len(tr) * frac), replace=False)
    preds = {}
    for tgt, base in (("y_v", "vi"), ("y_q", "qi")):
        y = (df[tgt] - df[base]).to_numpy(np.float32)
        t0 = time.time()
        dtr = lgb.Dataset(df.iloc[tr][feats], y[tr], categorical_feature=["panel"], free_raw_data=True)
        m = lgb.train(PARAMS, dtr, rounds)
        preds[tgt] = df.loc[test, base].to_numpy() + m.predict(df.loc[test, feats], num_threads=8)
        imp = pd.Series(m.feature_importance("gain"), feats).sort_values(ascending=False)
        print(tgt, f"trained {time.time()-t0:.0f}s", "top:", imp.head(12).round(0).to_dict(), flush=True)
        m.save_model(str(W/f"cv_{tgt}.txt"))
    te = df.loc[test, ["panel", "d", "t", "l", "lanes"]].copy(); te["pv"] = preds["y_v"]; te["pq"] = preds["y_q"] * te.lanes
    te.to_parquet(W/"cv_preds.parquet")
    # local scoring per panel on test months only (other days: predictions = truth so they do not count in T1; T3 uses truth)
    rows = []
    for pi, panel in enumerate(cache.PANELS):
        z = cache.load(panel, "train"); lanes = sl.lanes_t1(panel, z["links"]); t3 = sl.T3(panel, z)
        mon = pd.to_datetime(z["dates"]).month.to_numpy(); keep = np.isin(mon, [12, 1, 2])
        zz = {k: (v[keep] if isinstance(v, np.ndarray) and v.ndim >= 1 and v.shape[0] == len(mon) else v) for k, v in z.items()}
        sub = te[te.panel == pi]; dmap = {d: i for i, d in enumerate(np.flatnonzero(keep))}
        for name in ("tinterp", "gbdt"):
            if name == "tinterp": PV, PQ = tb.time_interp(zz, "speed"), tb.time_interp(zz, "flow")
            else:
                PV = np.full(zz["speed"].shape, np.nan, np.float32); PQ = PV.copy()
                dd = sub.d.map(dmap).to_numpy(); PV[dd, sub.t, sub.l] = sub.pv; PQ[dd, sub.t, sub.l] = sub.pq
            s1 = sl.score_t1(zz, PV, PQ, lanes); s3 = t3_sub(t3, zz, keep).score(zz, PV, PQ)
            rows.append(dict(panel=panel, model=name, S_state=s1["S_state"], S_phys=s3["S_physics"],
                             rmse_v=np.mean([s1[r]["rmse_v"] for r in sl.REGIMES]), rmse_q=np.mean([s1[r]["rmse_q"] for r in sl.REGIMES]),
                             LWR=np.mean([s3[r]["S_LWR"] for r in sl.REGIMES])))
    r = pd.DataFrame(rows); pd.set_option("display.width", 200); print(r.round(4).to_string())
    fam = r.assign(fam=r.panel.str.rsplit("_", n=1).str[0]).groupby(["model", "fam"])[["S_state", "S_phys"]].mean().groupby("model").mean()
    print(fam.round(4)); print("J=0.35*state+0.15*phys:", (0.35 * fam.S_state + 0.15 * fam.S_phys).round(4).to_dict())


def t3_sub(t3, zz, keep):
    import copy
    t = copy.copy(t3); t.ramp_ok = t3.ramp_ok[keep]; return t


if __name__ == "__main__":
    if sys.argv[1] == "build":
        for p in (sys.argv[2:] or cache.PANELS): build_panel(p)
    else:
        cv(float(sys.argv[2]) if len(sys.argv) > 2 else 0.3)
