"""Task 1 final: fit residual LightGBM models on all train target cells, predict validation/private targets.

Usage: t1_predict.py fit [frac] | predict | all [frac]
Output: work/t1/state_submission.csv  (panel,timestamp,station_id,link_id,mask_regime,speed_kmh,flow_vph)
"""
import sys, time, numpy as np, pandas as pd, lightgbm as lgb
from pathlib import Path
sys.path.insert(0, "tfb")
import cache, t1_base as tb, t1_features as tf
from t1_train import PARAMS, DROP, W as FEAT

W = Path.home()/"tfb/work/t1"; W.mkdir(parents=True, exist_ok=True)
NONFEAT = ("d", "t", "l", "y_v", "y_q", "month")


def fit(frac=0.3, rounds=800):
    rng = np.random.default_rng(0); parts = []
    for p in cache.PANELS:
        x = pd.read_parquet(FEAT/f"{p}.parquet"); x = x[np.isfinite(x.y_v) & np.isfinite(x.y_q)]
        parts.append(x.iloc[rng.choice(len(x), int(len(x) * frac), replace=False)])
    df = pd.concat(parts, ignore_index=True); del parts
    feats = [c for c in df.columns if c not in NONFEAT]
    for tgt, base in (("y_v", "vi"), ("y_q", "qi")):
        t0 = time.time()
        m = lgb.train(PARAMS, lgb.Dataset(df[feats], (df[tgt] - df[base]).to_numpy(np.float32), categorical_feature=["panel"]), rounds)
        m.save_model(str(W/f"final_{tgt}.txt")); print(tgt, f"{time.time()-t0:.0f}s", flush=True)
    pd.Series(feats).to_csv(W/"feats.csv", index=False)


def predict(splits=("validation", "private")):
    mv = lgb.Booster(model_file=str(W/"final_y_v.txt")); mq = lgb.Booster(model_file=str(W/"final_y_q.txt"))
    feats = pd.read_csv(W/"feats.csv").iloc[:, 0].tolist(); out = []
    for pi, panel in enumerate(cache.PANELS):
        ztr = cache.load(panel, "train")
        for split in splits:
            z = cache.load(panel, split)
            pv, pq = tb.profile(ztr, z, "speed"), tb.profile(ztr, z, "flow")
            X, idx = tf.build(panel, z, pv, pq); X = X.drop(columns=DROP); X["panel"] = pi
            v = X.vi.to_numpy() + mv.predict(X[feats], num_threads=8)
            q = (X.qi.to_numpy() + mq.predict(X[feats], num_threads=8)) * X.lanes.to_numpy()
            dates = z["dates"][idx[:, 0]]; t = idx[:, 1]
            ts = pd.Series(dates).str.cat([pd.Series(t // 12).map("{:02d}".format), pd.Series(t % 12 * 5).map("{:02d}".format)], sep="|")
            ts = ts.str.replace(r"^(.{10})\|(\d\d)\|(\d\d)$", r"\1T\2:\3:00Z", regex=True)
            res = pd.DataFrame({"timestamp": ts.to_numpy(), "link_id": z["links"][idx[:, 2]], "speed_kmh": np.clip(v, 5.0, 140.0), "flow_vph": np.clip(q, 0.0, None)})
            tpl = pd.read_csv(cache.REL/"task1"/panel/split/"sample_submission_state.csv", usecols=["panel", "timestamp", "station_id", "link_id", "mask_regime"], dtype=str)
            m = tpl.merge(res, on=["timestamp", "link_id"], how="left")
            assert m.speed_kmh.notna().all(), (panel, split, m.speed_kmh.isna().sum())
            out.append(m); print(panel, split, len(m), flush=True)
    o = pd.concat(out, ignore_index=True); o.to_csv(W/"state_submission.csv", index=False); print("wrote", len(o))


if __name__ == "__main__":
    a = sys.argv[1]; frac = float(sys.argv[2]) if len(sys.argv) > 2 else 0.3
    if a in ("fit", "all"): fit(frac)
    if a in ("predict", "all"): predict()
