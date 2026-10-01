"""Task 1 production model: train rows + self-supervised target-month rows, own-month profile features.

Validated on the re-mask proxy (t1_ownmonth.py): selfsup 0.9368/0.9358 vs base 0.9337/0.9323 (validation/private).
Self-supervised rows: in each validation/private month, re-mask 10% of observed eligible cells (two seeds),
targets = those cells, features built with that month's own profile. Task 1 is offline, so the whole month is usable.
Output: work/t1/state_submission_ss.csv (template-keyed, speeds clipped to [5, 140]).
"""
import sys, numpy as np, pandas as pd, lightgbm as lgb
from pathlib import Path
sys.path.insert(0, "tfb")
import cache, t1_proxy as tp
import t1_features as tf
tf.V2 = False
from t1_train import DROP, PARAMS
from t1_ownmonth import own_profile
W = Path.home()/"tfb/work/t1"; F = Path.home()/"tfb/work/t1feat"


def main():
    feats = pd.read_csv(W/"feats.csv").iloc[:, 0].tolist(); rng = np.random.default_rng(0); parts = []
    for p in cache.PANELS:
        x = pd.read_parquet(F/f"{p}.parquet"); x = x[np.isfinite(x.y_v) & np.isfinite(x.y_q)]
        parts.append(x.iloc[rng.choice(len(x), int(len(x) * 0.3), replace=False)])
    for pi, panel in enumerate(cache.PANELS):
        for split in ("validation", "private"):
            z = cache.load(panel, split)
            for seed in (1, 2):
                z3 = tp.remask(z, 0.1, seed=seed)
                X, idx = tf.build(panel, z3, own_profile(z3, "speed"), own_profile(z3, "flow")); X = X.drop(columns=DROP); X["panel"] = pi
                X["y_v"] = z3["speed"][idx[:, 0], idx[:, 1], idx[:, 2]]; X["y_q"] = z3["flow"][idx[:, 0], idx[:, 1], idx[:, 2]] / X.lanes
                parts.append(X.sample(frac=0.5, random_state=seed))
        print(panel, "ss rows", flush=True)
    tr = pd.concat(parts, ignore_index=True); tr = tr[np.isfinite(tr.y_v) & np.isfinite(tr.y_q)]; del parts
    ms = {}
    for tgt, base in (("y_v", "vi"), ("y_q", "qi")):
        ms[tgt] = lgb.train(PARAMS, lgb.Dataset(tr[feats], (tr[tgt] - tr[base]).to_numpy(np.float32), categorical_feature=["panel"]), 800)
        ms[tgt].save_model(str(W/f"final_ss_{tgt}.txt")); print("trained", tgt, flush=True)
    del tr; out = []
    for pi, panel in enumerate(cache.PANELS):
        for split in ("validation", "private"):
            z = cache.load(panel, split)
            X, idx = tf.build(panel, z, own_profile(z, "speed"), own_profile(z, "flow")); X = X.drop(columns=DROP); X["panel"] = pi
            v = np.clip(X.vi.to_numpy() + ms["y_v"].predict(X[feats]), 5.0, 140.0)
            q = np.clip((X.qi.to_numpy() + ms["y_q"].predict(X[feats])) * X.lanes.to_numpy(), 0.0, None)
            t = idx[:, 1]; dates = z["dates"][idx[:, 0]]
            ts = [f"{a}T{b // 12:02d}:{b % 12 * 5:02d}:00Z" for a, b in zip(dates, t)]
            res = pd.DataFrame({"timestamp": ts, "link_id": z["links"][idx[:, 2]], "speed_kmh": v, "flow_vph": q})
            tpl = pd.read_csv(cache.REL/"task1"/panel/split/"sample_submission_state.csv", usecols=["panel", "timestamp", "station_id", "link_id", "mask_regime"], dtype=str)
            m = tpl.merge(res, on=["timestamp", "link_id"], how="left"); assert m.speed_kmh.notna().all(), (panel, split)
            out.append(m); print(panel, split, len(m), flush=True)
    pd.concat(out, ignore_index=True).to_csv(W/"state_submission_ss.csv", index=False); print("wrote state_submission_ss.csv", flush=True)


if __name__ == "__main__":
    main()
