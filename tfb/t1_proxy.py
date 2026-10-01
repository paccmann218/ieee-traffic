"""Task 1 new-scenario proxy: re-mask observed eligible cells in validation/private, reconstruct, score.

Admissibility check (critic rule c): must reproduce the LB finding that the train-fitted GBDT beats time
interpolation by only ~+0.007 in 0.35*S_state+0.15*S_phys (vs +0.021 in month-blocked CV).
Here only S_state is scored (physics needs the organizer flux); the S_state gain is the comparable quantity.

Usage: t1_proxy.py [frac]
"""
import sys, numpy as np, pandas as pd, lightgbm as lgb
from pathlib import Path
sys.path.insert(0, "tfb")
import cache, t1_base as tb, score_local as sl
import t1_features as tf
tf.V2 = False                           # final models were trained on v1 features
from t1_train import DROP
W = Path.home()/"tfb/work/t1"
REGIME_RATE = {"R1": 0.20, "R2": 0.30, "R3": 0.50}


def remask(z, frac, seed=0):
    """Blank an extra `frac` of observed eligible non-target cells (link-time level) and make them the targets."""
    rng = np.random.default_rng(seed); z2 = dict(z)
    obs = z["elig"].astype(bool) & np.isfinite(z["mspeed"]) & np.isfinite(z["mflow"]) & (z["target"] == 0)
    pick = obs & (rng.random(obs.shape) < frac)
    for k in ("mspeed", "mflow", "mocc"): z2[k] = np.where(pick, np.nan, z[k])
    z2["target"] = pick.astype(np.int8); z2["speed"] = np.where(pick, z["mspeed"], np.nan); z2["flow"] = np.where(pick, z["mflow"], np.nan)
    return z2


def run(frac=0.1):
    mv = lgb.Booster(model_file=str(W/"final_y_v.txt")); mq = lgb.Booster(model_file=str(W/"final_y_q.txt"))
    feats = pd.read_csv(W/"feats.csv").iloc[:, 0].tolist(); rows = []
    for pi, panel in enumerate(cache.PANELS):
        ztr = cache.load(panel, "train")
        for split in ("validation", "private"):
            z2 = remask(cache.load(panel, split), frac); lanes = sl.lanes_t1(panel, z2["links"])
            pv, pq = tb.profile(ztr, z2, "speed"), tb.profile(ztr, z2, "flow")
            X, idx = tf.build(panel, z2, pv, pq); X = X.drop(columns=DROP); X["panel"] = pi
            PV = np.full(z2["mspeed"].shape, np.nan, np.float32); PQ = PV.copy(); IV = PV.copy(); IQ = PV.copy()
            d, t, l = idx[:, 0], idx[:, 1], idx[:, 2]
            PV[d, t, l] = X.vi.to_numpy() + mv.predict(X[feats]); PQ[d, t, l] = (X.qi.to_numpy() + mq.predict(X[feats])) * X.lanes.to_numpy()
            IV[d, t, l] = X.vi.to_numpy(); IQ[d, t, l] = X.qi.to_numpy() * X.lanes.to_numpy()
            PR = np.full_like(PV, np.nan); QR = PR.copy(); PR[d, t, l] = pv[d, t, l]; QR[d, t, l] = pq[d, t, l]
            for name, a, b in (("gbdt", PV, PQ), ("tinterp", IV, IQ), ("profile", PR, QR)):
                s = sl.score_t1(z2, a, b, lanes)
                rows.append(dict(panel=panel, split=split, model=name, S_state=s["S_state"],
                                 rmse_v=np.nanmean([s[r].get("rmse_v", np.nan) for r in sl.REGIMES]), rmse_q=np.nanmean([s[r].get("rmse_q", np.nan) for r in sl.REGIMES])))
        print(panel, flush=True)
    r = pd.DataFrame(rows); r.to_csv(W/"proxy_t1.csv", index=False)
    fam = r.assign(fam=r.panel.str.rsplit("_", n=1).str[0]).groupby(["split", "model", "fam"]).S_state.mean().groupby(["split", "model"]).mean()
    print(fam.round(4).unstack(0)); print(r.groupby(["split", "model"])[["rmse_v", "rmse_q"]].mean().round(3))


if __name__ == "__main__":
    run(float(sys.argv[1]) if len(sys.argv) > 1 else 0.1)
