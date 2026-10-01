"""Controlled GBDT ablations on the existing Task 1 feature tables.

Variants change one training decision while preserving the Dec-Feb holdout and
the exact prediction/scoring path used by ``t1_train.py``:

  full             all June-Nov rows instead of a 30% sample
  regime_balanced  30% sample with inverse-frequency R1/R2/R3 weights
  full_balanced    all rows with inverse-frequency regime weights
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

sys.path.insert(0, "tfb")
import cache
import score_local as sl
import t1_base as tb
import t1_train as base


OUT = Path.home() / "tfb/work/t1_gbdt_ablate"
VARIANTS = {
    "full": (1.0, False),
    "regime_balanced": (0.3, True),
    "full_balanced": (1.0, True),
}


def load_features() -> pd.DataFrame:
    parts = [pd.read_parquet(base.W / f"{panel}.parquet") for panel in cache.PANELS]
    frame = pd.concat(parts, ignore_index=True)
    return frame[np.isfinite(frame.y_v) & np.isfinite(frame.y_q)].reset_index(drop=True)


def train_predict(frame: pd.DataFrame, variant: str) -> pd.DataFrame:
    frac, balanced = VARIANTS[variant]
    features = [c for c in frame.columns if c not in ("d", "t", "l", "y_v", "y_q", "month")]
    test = frame.month.isin([12, 1, 2]).to_numpy()
    train_idx = np.flatnonzero(~test)
    if frac < 1:
        train_idx = np.random.default_rng(0).choice(
            train_idx, int(len(train_idx) * frac), replace=False
        )
    weight = None
    if balanced:
        regime = frame.iloc[train_idx].regime.to_numpy()
        counts = np.bincount(regime.astype(int), minlength=3)
        weight = len(regime) / (3 * np.maximum(counts[regime.astype(int)], 1))
    predictions = {}
    OUT.mkdir(parents=True, exist_ok=True)
    for target, interpolation in (("y_v", "vi"), ("y_q", "qi")):
        residual = (frame[target] - frame[interpolation]).to_numpy(np.float32)
        dataset = lgb.Dataset(
            frame.iloc[train_idx][features],
            residual[train_idx],
            weight=weight,
            categorical_feature=["panel"],
            free_raw_data=True,
        )
        started = time.time()
        model = lgb.train(base.PARAMS, dataset, 800)
        predictions[target] = (
            frame.loc[test, interpolation].to_numpy()
            + model.predict(frame.loc[test, features], num_threads=8)
        )
        model.save_model(str(OUT / f"{variant}_{target}.txt"))
        print(target, variant, f"{time.time() - started:.0f}s", flush=True)
    result = frame.loc[test, ["panel", "d", "t", "l", "lanes"]].copy()
    result["pv"] = predictions["y_v"]
    result["pq"] = predictions["y_q"] * result.lanes
    result.to_parquet(OUT / f"{variant}_predictions.parquet")
    return result


def evaluate(predictions: pd.DataFrame, variant: str) -> pd.DataFrame:
    rows = []
    for pi, panel in enumerate(cache.PANELS):
        z = cache.load(panel, "train")
        month = pd.to_datetime(z["dates"]).month.to_numpy()
        keep = np.isin(month, [12, 1, 2])
        zz = {
            key: (
                value[keep]
                if isinstance(value, np.ndarray)
                and value.ndim >= 1
                and value.shape[0] == len(month)
                else value
            )
            for key, value in z.items()
        }
        sub = predictions[predictions.panel == pi]
        dmap = {day: i for i, day in enumerate(np.flatnonzero(keep))}
        dd = sub.d.map(dmap).to_numpy()
        speed = np.full(zz["speed"].shape, np.nan, np.float32)
        flow = np.full_like(speed, np.nan)
        speed[dd, sub.t, sub.l] = sub.pv
        flow[dd, sub.t, sub.l] = sub.pq
        state = sl.score_t1(zz, speed, flow, sl.lanes_t1(panel, z["links"]))
        physics = base.t3_sub(sl.T3(panel, z), zz, keep).score(zz, speed, flow)
        for regime in sl.REGIMES:
            rows.append({
                "variant": variant,
                "panel": panel,
                "family": panel.rsplit("_", 1)[0],
                "regime": regime,
                "S_state": state[regime]["S"],
                "S_physics_local": physics[regime]["S_phys"],
                "S_LWR_local": physics[regime]["S_LWR"],
                "rmse_v": state[regime]["rmse_v"],
                "rmse_q_lane": state[regime]["rmse_q"],
            })
    detail = pd.DataFrame(rows)
    detail.to_csv(OUT / f"{variant}_detail.csv", index=False)
    macro = detail.groupby(["variant", "regime", "family"])[
        ["S_state", "S_physics_local", "S_LWR_local"]
    ].mean().groupby(["variant", "regime"]).mean()
    overall = macro.groupby("variant").mean()
    overall["J_local"] = 0.35 * overall.S_state + 0.15 * overall.S_physics_local
    print(macro.round(4).to_string())
    print(overall.round(4).to_string())
    return detail


def main() -> None:
    variant = sys.argv[1] if len(sys.argv) > 1 else "full"
    if variant not in VARIANTS:
        raise SystemExit(f"variant must be one of {sorted(VARIANTS)}")
    frame = load_features()
    evaluate(train_predict(frame, variant), variant)


if __name__ == "__main__":
    main()
