"""Leak-free month-blocked Task 1 CV with profiles rebuilt inside the fold.

The older feature tables computed weekday profiles from all train months, so
December-February influenced their own profile features.  This script rebuilds
both train and test features:

  train: June-November, leave-one-day-out profile within June-November
  test:  December-February, profile from June-November only

The actual published target masks and equal regime/family aggregation are used.
"""
from __future__ import annotations

import copy
import os
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
import t1_features as tf
import t1_train as old


tf.V2 = False
OUT = Path.home() / "tfb/work/t1_faithful"
KEEP_INELIGIBLE = os.environ.get("T1_KEEP_INELIG", "0") == "1"
DROP = [] if KEEP_INELIGIBLE else ["iv", "iq"]


def subset(z: dict, keep: np.ndarray) -> dict:
    days = len(z["dates"])
    return {
        key: (
            value[keep]
            if isinstance(value, np.ndarray)
            and value.ndim >= 1
            and value.shape[0] == days
            and key not in ("links", "ramp_ids", "ramp_type")
            else value
        )
        for key, value in z.items()
    }


def build_panel(panel: str) -> None:
    z = cache.load(panel, "train")
    month = pd.to_datetime(z["dates"]).month.to_numpy()
    is_test = np.isin(month, [12, 1, 2])
    train, test = subset(z, ~is_test), subset(z, is_test)
    for split, target in (("train", train), ("test", test)):
        if split == "train":
            pv = tb.profile(train, ch="speed")
            pq = tb.profile(train, ch="flow")
        else:
            pv = tb.profile(train, test, "speed")
            pq = tb.profile(train, test, "flow")
        started = time.time()
        frame, idx = tf.build(panel, target, pv, pq)
        frame = frame.drop(columns=DROP)
        frame["d"], frame["t"], frame["l"] = idx[:, 0], idx[:, 1], idx[:, 2]
        frame["y_v"] = target["speed"][idx[:, 0], idx[:, 1], idx[:, 2]]
        frame["y_q"] = target["flow"][idx[:, 0], idx[:, 1], idx[:, 2]] / frame.lanes
        frame["panel"] = cache.PANELS.index(panel)
        frame.to_parquet(OUT / f"{panel}_{split}.parquet")
        print(panel, split, frame.shape, f"{time.time() - started:.0f}s", flush=True)


def train(frac: float = 0.3, balanced: bool = False) -> pd.DataFrame:
    tr = pd.concat([pd.read_parquet(OUT / f"{p}_train.parquet") for p in cache.PANELS], ignore_index=True)
    te = pd.concat([pd.read_parquet(OUT / f"{p}_test.parquet") for p in cache.PANELS], ignore_index=True)
    tr = tr[np.isfinite(tr.y_v) & np.isfinite(tr.y_q)].reset_index(drop=True)
    te = te[np.isfinite(te.y_v) & np.isfinite(te.y_q)].reset_index(drop=True)
    features = [c for c in tr if c not in ("d", "t", "l", "y_v", "y_q")]
    idx = np.arange(len(tr))
    if frac < 1:
        idx = np.random.default_rng(0).choice(idx, int(len(idx) * frac), replace=False)
    weight = None
    if balanced:
        regime = tr.iloc[idx].regime.to_numpy(int)
        counts = np.bincount(regime, minlength=3)
        weight = len(regime) / (3 * np.maximum(counts[regime], 1))
    predictions = {}
    tag = f"f{frac:g}_{'bal' if balanced else 'plain'}_{'inel' if KEEP_INELIGIBLE else 'elig'}"
    for target, interpolation in (("y_v", "vi"), ("y_q", "qi")):
        residual = (tr[target] - tr[interpolation]).to_numpy(np.float32)
        data = lgb.Dataset(
            tr.iloc[idx][features], residual[idx], weight=weight,
            categorical_feature=["panel"], free_raw_data=True,
        )
        model = lgb.train(old.PARAMS, data, 800)
        predictions[target] = te[interpolation].to_numpy() + model.predict(te[features], num_threads=8)
        model.save_model(str(OUT / f"{tag}_{target}.txt"))
    pred = te[["panel", "d", "t", "l", "lanes"]].copy()
    pred["pv"] = predictions["y_v"]
    pred["pq"] = predictions["y_q"] * pred.lanes
    pred.to_parquet(OUT / f"{tag}_predictions.parquet")
    return pred


def evaluate(pred: pd.DataFrame, tag: str) -> pd.DataFrame:
    rows = []
    for pi, panel in enumerate(cache.PANELS):
        z = cache.load(panel, "train")
        month = pd.to_datetime(z["dates"]).month.to_numpy()
        keep = np.isin(month, [12, 1, 2])
        test = subset(z, keep)
        sub = pred[pred.panel == pi]
        speed = np.full(test["speed"].shape, np.nan, np.float32)
        flow = np.full_like(speed, np.nan)
        dd = sub.d.to_numpy(int)
        tt = sub.t.to_numpy(int)
        ll = sub.l.to_numpy(int)
        speed[dd, tt, ll] = sub.pv.to_numpy()
        flow[dd, tt, ll] = sub.pq.to_numpy()
        state = sl.score_t1(test, speed, flow, sl.lanes_t1(panel, z["links"]))
        t3 = copy.copy(sl.T3(panel, z))
        t3.ramp_ok = t3.ramp_ok[keep]
        physics = t3.score(test, speed, flow)
        for regime in sl.REGIMES:
            rows.append({
                "tag": tag, "panel": panel, "family": panel.rsplit("_", 1)[0],
                "regime": regime, "S_state": state[regime]["S"],
                "S_physics_local": physics[regime]["S_phys"],
                "S_LWR_local": physics[regime]["S_LWR"],
                "rmse_v": state[regime]["rmse_v"],
                "rmse_q_lane": state[regime]["rmse_q"],
            })
    detail = pd.DataFrame(rows)
    detail.to_csv(OUT / f"{tag}_detail.csv", index=False)
    macro = detail.groupby(["tag", "regime", "family"])[
        ["S_state", "S_physics_local", "S_LWR_local"]
    ].mean().groupby(["tag", "regime"]).mean()
    overall = macro.groupby("tag").mean()
    overall["J_local"] = 0.35 * overall.S_state + 0.15 * overall.S_physics_local
    print(macro.round(4).to_string())
    print(overall.round(4).to_string())
    return detail


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    mode = sys.argv[1] if len(sys.argv) > 1 else "build"
    if mode == "build":
        for panel in (sys.argv[2:] or cache.PANELS):
            build_panel(panel)
        return
    frac = float(sys.argv[2]) if len(sys.argv) > 2 else 0.3
    balanced = len(sys.argv) > 3 and sys.argv[3] == "balanced"
    tag = f"f{frac:g}_{'bal' if balanced else 'plain'}_{'inel' if KEEP_INELIGIBLE else 'elig'}"
    evaluate(train(frac, balanced), tag)


if __name__ == "__main__":
    main()
