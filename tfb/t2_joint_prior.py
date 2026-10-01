"""Joint-mask onset prior with leave-one-month-out evaluation.

Unlike a per-cell classifier or fixed top-k decoder, this model learns the
distribution of complete onset masks from train days satisfying the same legal
event definition:

  no visible queue in [T-60,T), no queue in T+5..T+25, queue at T+30.

The event is known from ``window_index.condition`` at inference.  Candidate
outcomes come only from the released train split.  The Bayes action maximizes
empirical expected IoU over complete candidate masks, preserving their joint
size and shape.

``evaluate`` is leave-one-month-out.  ``build_rows`` fits all train months and
emits validation/private template keys.  No validation/private label is used.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.ndimage import median_filter

sys.path.insert(0, "tfb")
import cache
import t2_persist as tp
from t2 import daytype, statics


REL = cache.REL
OUT = Path.home() / "tfb/work/t2_joint_prior"


@dataclass
class Event:
    month: str
    weekend: int
    slot: int
    mask_noisy: np.ndarray
    mask_latent: np.ndarray
    known: np.ndarray
    features: np.ndarray


def iou_1d(a: np.ndarray, b: np.ndarray, known: np.ndarray | None = None) -> float:
    k = np.ones_like(a, bool) if known is None else known
    u = ((a | b) & k).sum()
    return 1.0 if u == 0 else float((a & b & k).sum() / u)


def history_features(
    z: dict,
    d: int,
    s0: int,
    vcut: np.ndarray,
    cap: np.ndarray,
    version: str = "v1",
) -> np.ndarray:
    el = z["elig"][d, s0 - 12:s0].astype(bool)
    raw_v = z["speed"][d, s0 - 12:s0]
    raw_q = z["flow"][d, s0 - 12:s0]
    if version == "v2":
        # Same legal hour as U2: last released value, including ineligible cells.
        speed = np.where(np.isfinite(raw_v), raw_v, np.nan)
        flow = np.where(np.isfinite(raw_q), raw_q, np.nan)
    else:
        speed = np.where(el, raw_v, np.nan)
        flow = np.where(el, raw_q, np.nan)
    sv = pd.DataFrame(speed).ffill().to_numpy()
    qv = pd.DataFrame(flow).ffill().to_numpy()
    ratio = sv / vcut[None, :]
    cap_sum = max(float(np.nansum(cap)), 1.0)
    total_q = np.nansum(qv, axis=1) / cap_sum
    mean_r = np.nanmean(ratio, axis=1)
    rf = np.where(z["relig"][d, s0 - 12:s0].astype(bool), z["rflow"][d, s0 - 12:s0], np.nan)
    ramp = np.nansum(rf, axis=1) / cap_sum if rf.shape[1] else np.zeros(12)
    base = [
        total_q[-1], np.nanmean(total_q), total_q[-1] - total_q[0],
        mean_r[-1], np.nanmean(mean_r), mean_r[-1] - mean_r[0],
        np.nanmin(ratio), el.mean(),
        ramp[-1], np.nanmean(ramp), ramp[-1] - ramp[0],
        np.sin(2 * np.pi * s0 / 288), np.cos(2 * np.pi * s0 / 288),
    ]
    if version == "v2":
        last = ratio[-1]
        finite = last[np.isfinite(last)]
        hot = float(np.nanmean(last < 1.35)) if finite.size else 0.0
        slow = np.sort(np.nan_to_num(last, nan=9.0))[:5]
        base.extend([hot, float(np.mean(slow)), float(np.nanmin(last) if finite.size else 9.0)])
    return np.asarray(base, dtype=float)


def is_onset_event(z: dict, d: int, s0: int, vcut: np.ndarray) -> bool:
    if s0 < 12 or s0 + 6 >= 288:
        return False
    el = z["elig"][d].astype(bool)
    speed = z["speed"][d]
    q = (np.nan_to_num(speed, nan=1e9) <= vcut) & el
    if el[s0 - 12:s0].mean() < 0.7:
        return False
    return not q[s0 - 12:s0 + 6].any() and q[s0 + 6].any()


def make_event(
    z: dict,
    d: int,
    s0: int,
    vcut: np.ndarray,
    cap: np.ndarray,
    weekend: np.ndarray,
    version: str = "v1",
) -> Event:
    el = z["elig"][d, s0 + 6].astype(bool)
    speed = z["speed"][d]
    noisy = (np.nan_to_num(speed[s0 + 6], nan=1e9) <= vcut) & el
    sm = median_filter(np.nan_to_num(speed, nan=1e9), size=(3, 1), mode="nearest")
    latent = tp.fill((sm[s0 + 6] <= vcut) & el)
    return Event(
        month=str(z["dates"][d])[:7],
        weekend=int(weekend[d]),
        slot=s0,
        mask_noisy=noisy,
        mask_latent=latent,
        known=el & np.isfinite(speed[s0 + 6]),
        features=history_features(z, d, s0, vcut, cap, version=version),
    )


def event_pool(
    z: dict,
    slots: set[int],
    vcut: np.ndarray,
    cap: np.ndarray,
    radius: int = 0,
    version: str = "v1",
) -> list[Event]:
    weekend = (np.asarray(daytype(z["dates"])) >= 5).astype(int)
    wanted = sorted({s + off for s in slots for off in range(-radius, radius + 1)})
    out = []
    for d in range(len(z["dates"])):
        for s0 in wanted:
            if is_onset_event(z, d, s0, vcut):
                out.append(make_event(z, d, s0, vcut, cap, weekend, version=version))
    return out


def standard_dist(query: np.ndarray, x: np.ndarray) -> np.ndarray:
    scale = np.nanstd(x, axis=0)
    scale = np.where(np.isfinite(scale) & (scale > 1e-6), scale, 1.0)
    z = (x - query) / scale
    return np.sqrt(np.nanmean(np.where(np.isfinite(z), z * z, 4.0), axis=1))


def unique_actions(
    outcomes: np.ndarray,
    marginal: np.ndarray,
    observed_only: bool = False,
) -> list[np.ndarray]:
    actions = [o.copy() for o in outcomes]
    if not observed_only:
        order = np.argsort(-marginal)
        actions.extend(np.isin(np.arange(len(marginal)), order[:k]) for k in range(1, min(15, len(order)) + 1))
    uniq: dict[bytes, np.ndarray] = {}
    for a in actions:
        uniq[np.packbits(a).tobytes()] = a
    return list(uniq.values())


def bayes_action(
    events: list[Event],
    label: str,
    query: np.ndarray | None = None,
    neighbours: int | None = None,
    observed_only: bool = False,
    size_gate: bool = False,
) -> np.ndarray:
    events = [e for e in events if e.known.any()]
    if not events:
        raise ValueError("No onset analogues satisfy the training-only event filter")
    if query is not None and neighbours is not None and len(events) > neighbours:
        x = np.stack([e.features for e in events])
        keep = np.argsort(standard_dist(query, x))[:neighbours]
        events = [events[i] for i in keep]
    outcomes = np.stack([getattr(e, f"mask_{label}") for e in events])
    known = np.stack([e.known for e in events])
    marginal = (outcomes & known).sum(0) / np.maximum(known.sum(0), 1)
    actions = unique_actions(outcomes, marginal, observed_only=observed_only)
    if size_gate:
        sizes = (outcomes & known).sum(1)
        pred_k = float(np.mean(sizes))
        gated = [a for a in actions if abs(int(a.sum()) - pred_k) <= 1.5]
        if gated:
            actions = gated
    expected = [np.mean([iou_1d(a, y, k) for y, k in zip(outcomes, known)]) for a in actions]
    return actions[int(np.argmax(expected))]


def select_candidates(pool: list[Event], weekend: int, slot: int, radius: int) -> tuple[list[Event], str]:
    """Deterministic sparse-support fallback, fixed before leaderboard scoring."""
    tiers = [
        ("same_daytype_local", [e for e in pool if e.weekend == weekend and abs(e.slot - slot) <= radius]),
        ("all_daytypes_local", [e for e in pool if abs(e.slot - slot) <= radius]),
        ("same_daytype_any_slot", [e for e in pool if e.weekend == weekend]),
        ("all_events", list(pool)),
    ]
    for name, events in tiers:
        if events:
            return events, name
    return [], "empty"


def query_event(z: dict, d: int, s0: int, vcut: np.ndarray, cap: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    speed = z["speed"][d]
    known = z["elig"][d, s0 + 1:s0 + 7].astype(bool) & np.isfinite(speed[s0 + 1:s0 + 7])
    noisy = (np.nan_to_num(speed[s0 + 1:s0 + 7], nan=1e9) <= vcut) & known
    sm = median_filter(np.nan_to_num(speed, nan=1e9), size=(3, 1), mode="nearest")
    latent = tp.fill((sm[s0 + 1:s0 + 7] <= vcut) & known)
    return history_features(z, d, s0, vcut, cap), noisy, latent, known


def parse_method(method: str) -> dict:
    return {
        "radius": 6 if method.startswith("r6_") else 0,
        "neighbours": 10 if ("analog10" in method or "size10" in method) else None,
        "version": "v2" if "_v2" in method else "v1",
        "label": "latent" if method.endswith("_latent") else "noisy",
        "observed_only": "obs" in method,
        "size_gate": "size" in method,
    }


def score_h6(mask: np.ndarray, truth: np.ndarray, known: np.ndarray) -> float:
    pred = np.zeros_like(truth)
    pred[5] = mask
    u = ((pred | truth) & known).sum()
    return 1.0 if u == 0 else float((pred & truth & known).sum() / u)


def evaluate() -> pd.DataFrame:
    rows = []
    for panel in tp.Q_PANELS:
        z = cache.load(panel, "train")
        links = list(z["links"])
        dates = list(z["dates"])
        vcut = statics(panel, links)["vcut"]
        cap = pd.read_csv(
            REL / "corridors" / panel / "network" / "fd_parameters.csv",
            dtype={"link_id": str},
        ).groupby("link_id").capacity_vph.mean().reindex(links).to_numpy()
        wi = pd.read_csv(REL / "task2" / panel / "train" / "window_index.csv")
        wi = wi[wi.condition == "queue_onset"]
        slots = {tp.slot_of(x) for x in wi.forecast_origin}
        pools = {r: event_pool(z, slots, vcut, cap, radius=r, version="v1") for r in (0, 6)}
        pool_v2 = event_pool(z, slots, vcut, cap, radius=6, version="v2")
        weekends = (np.asarray(daytype(z["dates"])) >= 5).astype(int)
        for w in wi.itertuples():
            d = dates.index(w.date)
            s0 = tp.slot_of(w.forecast_origin)
            query, noisy, latent, known = query_event(z, d, s0, vcut, cap)
            zz = dict(z)
            zz["speed"], zz["flow"] = z["mspeed"], z["mflow"]
            query = history_features(zz, d, s0, vcut, cap, version="v1")
            query_v2 = history_features(zz, d, s0, vcut, cap, version="v2")
            rec = {"panel": panel, "date": w.date, "window_id": w.window_id}
            for radius, pool in pools.items():
                held_pool = [e for e in pool if e.month != w.date[:7]]
                candidates, tier = select_candidates(
                    held_pool, int(weekends[d]), s0, radius
                )
                rec[f"n_r{radius}"] = len(candidates)
                rec[f"tier_r{radius}"] = tier
                for train_label in ("noisy", "latent"):
                    if not candidates:
                        continue
                    for method, n_near in (("bayes", None), ("analog10", 10)):
                        action = bayes_action(candidates, train_label, query, n_near)
                        rec[f"r{radius}_{method}_{train_label}_k"] = int(action.sum())
                        rec[f"r{radius}_{method}_{train_label}_noisy"] = score_h6(action, noisy, known)
                        rec[f"r{radius}_{method}_{train_label}_latent"] = score_h6(action, latent, known)
            held_v2 = [e for e in pool_v2 if e.month != w.date[:7]]
            cand_v2, rec["tier_v2"] = select_candidates(held_v2, int(weekends[d]), s0, 6)
            rec["n_v2"] = len(cand_v2)
            extras = (
                ("r6_analog10_v2", 10, False, False),
                ("r6_analog10obs_v2", 10, True, False),
                ("r6_size10_v2", 10, False, True),
            )
            for train_label in ("noisy", "latent"):
                if not cand_v2:
                    continue
                for name, n_near, obs, size in extras:
                    action = bayes_action(
                        cand_v2, train_label, query_v2, n_near,
                        observed_only=obs, size_gate=size,
                    )
                    rec[f"{name}_{train_label}_k"] = int(action.sum())
                    rec[f"{name}_{train_label}_noisy"] = score_h6(action, noisy, known)
                    rec[f"{name}_{train_label}_latent"] = score_h6(action, latent, known)
            rows.append(rec)
    out = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT / "eval_windows.csv", index=False)
    value_cols = [c for c in out if c.endswith(("_noisy", "_latent"))]
    summary = []
    for c in value_cols:
        delta = out[c] - out["r0_bayes_noisy_noisy"] if c.endswith("_noisy") and "r0_bayes_noisy_noisy" in out else np.nan
        by_date = pd.DataFrame({"date": out.date, "x": out[c]}).groupby("date").x.mean()
        summary.append({
            "method": c,
            "mean": out[c].mean(),
            "se_date": by_date.std(ddof=1) / np.sqrt(len(by_date)),
            "n": out[c].notna().sum(),
            "delta_vs_r0_noisy": np.nanmean(delta) if np.ndim(delta) else np.nan,
        })
    sm = pd.DataFrame(summary).sort_values("mean", ascending=False)
    sm.to_csv(OUT / "summary.csv", index=False)
    print(out[[c for c in out if c.startswith("n_r")]].describe().round(1).to_string())
    print(sm.round(4).to_string(index=False))
    return out


def fitted_action(panel: str, split: str, w, method: str = "r6_bayes_latent") -> tuple[np.ndarray, list[str]]:
    ztr = cache.load(panel, "train")
    links = list(ztr["links"])
    vcut = statics(panel, links)["vcut"]
    cap = pd.read_csv(
        REL / "corridors" / panel / "network" / "fd_parameters.csv",
        dtype={"link_id": str},
    ).groupby("link_id").capacity_vph.mean().reindex(links).to_numpy()
    z = cache.load(panel, split)
    d = list(z["dates"]).index(w.date)
    s0 = tp.slot_of(w.forecast_origin)
    assert 12 <= s0 and s0 + 6 < 288, f"window {w.window_id} crosses midnight"
    # Inference features read the masked legal history.  Recreate a temporary
    # view because history_features expects the train cache's channel names.
    zz = dict(z)
    zz["speed"] = z["mspeed"]
    zz["flow"] = z["mflow"]
    spec = parse_method(method)
    query = history_features(zz, d, s0, vcut, cap, version=spec["version"])
    pool = event_pool(ztr, {s0}, vcut, cap, radius=spec["radius"], version=spec["version"])
    weekend = int(np.asarray(daytype([w.date]))[0] >= 5)
    candidates, _ = select_candidates(pool, weekend, s0, spec["radius"])
    if not candidates:
        return np.zeros(len(links), dtype=bool), links
    return bayes_action(
        candidates, spec["label"], query, spec["neighbours"],
        observed_only=spec["observed_only"], size_gate=spec["size_gate"],
    ), links


def build_rows(split: str, method: str = "r6_bayes_latent") -> pd.DataFrame:
    parts = []
    for panel in tp.Q_PANELS:
        ztr = cache.load(panel, "train")
        links = list(ztr["links"])
        vcut = statics(panel, links)["vcut"]
        cap = pd.read_csv(
            REL / "corridors" / panel / "network" / "fd_parameters.csv",
            dtype={"link_id": str},
        ).groupby("link_id").capacity_vph.mean().reindex(links).to_numpy()
        z = cache.load(panel, split)
        dates = list(z["dates"])
        zz = dict(z)
        zz["speed"], zz["flow"] = z["mspeed"], z["mflow"]
        wi = pd.read_csv(REL / "task2" / panel / split / "window_index.csv")
        wi = wi[wi.condition == "queue_onset"]
        spec = parse_method(method)
        pool = event_pool(
            ztr, {tp.slot_of(x) for x in wi.forecast_origin}, vcut, cap,
            radius=spec["radius"], version=spec["version"],
        )
        for w in wi.itertuples():
            d = dates.index(w.date)
            s0 = tp.slot_of(w.forecast_origin)
            assert 12 <= s0 and s0 + 6 < 288, f"window {w.window_id} crosses midnight"
            query = history_features(zz, d, s0, vcut, cap, version=spec["version"])
            weekend = int(np.asarray(daytype([w.date]))[0] >= 5)
            candidates, _ = select_candidates(pool, weekend, s0, spec["radius"])
            if not candidates:
                action = np.zeros(len(links), dtype=bool)
            else:
                action = bayes_action(
                    candidates, spec["label"], query, spec["neighbours"],
                    observed_only=spec["observed_only"], size_gate=spec["size_gate"],
                )
            ts = [f"{w.date}T{(s0 + h) // 12:02d}:{(s0 + h) % 12 * 5:02d}:00Z" for h in range(1, 7)]
            pred = np.zeros((6, len(links)), int)
            pred[5] = action.astype(int)
            parts.append(pd.DataFrame({
                "window_id": w.window_id,
                "timestamp": np.repeat(ts, len(links)),
                "link_id": np.tile(links, 6),
                "queue_pred": pred.reshape(-1),
            }))
    return pd.concat(parts, ignore_index=True)


if __name__ == "__main__":
    evaluate()
