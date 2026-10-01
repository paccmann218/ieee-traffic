"""Information-set and representation oracles for Task 2.

The purpose of this file is to measure whether a model family has enough
representational headroom before fitting it.  Oracle columns are diagnostics
only and must never be submitted.

Train labels are proxies, not ``queue_true`` from ``score_task2.py``:
  noisy   released observations thresholded at v_cut
  latent  3-slot centered temporal median plus one-link spatial hole filling
          (non-causal; uses t+1 inside the horizon).  Rank methods with it,
          do not quote it as an official IoU.

Onset ``prior_top3`` / ``prior_oracle_size`` are h=6-only submissions scored
with full-window IoU, the same shape as U1/J1.  They are not comparable to a
mask that is positive on T+5..T+25.  ``oracle_h6`` is the exact last-step
location under that same submission shape.

``ffill_h*`` columns are per-step diagnostics.  Window IoU is not their mean.

Validation:
  * onset frequency ranks are leave-one-month-out;
  * ongoing front parameters are selected leave-one-date-out;
  * uncertainty is clustered by date because the 80 windows use few dates.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.ndimage import median_filter

sys.path.insert(0, "tfb")
import cache
import t2_persist as tp
from t2 import daytype, statics


REL = cache.REL
OUT = Path.home() / "tfb/work/t2_oracles"
K_MAX = 15
FRONT_DELTAS = tuple(range(-4, 5))


def iou(pred: np.ndarray, truth: np.ndarray, known: np.ndarray) -> float:
    inter = (pred & truth & known).sum()
    union = ((pred | truth) & known).sum()
    return 1.0 if union == 0 else float(inter / union)


def label_views(z: dict, d: int, s0: int, vcut: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    speed = z["speed"][d]
    known = z["elig"][d, s0 + 1:s0 + 7].astype(bool) & np.isfinite(speed[s0 + 1:s0 + 7])
    noisy = np.nan_to_num(speed[s0 + 1:s0 + 7], nan=1e9) <= vcut[None, :]
    smooth = median_filter(np.nan_to_num(speed, nan=1e9), size=(3, 1), mode="nearest")
    latent = tp.fill(smooth[s0 + 1:s0 + 7] <= vcut[None, :])
    return noisy, latent, known


def date_clustered_se(frame: pd.DataFrame, value: str, ref: str | None = None) -> float:
    x = frame[value] if ref is None else frame[value] - frame[ref]
    by_date = pd.DataFrame({"date": frame.date, "x": x}).groupby("date").x.mean()
    return float(by_date.std(ddof=1) / np.sqrt(len(by_date))) if len(by_date) > 1 else float("nan")


def date_clustered_ci(
    frame: pd.DataFrame,
    value: str,
    ref: str | None = None,
    *,
    reps: int = 10_000,
) -> tuple[float, float]:
    """Deterministic percentile interval, resampling whole scenario dates."""
    x = frame[value] if ref is None else frame[value] - frame[ref]
    by_date = pd.DataFrame({"date": frame.date, "x": x}).groupby("date").x.mean().to_numpy()
    if len(by_date) < 2:
        return float("nan"), float("nan")
    rng = np.random.default_rng(20261001)
    means = by_date[rng.integers(0, len(by_date), size=(reps, len(by_date)))].mean(axis=1)
    lo, hi = np.quantile(means, [0.025, 0.975])
    return float(lo), float(hi)


def onset_rows() -> pd.DataFrame:
    rows: list[dict] = []
    for panel in tp.Q_PANELS:
        z = cache.load(panel, "train")
        links = list(z["links"])
        dates = np.asarray(z["dates"]).astype(str)
        months = np.asarray([d[:7] for d in dates])
        weekend = (np.asarray(daytype(z["dates"])) >= 5).astype(int)
        vcut = statics(panel, links)["vcut"]
        eligible = z["elig"].astype(bool)
        queued = (np.nan_to_num(z["speed"], nan=1e9) <= vcut) & eligible
        valid = eligible & np.isfinite(z["speed"])
        wi = pd.read_csv(REL / "task2" / panel / "train" / "window_index.csv")
        for w in wi[wi.condition == "queue_onset"].itertuples():
            d = list(dates).index(w.date)
            s0 = tp.slot_of(w.forecast_origin)
            noisy, latent, known = label_views(z, d, s0, vcut)
            train_days = (months != months[d]) & (weekend == weekend[d])
            freq = queued[train_days, s0 + 6].sum(0) / np.maximum(valid[train_days, s0 + 6].sum(0), 1)
            order = np.argsort(-freq)
            rec = {"panel": panel, "date": w.date, "window_id": w.window_id}
            for label_name, truth in (("noisy", noisy), ("latent", latent)):
                scores = []
                for k in range(1, min(K_MAX, len(links)) + 1):
                    pred = np.zeros_like(truth)
                    # Legal onset shape: positives only at T+30, same as U1/J1.
                    pred[5, order[:k]] = True
                    scores.append(iou(pred, truth, known))
                    if k == 3:
                        rec[f"prior_top3_{label_name}"] = scores[-1]
                rec[f"prior_oracle_size_{label_name}"] = max(scores)
                # Exact h=6 location isolates timing error from location/size.
                pred_h6 = np.zeros_like(truth)
                pred_h6[5] = truth[5]
                rec[f"oracle_h6_{label_name}"] = iou(pred_h6, truth, known)
                rec[f"oracle_full_{label_name}"] = iou(truth, truth, known)
                rec[f"n_true_{label_name}"] = int((truth & known).sum())
                rec[f"n_h6_{label_name}"] = int((truth[5] & known[5]).sum())
            rows.append(rec)
    return pd.DataFrame(rows)


def runs(mask: np.ndarray) -> list[tuple[int, int]]:
    x = np.r_[False, np.asarray(mask, bool), False].astype(np.int8)
    edge = np.diff(x)
    return [(int(a), int(b - 1)) for a, b in zip(np.flatnonzero(edge == 1), np.flatnonzero(edge == -1))]


def front_path(q0: np.ndarray, left_delta: int, right_delta: int, clear_after: int = 7) -> np.ndarray:
    """Move each observed queue component's two fronts linearly over 30 min.

    Deltas are link-index changes at T+30.  This is a representation family,
    not an assertion about which index direction is upstream.
    """
    out = np.zeros((6, len(q0)), bool)
    rr = runs(tp.fill(q0))
    for h in range(1, 7):
        if h >= clear_after:
            continue
        for a, b in rr:
            aa = int(round(a + left_delta * h / 6))
            bb = int(round(b + right_delta * h / 6))
            aa, bb = max(0, aa), min(len(q0) - 1, bb)
            if aa <= bb:
                out[h - 1, aa:bb + 1] = True
    return out


def ongoing_rows() -> tuple[pd.DataFrame, list[str]]:
    records: list[dict] = []
    param_names = [f"L{a:+d}_R{b:+d}_C7" for a in FRONT_DELTAS for b in FRONT_DELTAS]
    param_names += [f"L+0_R+0_C{c}" for c in range(2, 7)]
    for panel in tp.Q_PANELS:
        z = cache.load(panel, "train")
        dates = list(z["dates"])
        vcut = statics(panel, z["links"])["vcut"]
        wi = pd.read_csv(REL / "task2" / panel / "train" / "window_index.csv")
        for w in wi[wi.condition == "queue_ongoing"].itertuples():
            d = dates.index(w.date)
            s0 = tp.slot_of(w.forecast_origin)
            noisy, latent, known = label_views(z, d, s0, vcut)
            hist = z["speed"][d, s0 - 12:s0]
            q0 = tp.variants(hist, hist, vcut)["ffill_inel"]
            baseline = np.repeat(q0[None], 6, axis=0)
            candidates: list[np.ndarray] = [
                front_path(q0, a, b) for a in FRONT_DELTAS for b in FRONT_DELTAS
            ]
            candidates += [front_path(q0, 0, 0, c) for c in range(2, 7)]
            rec = {
                "panel": panel,
                "date": w.date,
                "window_id": w.window_id,
                "n_q0": int(q0.sum()),
            }
            for label_name, truth in (("noisy", noisy), ("latent", latent)):
                vals = np.asarray([iou(p, truth, known) for p in candidates])
                best = int(vals.argmax())
                for name, value in zip(param_names, vals):
                    rec[f"{name}|{label_name}"] = float(value)
                rec[f"ffill_{label_name}"] = iou(baseline, truth, known)
                rec[f"front_oracle_{label_name}"] = float(vals[best])
                # Per-step persistence curve (diagnostic, not macro IoU).
                for h in range(6):
                    rec[f"ffill_h{h + 1}_{label_name}"] = iou(
                        baseline[h:h + 1], truth[h:h + 1], known[h:h + 1]
                    )
                    rec[f"front_bestpath_h{h + 1}_{label_name}"] = iou(
                        candidates[best][h:h + 1],
                        truth[h:h + 1],
                        known[h:h + 1],
                    )
                    rec[f"front_step_oracle_h{h + 1}_{label_name}"] = max(
                        iou(p[h:h + 1], truth[h:h + 1], known[h:h + 1])
                        for p in candidates
                    )
            records.append(rec)
    return pd.DataFrame(records), param_names


def add_leave_date_front_choice(frame: pd.DataFrame, params: list[str]) -> pd.DataFrame:
    out = frame.copy()
    for label in ("noisy", "latent"):
        chosen, score = [], []
        cols = [f"{p}|{label}" for p in params]
        for row in frame.itertuples():
            train = frame.date != row.date
            means = frame.loc[train, cols].mean(axis=0)
            best_col = str(means.idxmax())
            chosen.append(best_col.split("|")[0])
            hit = frame.loc[
                (frame.panel == row.panel) & (frame.window_id == row.window_id),
                best_col,
            ]
            score.append(float(hit.iloc[0]))
        out[f"front_lodo_{label}"] = score
        out[f"front_lodo_param_{label}"] = chosen
    return out


def summarize(frame: pd.DataFrame, condition: str) -> pd.DataFrame:
    methods = sorted({
        c[:-6] for c in frame.columns
        if c.endswith("_noisy")
        and not c.startswith(("n_", "ffill_h"))
        and pd.api.types.is_numeric_dtype(frame[c])
    })
    rows = []
    for method in methods:
        rec = {"condition": condition, "method": method, "n": len(frame), "n_dates": frame.date.nunique()}
        for label in ("noisy", "latent"):
            col = f"{method}_{label}"
            if col not in frame:
                continue
            rec[label] = float(frame[col].mean())
            rec[f"se_date_{label}"] = date_clustered_se(frame, col)
            lo, hi = date_clustered_ci(frame, col)
            rec[f"ci95_lo_date_{label}"] = lo
            rec[f"ci95_hi_date_{label}"] = hi
            ref = "prior_top3_noisy" if condition == "onset" and label == "noisy" else (
                "prior_top3_latent" if condition == "onset" else f"ffill_{label}"
            )
            rec[f"delta_{label}"] = float((frame[col] - frame[ref]).mean())
            rec[f"se_delta_date_{label}"] = date_clustered_se(frame, col, ref)
            lo, hi = date_clustered_ci(frame, col, ref)
            rec[f"ci95_lo_delta_date_{label}"] = lo
            rec[f"ci95_hi_delta_date_{label}"] = hi
        rows.append(rec)
    return pd.DataFrame(rows)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    onset = onset_rows()
    ongoing, params = ongoing_rows()
    ongoing = add_leave_date_front_choice(ongoing, params)
    onset.to_csv(OUT / "onset_windows.csv", index=False)
    ongoing.to_csv(OUT / "ongoing_windows.csv", index=False)
    summary = pd.concat([summarize(onset, "onset"), summarize(ongoing, "ongoing")], ignore_index=True)
    summary.to_csv(OUT / "summary.csv", index=False)
    show = summary[summary.method.isin(
        ["prior_top3", "prior_oracle_size", "oracle_h6", "oracle_full",
         "ffill", "front_lodo", "front_oracle"]
    )]
    print(show.round(4).to_string(index=False))
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
