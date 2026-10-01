"""Locations of the 90-minute Task 2 hole and the adjacent Task 1 cells.

MASK_SPEC blanks mainline for the 30-minute horizon plus a 60-minute buffer
around every Task 2 window (18 slots). Those cells are not Task 1 targets, but
time interpolation still crosses the hole, so Task 1 targets next to the edges
have a garbage tinterp base. t1_proxy never samples this slice.

Do not fill the extra eligible blanks as a scorer exploit. Use the locations
as a mask of "temporal interpolation is invalid here."

D12_I405 has no Task 2 windows (physics canary).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, "tfb")
import cache

REL = cache.REL
Q_PANELS = cache.PANELS[:8]
HOLE_AFTER_ORIGIN = 18   # T .. T+90 inclusive of origin slot (5 min + 30 + 60)
NEAR_BAND = 3            # slots just before / after the hole
GAP_DISABLE = 12         # disable tinterp when bounding gap exceeds 60 min


def slot_of(ts: str) -> int:
    return int(ts[11:13]) * 12 + int(ts[14:16]) // 5


def hole_slots(s0: int, n_slots: int = 288) -> np.ndarray:
    lo = max(0, s0)
    hi = min(n_slots, s0 + HOLE_AFTER_ORIGIN + 1)
    return np.arange(lo, hi)


def near_slots(s0: int, n_slots: int = 288) -> np.ndarray:
    hole = hole_slots(s0, n_slots)
    if hole.size == 0:
        return hole
    lo, hi = int(hole[0]), int(hole[-1])
    before = np.arange(max(0, lo - NEAR_BAND), lo)
    after = np.arange(hi + 1, min(n_slots, hi + 1 + NEAR_BAND))
    return np.concatenate([before, after])


def near_window_mask(panel: str, z: dict, split: str = "train"):
    """Return (near [D,288,L] bool, hole [D,288,L] bool).

    near: Task 1-relevant cells adjacent to a window hole (same day, all links).
    hole: the blanked 90-minute block itself (not scored for Task 1).
    """
    D, T, L = z["target"].shape
    near = np.zeros((D, T, L), bool)
    hole = np.zeros((D, T, L), bool)
    path = REL / "task2" / panel / split / "window_index.csv"
    if panel not in Q_PANELS or not path.exists():
        return near, hole
    wi = pd.read_csv(path)
    dates = list(z["dates"])
    di = {str(d): i for i, d in enumerate(dates)}
    for w in wi.itertuples():
        d = di.get(str(w.date))
        if d is None:
            continue
        s0 = slot_of(w.forecast_origin)
        hs = hole_slots(s0, T)
        ns = near_slots(s0, T)
        if hs.size:
            hole[d, hs, :] = True
        if ns.size:
            near[d, ns, :] = True
    return near, hole


def large_gap_mask(speed_obs, max_gap: int = GAP_DISABLE):
    """True where the nearest observed times on the same link are > max_gap slots apart.

    This catches Task 2 buffers even if window_index is unavailable.
    speed_obs: [D,T,L] with NaN at blanks.
    """
    D, T, L = speed_obs.shape
    out = np.zeros((D, T, L), bool)
    idx = np.arange(T)[:, None]
    for d in range(D):
        a = speed_obs[d]
        ok = np.isfinite(a)
        prev = np.maximum.accumulate(np.where(ok, idx, -1), axis=0)
        nxt = np.minimum.accumulate(np.where(ok, idx, 10**6)[::-1], axis=0)[::-1]
        gap = np.clip(nxt, 0, T - 1) - np.clip(prev, 0, T - 1)
        out[d] = (~ok) & (gap > max_gap)
    return out


def disable_tinterp_mask(panel: str, z: dict, split: str = "train"):
    """Cells where Architecture C must ignore time interpolation."""
    near, hole = near_window_mask(panel, z, split)
    obs = np.where(z["elig"].astype(bool) & np.isfinite(z["mspeed"]), z["mspeed"], np.nan)
    return near | hole | large_gap_mask(obs)


def score_near(z: dict, pv, pq, lanes, near_mask) -> dict:
    """Task 1 S_state restricted to near-window target cells (t1_proxy never tests this)."""
    import score_local as sl
    z2 = dict(z)
    z2["target"] = z["target"].astype(bool) & np.asarray(near_mask, bool)
    return sl.score_t1(z2, pv, pq, lanes)


def selftest() -> None:
    s0 = 12 * 16  # 16:00
    h = hole_slots(s0)
    n = near_slots(s0)
    assert h[0] == s0 and h[-1] == s0 + HOLE_AFTER_ORIGIN
    assert n[0] == s0 - NEAR_BAND and n[-1] == s0 + HOLE_AFTER_ORIGIN + NEAR_BAND
    assert np.intersect1d(h, n).size == 0
    D, T, L = 2, 288, 5
    z = dict(
        target=np.zeros((D, T, L), np.int8),
        dates=np.array(["2030-06-01", "2030-06-02"]),
        elig=np.ones((D, T, L), np.int8),
        mspeed=np.full((D, T, L), 80.0, np.float32),
    )
    z["mspeed"][0, s0:s0 + 19, :] = np.nan
    lg = large_gap_mask(z["mspeed"], max_gap=12)
    assert lg[0, s0 + 9, 0]
    assert not lg[1, s0 + 9, 0]
    z["target"] = np.zeros((D, T, L), np.int8)
    z["target"][0, s0 - 2, :] = 1
    z["speed"] = np.where(np.isnan(z["mspeed"]), np.nan, z["mspeed"])
    z["flow"] = np.full((D, T, L), 3000.0, np.float32)
    z["elig"] = np.ones((D, T, L), np.int8)
    z["regime"] = np.array(["R1", "R1"])
    near = np.zeros((D, T, L), bool)
    near[0, s0 - NEAR_BAND:s0, :] = True
    pv = np.nan_to_num(z["speed"], nan=80.0)
    pq = z["flow"]
    sn = score_near(z, pv, pq, np.full(L, 4.0), near)
    assert "S_state" in sn
    # Adjacent-to-hole slice: tinterp crosses the 90-min blank; Newell uses same-time neighbours.
    import t1_base as tb
    import t1_newell as tn
    Vs, Qs, fd = tn.synthetic_shock(T=80, L=16, seed=4)
    rng = np.random.default_rng(4)
    ok = rng.random(Vs.shape) > 0.25
    s0 = 40
    ok[s0:s0 + HOLE_AFTER_ORIGIN + 1, :] = False
    edge = np.zeros(Vs.shape, bool)
    edge[s0 - NEAR_BAND:s0, :] = ~ok[s0 - NEAR_BAND:s0, :]
    prior_v = tb.interp_day(np.where(ok, Vs, np.nan))
    prior_q = tb.interp_day(np.where(ok, Qs, np.nan))
    Vn, Qn, _ = tn.newell_fill_day(
        Vs, Qs, ok, fd["x"], fd["vf"], fd["cap"], fd["kc"], fd["kj"], fd["lanes"],
        prior_v, prior_q,
    )
    if edge.any():
        rmse_t = float(np.sqrt(np.mean((prior_v[edge] - Vs[edge]) ** 2)))
        rmse_n = float(np.sqrt(np.mean((Vn[edge] - Vs[edge]) ** 2)))
        print("near-hole rmse tinterp", round(rmse_t, 3), "newell", round(rmse_n, 3), flush=True)
        assert rmse_n <= rmse_t + 5.0
    print("t1_near_window selftest OK", "hole", len(h), "near", len(n), "S_state_near", sn["S_state"], flush=True)


if __name__ == "__main__":
    selftest()
