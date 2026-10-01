"""Task 1 base estimators on dense caches.

time_interp: per link, linear interpolation in time between the nearest usable observed cells of the same day
             (usable = eligible and not blank in the masked layer). Edges: nearest value.
profile:     weekday x slot mean per link from TRAIN observed layer (eligible), excluding the day being predicted
             when predicting on train (leave-one-day-out) to avoid leakage in local CV.
"""
from __future__ import annotations
import numpy as np
import pandas as pd


def usable(z, ch):
    """Observed cells usable as anchors in the masked layer: eligible, finite, not a target/blank."""
    x = z["m" + ch]
    return np.where(z["elig"].astype(bool) & np.isfinite(x), x, np.nan)


def interp_day(a):
    """a [288, L] with NaN gaps -> linear interp in time per column; edges held; all-NaN column stays NaN."""
    T, L = a.shape; out = a.copy(); t = np.arange(T)
    for l in range(L):
        ok = np.isfinite(a[:, l])
        if ok.sum() == 0: continue
        out[:, l] = np.interp(t, t[ok], a[ok, l])
    return out


def time_interp(z, ch):
    x = usable(z, ch)
    return np.stack([interp_day(x[d]) for d in range(x.shape[0])])


def profile(z_train, z_target=None, ch="speed", loo=True):
    """Weekday x slot x link mean from the train observed layer."""
    x = np.where(z_train["elig"].astype(bool), z_train[ch], np.nan)
    wd = pd.to_datetime(z_train["dates"]).weekday.to_numpy()
    S = np.zeros((7, 288, x.shape[2])); N = np.zeros_like(S)
    for w in range(7):
        xm = x[wd == w]; S[w] = np.nansum(xm, 0); N[w] = np.isfinite(xm).sum(0)
    if z_target is None:  # leave-one-day-out on train
        xs = np.nan_to_num(x); xn = np.isfinite(x)
        return np.stack([(S[wd[d]] - xs[d]) / np.maximum(N[wd[d]] - xn[d], 1) for d in range(x.shape[0])])
    wt = pd.to_datetime(z_target["dates"]).weekday.to_numpy()
    return np.stack([S[w] / np.maximum(N[w], 1) for w in wt])
