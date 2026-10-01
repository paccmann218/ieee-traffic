"""Candidate single-change files for the Oct 1 manual slots (plan v4 section 6; CHANGELOG section 8). Built, not uploaded.
U3 = U2 with ONSET rows = non-learned recurrent-bottleneck prior: train queue frequency at slot T+30 by link and weekday type
     (all train months), top-3 links at h=6, chosen WITHOUT horizon eligibility flags; ties (7/80 windows have < 3 positive
     frequencies) broken by the +-30 min frequency, then the all-day frequency. Nothing is fitted. On the 40 released
     train onset windows (frequency leave-one-month-out) this rule scores 0.558 latent / 0.621 noisy IoU.
     Q: (LB_U3 - LB_U2) / 0.15 = onset IoU of the prior minus onset IoU of U1's tree.
L0 = U2 with STATE = time interpolation of the released masked layer (t1_base.time_interp; train weekday profile where a
     link has no usable anchor that day). Q: LB_L0 - LB_U2 = 0.35 dS_state + 0.15 dS_physics of interpolation vs the GBDT.
Checks: cell-level isolation against U2 (only the intended rows move), row counts, ranges, empty-road rule.
"""
import sys, subprocess, numpy as np, pandas as pd
from pathlib import Path
sys.path.insert(0, "tfb")
import cache
import t1_base as tb
import t2_persist as tp
import build_u1u2 as b
import t2_onset_prior as op
from t2 import statics, daytype

W = b.W; SUBS = b.SUBS; REL = cache.REL; KEYS = ["window_id", "timestamp", "link_id"]


def onset_prior_rows(split, k=3):
    out = []
    for panel in tp.Q_PANELS:
        ztr = cache.load(panel, "train"); links = list(ztr["links"]); vcut = statics(panel, links)["vcut"]
        wt_tr = (np.asarray(daytype(ztr["dates"])) >= 5).astype(int); el = ztr["elig"].astype(bool)
        qd = (np.nan_to_num(ztr["speed"], nan=1e9) <= vcut) & el; vd = el & np.isfinite(ztr["speed"])
        z = cache.load(panel, split); assert list(z["links"]) == links; dates = list(z["dates"]); wt = (np.asarray(daytype(z["dates"])) >= 5).astype(int)
        wi = pd.read_csv(REL/"task2"/panel/split/"window_index.csv"); wi = wi[wi.condition == "queue_onset"]
        for w in wi.itertuples():
            d = dates.index(w.date); s0 = tp.slot_of(w.forecast_origin); assert 12 <= s0 and s0 + 6 < 288
            m = wt_tr == wt[d]; keys = op.rank_keys(qd, vd, None, m, s0 + 6)   # point frequency, ties -> +-30 min, then all-day
            sel = np.zeros((6, len(links)), int); sel[5, op.order_by(list(keys))[:k]] = 1
            ts = [f"{w.date}T{(s0 + j) // 12:02d}:{(s0 + j) % 12 * 5:02d}:00Z" for j in range(1, 7)]
            out.append(pd.DataFrame({"window_id": w.window_id, "timestamp": np.repeat(ts, len(links)), "link_id": np.tile(links, 6), "queue_pred": sel.reshape(-1)}))
    return pd.concat(out, ignore_index=True)


def tinterp_state():
    out = []
    for panel in cache.PANELS:
        ztr = cache.load(panel, "train")
        for split in ("validation", "private"):
            z = cache.load(panel, split); links = z["links"]
            v, q = tb.time_interp(z, "speed"), tb.time_interp(z, "flow")
            pv, pq = tb.profile(ztr, z, "speed"), tb.profile(ztr, z, "flow")
            v = np.where(np.isfinite(v), v, pv); q = np.where(np.isfinite(q), q, pq)
            tpl = pd.read_csv(REL/"task1"/panel/split/"sample_submission_state.csv", usecols=["panel", "timestamp", "station_id", "link_id", "mask_regime"], dtype=str)
            di = {dd: i for i, dd in enumerate(z["dates"])}; li = {l: i for i, l in enumerate(links)}
            d_ = tpl.timestamp.str[:10].map(di).to_numpy(); s_ = (tpl.timestamp.str[11:13].astype(int) * 12 + tpl.timestamp.str[14:16].astype(int) // 5).to_numpy(); l_ = tpl.link_id.map(li).to_numpy()
            tpl["speed_kmh"] = np.clip(v[d_, s_, l_], 5.0, 140.0); tpl["flow_vph"] = np.clip(q[d_, s_, l_], 0.0, None)
            assert tpl.speed_kmh.notna().all() and tpl.flow_vph.notna().all(), (panel, split)
            out.append(tpl); print(panel, split, len(tpl), flush=True)
    o = pd.concat(out, ignore_index=True); o.to_csv(W/"state_tinterp.csv", index=False); return W/"state_tinterp.csv"


def merge(state_csv, queue_csv, out_name):
    r = subprocess.run([sys.executable, "repo/src/merge_submissions.py", "--state", str(state_csv), "--queue", str(queue_csv), "--odme", "subs/odme_ref.csv",
                        "--key", str(REL/"submission_key.csv"), "--output", str(SUBS/f"{out_name}.csv")], capture_output=True, text=True, cwd=str(Path.home()/"tfb"))
    assert "Every scored cell was filled" in r.stdout, r.stdout[-500:]
    s = pd.read_csv(SUBS/f"{out_name}.csv"); st = s.task == "state"
    s.loc[st, "speed_kmh"] = s.loc[st, "speed_kmh"].clip(lower=5.0)
    assert len(s) == 6980503 and s.isna().sum().sum() == 0
    assert set(s.loc[s.task == "queue", "queue_pred"].unique()) <= {0, 1}
    assert (s.loc[s.task == "odme", "path_flow"] >= 0).all() and s.loc[st, "speed_kmh"].between(5, 140).all()
    low = (s.loc[st, "flow_vph"] < 50).mean(); assert low < 0.2, low
    s.to_csv(SUBS/f"{out_name}.csv", index=False)
    subprocess.run(["zip", "-q", "-j", "-o", str(SUBS/f"{out_name}.zip"), str(SUBS/f"{out_name}.csv")], check=True)
    return s


def isolate(new, ref, name):
    """Cell-level diff by task between two merged files (same submission_key order)."""
    assert (new.submission_id.values == ref.submission_id.values).all()
    for t, cols in (("state", ["speed_kmh", "flow_vph"]), ("queue", ["queue_pred"]), ("odme", ["path_flow"])):
        m = (new.task == t).values; nd = int((~np.isclose(new.loc[m, cols].values, ref.loc[m, cols].values, rtol=0, atol=1e-9)).any(1).sum())
        print(f"{name} vs U2: {t} rows differing = {nd}", flush=True); yield t, nd


def main():
    u2 = pd.read_csv(SUBS/"sub_U2.csv"); tpl = b.template(); q2 = pd.read_csv(W/"queue_U2.csv", dtype={"window_id": str, "link_id": str})
    wi = pd.concat([pd.read_csv(REL/"task2"/p/s/"window_index.csv") for p in tp.Q_PANELS for s in b.SPLITS])[["window_id", "condition"]]
    # U3: onset prior
    ons = pd.concat([onset_prior_rows(s) for s in b.SPLITS], ignore_index=True)
    q3 = b.assemble(tpl, q2, ons); c = q3.merge(wi, on="window_id").condition.to_numpy()
    dq = q3.queue_pred.values != q2.queue_pred.values; assert not dq[c == "queue_ongoing"].any(), "U3 must not touch ongoing rows"
    print("U3 onset cells changed vs U2:", int(dq.sum()), "| onset positives:", int(q3.queue_pred.values[c == "queue_onset"].sum()), flush=True)
    q3.to_csv(W/"queue_U3.csv", index=False); s3 = merge("work/t1/state_submission.csv", W/"queue_U3.csv", "sub_U3")
    d = dict(isolate(s3, u2, "U3")); assert d["state"] == 0 and d["odme"] == 0 and d["queue"] == int(dq.sum())
    if "u3only" in sys.argv: print("DONE (U3 only)", flush=True); return
    # L0: time-interpolation state
    st = tinterp_state(); s0 = merge(st, W/"queue_U2.csv", "sub_L0")
    d = dict(isolate(s0, u2, "L0")); assert d["queue"] == 0 and d["odme"] == 0 and d["state"] > 0
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
