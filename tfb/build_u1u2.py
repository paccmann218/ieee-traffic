"""Build U1 and U2 (plan v4 section 4a). Legal Task 2 information set: history [T-60, T) only.

R0 (measured, 0.71192) = v1 GBDT state + organizer persistence queue (subs/q_persist.csv) + V1 ODME.
U1 = R0 with ONSET rows from a train-only onset LightGBM (t2s tables; features from slots s0-12..s0-1; day-context
     columns excluded; h=6 only; expected-IoU decoder). Ongoing rows unchanged (organizer persistence).
U2 = U1 with ONGOING rows = forward-filled last released history observation per link (incl. ineligible values),
     chosen on 40 released train ongoing windows: +0.165 +- 0.019 IoU vs organizer persistence (noisy labels).
Checks: R0 queue reproduction, information-set assert (inputs outside [s0-12, s0) cannot change onset predictions),
merge gaps, row count, nulls, value ranges.
"""
import sys, subprocess, numpy as np, pandas as pd, lightgbm as lgb
from pathlib import Path
sys.path.insert(0, "tfb")
import cache
import t2s
from t2 import statics, queue_freq, daytype, pick_set
import t2_persist as tp

assert not t2s.INCL_INELIG, "U1 onset model uses the eligible-only tables (as validated)"
REL = cache.REL; W = Path.home()/"tfb/work/u1u2"; W.mkdir(parents=True, exist_ok=True)
SUBS = Path.home()/"tfb/subs"; SPLITS = ("validation", "private")


def fit_onset():
    tr = t2s.load("train", "onset"); tr = tr[tr.known]
    feats = [c for c in tr.columns if c not in t2s.META + ["cond"] + t2s.DAYCTX]
    assert not set(feats) & set(t2s.DAYCTX)
    m = lgb.train(t2s.PARAMS, lgb.Dataset(tr[feats], tr.y), 400); m.save_model(str(W/"onset_legal.txt"))
    print("onset model: rows", len(tr), "features", len(feats), flush=True)
    return m, feats


def onset_rows(m, feats, split, check=True):
    out, checked = [], 0
    for pi, panel in enumerate(t2s.Q_PANELS):
        ztr = cache.load(panel, "train"); st = statics(panel, ztr["links"]); _, _, _, S, N = queue_freq(ztr, st["vcut"]); links = list(ztr["links"])
        z = cache.load(panel, split); assert list(z["links"]) == links; dates = list(z["dates"]); dows = daytype(z["dates"])
        wi = pd.read_csv(REL/"task2"/panel/split/"window_index.csv"); wi = wi[wi.condition == "queue_onset"]
        for w in wi.itertuples():
            d = dates.index(w.date); s0 = tp.slot_of(w.forecast_origin); wt = int(dows[d] >= 5)
            assert 12 <= s0 and s0 + 6 < 288, f"window {w.window_id} crosses midnight: timestamp keys would not match"
            f6 = S[wt][s0 + 1:s0 + 7] / np.maximum(N[wt][s0 + 1:s0 + 7], 1)
            mv, mq, mo, el = t2s.day_layer(z, d)
            X = t2s.features(mv, mq, mo, st, s0, dows[d], f6, pi); p = m.predict(X[feats])
            if check and checked < 6:   # information-set assert: blank everything outside [s0-12, s0)
                mv2, mq2, mo2 = mv.copy(), mq.copy(), mo.copy()
                for a in (mv2, mq2, mo2): a[:s0 - 12] = np.nan; a[s0:] = np.nan
                p2 = m.predict(t2s.features(mv2, mq2, mo2, st, s0, dows[d], f6, pi)[feats])
                assert np.allclose(p, p2), f"information-set violation in {w.window_id}"; checked += 1
            p = np.where(X.h.to_numpy() == 6, p, 0.0); known = z["elig"][d, s0 + 1:s0 + 7].reshape(-1).astype(bool)
            sel = pick_set(p, known)
            if not (sel & known).any(): sel[np.argmax(np.where(known, p, -1))] = True
            ts = [f"{w.date}T{(s0 + k) // 12:02d}:{(s0 + k) % 12 * 5:02d}:00Z" for k in range(1, 7)]
            out.append(pd.DataFrame({"window_id": w.window_id, "timestamp": np.repeat(ts, len(links)), "link_id": np.tile(links, 6), "queue_pred": sel.astype(int)}))
    print(split, "onset rows", sum(len(o) for o in out), "information-set asserts passed:", checked, flush=True)
    return pd.concat(out, ignore_index=True)


def template():
    parts = [pd.read_csv(REL/"task2"/p/s/"sample_submission_queue.csv", dtype=str, usecols=["window_id", "timestamp", "link_id"]) for p in t2s.Q_PANELS for s in SPLITS]
    return pd.concat(parts, ignore_index=True)


def organizer_rows():
    q = pd.read_csv(SUBS/"q_persist.csv", dtype={"window_id": str, "link_id": str})
    q["timestamp"] = pd.to_datetime(q.timestamp, utc=True).dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    return q[["window_id", "timestamp", "link_id", "queue_pred"]]


def assemble(tpl, base, repl):
    """Start from base rows; overwrite rows present in repl."""
    keys = ["window_id", "timestamp", "link_id"]
    m = tpl.merge(base, on=keys, how="left")
    assert m.queue_pred.notna().all(), "base rows missing"
    if repl is not None:
        m = m.merge(repl.rename(columns={"queue_pred": "rp"}), on=keys, how="left")
        m["queue_pred"] = np.where(m.rp.notna(), m.rp, m.queue_pred); m = m.drop(columns="rp")
    m["queue_pred"] = m.queue_pred.astype(int); return m


def assert_r0_cells(q):
    """Cell-level (not count-level) check that template-keyed queue rows q equal the measured R0 file (0.71192).
    The merged file is in submission_key order, so compare through the key, never by position."""
    key = pd.read_csv(REL/"submission_key.csv", usecols=["submission_id", "task", "timestamp", "link_id", "window_id"], dtype=str)
    key = key[key.task == "queue"]; ref = pd.read_csv(SUBS/"sub_probe_v1state_persq.csv", usecols=["submission_id", "task", "queue_pred"])
    ref = ref[ref.task == "queue"]; ref["submission_id"] = ref.submission_id.astype(str)
    k = key.merge(ref[["submission_id", "queue_pred"]], on="submission_id", how="left", validate="one_to_one")
    k["timestamp"] = pd.to_datetime(k.timestamp, utc=True).dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    m = q.merge(k[["window_id", "timestamp", "link_id", "queue_pred"]].rename(columns={"queue_pred": "ref"}), on=["window_id", "timestamp", "link_id"], how="left", validate="one_to_one")
    assert m.ref.notna().all() and len(m) == len(key), "R0 key coverage mismatch"
    bad = int((m.ref.astype(int) != m.queue_pred.astype(int)).sum()); assert bad == 0, f"R0 queue mismatch in {bad} cells"
    print("R0 queue reproduced cell by cell (keyed): ones =", int(q.queue_pred.sum()), flush=True)


def merge_and_check(queue_csv, out_name):
    r = subprocess.run([sys.executable, "repo/src/merge_submissions.py", "--state", "work/t1/state_submission.csv", "--queue", str(queue_csv),
                        "--odme", "subs/odme_ref.csv", "--key", str(REL/"submission_key.csv"), "--output", str(SUBS/f"{out_name}.csv")],
                       capture_output=True, text=True, cwd=str(Path.home()/"tfb"))
    msg = r.stdout.strip().splitlines()[-2:]; print(out_name, "merge:", msg, flush=True)
    assert "Every scored cell was filled" in r.stdout, r.stdout[-500:]
    s = pd.read_csv(SUBS/f"{out_name}.csv"); st = s.task == "state"
    s.loc[st, "speed_kmh"] = s.loc[st, "speed_kmh"].clip(lower=5.0)
    assert len(s) == 6980503 and s.isna().sum().sum() == 0
    assert set(s.loc[s.task == "queue", "queue_pred"].unique()) <= {0, 1}
    assert (s.loc[s.task == "odme", "path_flow"] >= 0).all() and np.isfinite(s.path_flow).all()
    assert s.loc[st, "speed_kmh"].between(5, 140).all()
    low = (s.loc[st, "flow_vph"] < 50).mean(); assert low < 0.2, low
    s.to_csv(SUBS/f"{out_name}.csv", index=False)
    subprocess.run(["zip", "-q", "-j", "-o", str(SUBS/f"{out_name}.zip"), str(SUBS/f"{out_name}.csv")], check=True)
    print(out_name, "OK rows", len(s), "queue share", round(s.loc[s.task == "queue", "queue_pred"].mean(), 4), flush=True)


def main():
    tpl = template(); org = organizer_rows()
    # R0 reproduction check: organizer rows through our assembly == the measured R0 queue rows
    r0 = assemble(tpl, org, None); assert_r0_cells(r0)
    m, feats = fit_onset()
    ons = pd.concat([onset_rows(m, feats, s) for s in SPLITS], ignore_index=True)
    u1 = assemble(tpl, org, ons); u1.to_csv(W/"queue_U1.csv", index=False)
    per = pd.concat([tp.build_rows("ffill_inel", s) for s in SPLITS], ignore_index=True)
    u2 = assemble(tpl, org, pd.concat([ons, per], ignore_index=True)); u2.to_csv(W/"queue_U2.csv", index=False)
    wi = pd.concat([pd.read_csv(REL/"task2"/p/s/"window_index.csv") for p in t2s.Q_PANELS for s in SPLITS])[["window_id", "condition"]]
    for nm, q in (("R0", r0), ("U1", u1), ("U2", u2)):
        print(nm, q.merge(wi, on="window_id").groupby("condition").queue_pred.mean().round(4).to_dict(), flush=True)
    diff_on = (u1.merge(wi, on="window_id").query("condition=='queue_ongoing'").queue_pred.values != r0.merge(wi, on="window_id").query("condition=='queue_ongoing'").queue_pred.values).sum()
    assert diff_on == 0, "U1 must not change ongoing rows"
    merge_and_check(W/"queue_U1.csv", "sub_U1"); merge_and_check(W/"queue_U2.csv", "sub_U2")


if __name__ == "__main__":
    main()
