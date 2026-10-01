"""CRITIQUE.md section 0.7/0.8: the robust final candidate, built and NOT uploaded.
R_ff = R0 (GBDT state + V1 ODME) with ONGOING rows = ffill_inel (non-learned legal carry-forward, history [T-60, T) only)
and ONSET rows = organizer persistence (empty by construction). Cell-level checks against R0 and U2.
"""
import sys, numpy as np, pandas as pd
sys.path.insert(0, "tfb")
import build_u1u2 as b
import t2_persist as tp

KEYS = ["window_id", "timestamp", "link_id"]


def main():
    tpl = b.template(); org = b.organizer_rows()
    per = pd.concat([tp.build_rows("ffill_inel", s) for s in b.SPLITS], ignore_index=True)
    q = b.assemble(tpl, org, per); r0 = b.assemble(tpl, org, None)
    u2 = pd.read_csv(b.W/"queue_U2.csv", dtype={"window_id": str, "link_id": str})
    wi = pd.concat([pd.read_csv(b.REL/"task2"/p/s/"window_index.csv") for p in b.t2s.Q_PANELS for s in b.SPLITS])[["window_id", "condition"]]
    c = q.merge(wi, on="window_id")[["condition"]].condition.to_numpy()
    assert (q[KEYS].values == r0[KEYS].values).all() and (q[KEYS].values == u2[KEYS].values).all(), "row order differs"
    d_r0 = q.queue_pred.values != r0.queue_pred.values; d_u2 = q.queue_pred.values != u2.queue_pred.values
    assert not d_r0[c == "queue_onset"].any(), "onset rows must equal R0 (empty)"
    assert not d_u2[c == "queue_ongoing"].any(), "ongoing rows must equal U2"
    print("R_ff vs R0: changed cells", int(d_r0.sum()), "all ongoing; R_ff vs U2: changed cells", int(d_u2.sum()), "all onset", flush=True)
    q.to_csv(b.W/"queue_Rff.csv", index=False); b.merge_and_check(b.W/"queue_Rff.csv", "sub_Rff")


if __name__ == "__main__":
    main()
