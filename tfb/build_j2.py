"""Isolated J1 upgrades.  Built only after leave-month / leave-date gates.

J2_onset: J1 with onset method replaced (state/ODME/ongoing frozen).
J2_ongoing: J1 with ongoing = history-front rollout (state/ODME/onset frozen).
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "tfb")
import build_u1u2 as b
import t2_hist_front as hf
import t2_joint_prior as joint


def isolate(new: pd.DataFrame, reference: pd.DataFrame, name: str) -> dict[str, int]:
    assert np.array_equal(new.submission_id.to_numpy(), reference.submission_id.to_numpy())
    out = {}
    for task, columns in (
        ("state", ["speed_kmh", "flow_vph"]),
        ("queue", ["queue_pred"]),
        ("odme", ["path_flow"]),
    ):
        rows = new.task == task
        changed = ~np.isclose(
            new.loc[rows, columns].to_numpy(),
            reference.loc[rows, columns].to_numpy(),
            rtol=0,
            atol=1e-9,
        ).all(axis=1)
        out[task] = int(changed.sum())
        print(f"{name}: {task} rows differing = {out[task]}", flush=True)
    return out


def build_onset(method: str, name: str) -> None:
    template = b.template()
    qj = pd.read_csv(b.W / "queue_J1.csv", dtype={"window_id": str, "link_id": str})
    onset = pd.concat([joint.build_rows(split, method) for split in b.SPLITS], ignore_index=True)
    qn = b.assemble(template, qj, onset)
    index = pd.concat([
        pd.read_csv(b.REL / "task2" / panel / split / "window_index.csv")
        for panel in b.t2s.Q_PANELS for split in b.SPLITS
    ])[["window_id", "condition"]]
    condition = qn.merge(index, on="window_id").condition.to_numpy()
    changed = qn.queue_pred.to_numpy() != qj.queue_pred.to_numpy()
    assert not changed[condition == "queue_ongoing"].any()
    print(name, "onset cells vs J1:", int(changed.sum()), flush=True)
    path = b.W / f"queue_{name}.csv"
    qn.to_csv(path, index=False)
    b.merge_and_check(path, f"sub_{name}")
    diff = isolate(pd.read_csv(b.SUBS / f"sub_{name}.csv"), pd.read_csv(b.SUBS / "sub_J1.csv"), name)
    assert diff["state"] == 0 and diff["odme"] == 0


def build_ongoing() -> None:
    template = b.template()
    qj = pd.read_csv(b.W / "queue_J1.csv", dtype={"window_id": str, "link_id": str})
    ongoing = pd.concat([hf.build_rows(split) for split in b.SPLITS], ignore_index=True)
    qn = b.assemble(template, qj, ongoing)
    index = pd.concat([
        pd.read_csv(b.REL / "task2" / panel / split / "window_index.csv")
        for panel in b.t2s.Q_PANELS for split in b.SPLITS
    ])[["window_id", "condition"]]
    condition = qn.merge(index, on="window_id").condition.to_numpy()
    changed = qn.queue_pred.to_numpy() != qj.queue_pred.to_numpy()
    assert not changed[condition == "queue_onset"].any()
    print("J2og ongoing cells vs J1:", int(changed.sum()), flush=True)
    path = b.W / "queue_J2og.csv"
    qn.to_csv(path, index=False)
    b.merge_and_check(path, "sub_J2og")
    diff = isolate(pd.read_csv(b.SUBS / "sub_J2og.csv"), pd.read_csv(b.SUBS / "sub_J1.csv"), "J2og")
    assert diff["state"] == 0 and diff["odme"] == 0


def main() -> None:
    mode = sys.argv[1] if len(sys.argv) > 1 else "onset"
    if mode == "onset":
        method = sys.argv[2] if len(sys.argv) > 2 else "r6_analog10_v2_noisy"
        name = sys.argv[3] if len(sys.argv) > 3 else "J2"
        build_onset(method, name)
    elif mode == "ongoing":
        build_ongoing()
    else:
        raise SystemExit("mode must be onset|ongoing")


if __name__ == "__main__":
    main()
