"""Build isolated candidates from the honest-score-frontier experiments.

J1 changes only U2 onset rows:
  joint train-only onset-mask analogue, ±30 minute candidate pool, 10 nearest
  legal-history analogues, noisy train masks, empirical expected-IoU action.

The model was chosen before any J1 leaderboard result.  Its leave-one-month-out
train score is 0.960 against observed labels and 0.785 against the conservative
median proxy, versus 0.621/0.558 for the fixed top-3 prior.
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "tfb")
import build_u1u2 as b
import t2_joint_prior as joint


METHOD = "r6_analog10_noisy"


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
        print(f"{name} vs U2: {task} rows differing = {out[task]}", flush=True)
    return out


def main() -> None:
    import freeze_j1
    freeze_j1.refuse_j1_rebuild()
    template = b.template()
    q2 = pd.read_csv(b.W / "queue_U2.csv", dtype={"window_id": str, "link_id": str})
    onset = pd.concat(
        [joint.build_rows(split, METHOD) for split in b.SPLITS],
        ignore_index=True,
    )
    qj = b.assemble(template, q2, onset)
    index = pd.concat([
        pd.read_csv(b.REL / "task2" / panel / split / "window_index.csv")
        for panel in b.t2s.Q_PANELS for split in b.SPLITS
    ])[["window_id", "condition"]]
    condition = qj.merge(index, on="window_id").condition.to_numpy()
    changed = qj.queue_pred.to_numpy() != q2.queue_pred.to_numpy()
    assert not changed[condition == "queue_ongoing"].any()
    assert changed[condition == "queue_onset"].any()
    print(
        "J1 onset rows changed:", int(changed.sum()),
        "| positives:", int(qj.loc[condition == "queue_onset", "queue_pred"].sum()),
        flush=True,
    )
    qpath = b.W / "queue_J1.csv"
    qj.to_csv(qpath, index=False)
    b.merge_and_check(qpath, "sub_J1")
    new = pd.read_csv(b.SUBS / "sub_J1.csv")
    ref = pd.read_csv(b.SUBS / "sub_U2.csv")
    diff = isolate(new, ref, "J1")
    assert diff["state"] == 0 and diff["odme"] == 0 and diff["queue"] == int(changed.sum())


if __name__ == "__main__":
    main()
