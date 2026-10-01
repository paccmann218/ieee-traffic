"""Isolation and existence checks for the frontier candidates."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

SUBS = Path.home() / "tfb/subs"


def diffs(new: pd.DataFrame, ref: pd.DataFrame, name: str) -> dict[str, int]:
    assert np.array_equal(new.submission_id.to_numpy(), ref.submission_id.to_numpy())
    out = {}
    for task, columns in (
        ("state", ["speed_kmh", "flow_vph"]),
        ("queue", ["queue_pred"]),
        ("odme", ["path_flow"]),
    ):
        rows = new.task == task
        changed = ~np.isclose(
            new.loc[rows, columns].to_numpy(),
            ref.loc[rows, columns].to_numpy(),
            rtol=0,
            atol=1e-9,
        ).all(axis=1)
        out[task] = int(changed.sum())
        print(f"{name}: {task} rows differing = {out[task]}", flush=True)
    return out


def main() -> None:
    for name in ("sub_J1.zip", "sub_L0.zip", "sub_Rff.zip", "sub_U1.zip", "sub_U2.zip"):
        path = SUBS / name
        print(f"exists {name}: {path.exists()} size={path.stat().st_size if path.exists() else 0}")
    u2 = pd.read_csv(SUBS / "sub_U2.csv")
    j1 = pd.read_csv(SUBS / "sub_J1.csv")
    d = diffs(j1, u2, "J1 vs U2")
    assert d["state"] == 0 and d["odme"] == 0 and d["queue"] > 0
    print("J1 isolation OK")
    l0 = SUBS / "sub_L0.csv"
    if l0.exists():
        d0 = diffs(pd.read_csv(l0), u2, "L0 vs U2")
        print("L0 isolation", d0)
    feat = Path.home() / "tfb/work/t1feat/D7_I10_E.parquet"
    if feat.exists():
        cols = list(pd.read_parquet(feat).columns)
        print("t1feat has iv/iq:", "iv" in cols, "iq" in cols, "ncols", len(cols))
    audit = Path.home() / "tfb/work/contract_audit.csv"
    if audit.exists():
        frame = pd.read_csv(audit)
        print(frame.groupby("split")[[
            "template_rows", "cache_target_cells", "eligible_blank_cells",
            "eligible_blanks_not_targets", "targets_not_eligible_blanks",
        ]].sum().to_string())


if __name__ == "__main__":
    main()
