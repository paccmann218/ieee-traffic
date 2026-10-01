"""Read-only checks for release/template/scorer target-universe conflicts.

This does not attempt to exploit a scorer discrepancy.  It quantifies whether
the public Task 1 template equals "all eligible nulls" in the masked layer,
because MASK_SPEC.md says those sets differ around Task 2 horizons.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, "tfb")
import cache


OUT = Path.home() / "tfb/work/contract_audit.csv"


def main() -> None:
    rows = []
    for panel in cache.PANELS:
        for split in ("train", "validation", "private"):
            z = cache.load(panel, split)
            eligible = z["elig"].astype(bool)
            blank = ~np.isfinite(z["mspeed"]) & ~np.isfinite(z["mflow"])
            target = z["target"].astype(bool)
            tpl = pd.read_csv(
                cache.REL / "task1" / panel / split / "sample_submission_state.csv",
                usecols=["mask_regime"],
            )
            rows.append({
                "panel": panel,
                "split": split,
                "template_rows": len(tpl),
                "cache_target_cells": int(target.sum()),
                "eligible_blank_cells": int((eligible & blank).sum()),
                "eligible_blanks_not_targets": int((eligible & blank & ~target).sum()),
                "targets_not_eligible_blanks": int((target & ~(eligible & blank)).sum()),
            })
    out = pd.DataFrame(rows)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT, index=False)
    print(out.to_string(index=False))
    print("\nTotals:")
    print(out.groupby("split")[[
        "template_rows", "cache_target_cells", "eligible_blank_cells",
        "eligible_blanks_not_targets", "targets_not_eligible_blanks",
    ]].sum().to_string())


if __name__ == "__main__":
    main()
