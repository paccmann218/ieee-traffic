"""Freeze J1 as the legal control. Do not retune the onset analogue or Task 4.

Usage:
  python tfb/freeze_j1.py check
  python tfb/freeze_j1.py selftest

Rebuild of sub_J1.zip is refused unless TFB_ALLOW_J1_REBUILD=1 (reproduce-only).
"""
from __future__ import annotations

import hashlib
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, "tfb")
import legal as L

ROOT = Path(__file__).resolve().parents[1]
TFB = Path(__file__).resolve().parent


def md5_file(path: Path) -> str:
    h = hashlib.md5()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def refuse_j1_rebuild() -> None:
    """Call from J1 builders so a later experiment cannot silently overwrite the control."""
    if L.allow_j1_rebuild():
        print("TFB_ALLOW_J1_REBUILD=1: J1 rebuild permitted (reproduce-only)", flush=True)
        return
    raise SystemExit(
        "J1 is frozen (onset r6_analog10_noisy, ODME V1, ongoing ffill_inel). "
        "Do not retune analogue hyperparameters or Task 4. "
        "Set TFB_ALLOW_J1_REBUILD=1 only to reproduce the exact control file."
    )


def check_zip() -> dict:
    zpath = L.SUBS / L.J1_ZIP
    out = {
        "path": str(zpath),
        "exists": zpath.exists(),
        "md5": None,
        "md5_ok": None,
        "public_lb": L.J1_PUBLIC_LB,
        "onset_method": L.J1_ONSET_METHOD,
        "odme": L.J1_ODME,
        "ongoing": L.J1_ONGOING,
    }
    if zpath.exists():
        out["md5"] = md5_file(zpath)
        out["md5_ok"] = out["md5"] == L.J1_MD5
        out["size"] = zpath.stat().st_size
    return out


def check_t4_source() -> list[str]:
    """Task 4 must keep V1 on the split's synthetic_weak_prior.csv."""
    problems = []
    src = (TFB / "t4_build.py").read_text(encoding="utf-8")
    if "synthetic_weak_prior.csv" not in src:
        problems.append("t4_build.py does not read synthetic_weak_prior.csv")
    if re.search(r"read_csv\([^)]*base_od\.csv", src):
        problems.append("t4_build.py reads base_od.csv (illegal private prior)")
    if "def l2_exact" not in src or "lam=0.05" not in src:
        problems.append("t4_build.py V1 NNLS λ=0.05 is missing")
    return problems


def check_landmines() -> list[str]:
    problems = []
    skip = {"legal.py", "freeze_j1.py"}
    for path in TFB.glob("*.py"):
        if path.name in skip:
            continue
        text = path.read_text(encoding="utf-8")
        if re.search(r"read_csv\([^)]*base_od\.csv", text):
            problems.append(f"{path.name} reads base_od.csv")
    problems.extend(check_t4_source())
    return problems


def check() -> int:
    info = check_zip()
    print("J1 freeze", flush=True)
    for k, v in info.items():
        print(f"  {k}: {v}", flush=True)
    print(f"  refuse overwrite: {not L.allow_j1_rebuild()}", flush=True)
    print(f"  ship horizon ramps: {L.SHIP_HORIZON_RAMPS}", flush=True)
    problems = check_landmines()
    if info["exists"] and info["md5_ok"] is False:
        problems.append(f"sub_J1.zip md5 {info['md5']} != frozen {L.J1_MD5}")
    if problems:
        print("PROBLEMS:", *problems, sep="\n  ", flush=True)
        return 1
    print("landmines: none", flush=True)
    if not info["exists"]:
        print("note: sub_J1.zip not on this machine; freeze still in force", flush=True)
    return 0


def selftest() -> int:
    L.refuse_j1_write(Path("/tmp") / "sub_C0.zip")
    try:
        L.refuse_j1_write(L.SUBS / L.J1_ZIP)
    except PermissionError:
        pass
    else:
        raise AssertionError("refuse_j1_write must block sub_J1.zip")
    try:
        L.refuse_forbidden_read("corridors/D12_I5_N/network/base_od.csv")
    except PermissionError:
        pass
    else:
        raise AssertionError("base_od.csv must be refused")
    L.assert_odme_prior_path("task4/D12_I5_N/validation/synthetic_weak_prior.csv")
    assert L.J1_ONSET_METHOD == "r6_analog10_noisy"
    assert L.J1_ODME == "v1"
    assert L.SHIP_HORIZON_RAMPS is False
    assert abs(L.paid_J(1.0, 1.0) - 0.50) < 1e-12
    problems = check_t4_source()
    assert not problems, problems
    print("freeze_j1 selftest OK", flush=True)
    return 0


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "check"
    if cmd == "selftest":
        raise SystemExit(selftest())
    raise SystemExit(check())
