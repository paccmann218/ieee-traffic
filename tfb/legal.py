"""Contract freeze for the legal J1 control.

J1 (sub_J1.zip, public 0.84461) is the only selected file. Architecture C experiments
may write *new* isolated probes. They must not retune the onset analogue, reopen
Task 4, read base_od.csv, or take capacities/free-speeds from OSM links.csv.
"""
from __future__ import annotations

import os
from pathlib import Path

J1_NAME = "sub_J1"
J1_ZIP = "sub_J1.zip"
J1_CSV = "sub_J1.csv"
J1_MD5 = "78837f4bfaf68d06694b3e6895bb94ed"
J1_PUBLIC_LB = 0.84461
J1_ONSET_METHOD = "r6_analog10_noisy"
J1_ODME = "v1"
J1_ONGOING = "ffill_inel"

WEIGHT_STATE = 0.35
WEIGHT_QUEUE = 0.30
WEIGHT_PHYSICS = 0.15
WEIGHT_ODME = 0.20

FORBIDDEN_OUTPUTS = frozenset({J1_ZIP, J1_CSV})
FORBIDDEN_READS = frozenset({"base_od.csv"})

# Horizon ramp flows inside Task 2 [T, T+30] are released but gray-area. Do not ship.
SHIP_HORIZON_RAMPS = False

# Never Godunov-CTM on 25 m links (CFL ~0.8 s). Newell / LTM only.
FORBID_GODUNOV_DEFAULT = True

SUBS = Path.home() / "tfb/subs"
WORK = Path.home() / "tfb/work"
ARCH_C = WORK / "arch_c"


def paid_J(s_state: float, s_physics: float) -> float:
    """Paid Task 1+3 slice of S_total (queue and ODME frozen at J1)."""
    return WEIGHT_STATE * s_state + WEIGHT_PHYSICS * s_physics


def allow_j1_rebuild() -> bool:
    return os.environ.get("TFB_ALLOW_J1_REBUILD", "0") == "1"


def allow_t4_variant() -> bool:
    return os.environ.get("TFB_ALLOW_T4_VARIANT", "0") == "1"


def refuse_j1_write(path: Path | str) -> None:
    p = Path(path)
    if p.name in FORBIDDEN_OUTPUTS and not allow_j1_rebuild():
        raise PermissionError(
            f"Refusing to write {p.name}: J1 is the frozen legal control "
            f"(md5 {J1_MD5}, public {J1_PUBLIC_LB}). "
            "Architecture C probes must use a new name (e.g. sub_C0)."
        )


def refuse_forbidden_read(path: Path | str) -> None:
    p = Path(path)
    if p.name in FORBIDDEN_READS:
        raise PermissionError(
            f"Refusing to read {p}: base_od.csv holds the private-month OD prior. "
            "Task 4 V1 must use the split's own synthetic_weak_prior.csv."
        )


def assert_odme_prior_path(path: Path | str) -> Path:
    p = Path(path)
    refuse_forbidden_read(p)
    if p.name != "synthetic_weak_prior.csv":
        raise PermissionError(
            f"ODME prior must be synthetic_weak_prior.csv from the split folder, not {p.name}"
        )
    return p


def assert_fd_authority(frame, source: str) -> None:
    """Capacities and free-speeds must come from fd_parameters.csv, never OSM links.csv."""
    if "links.csv" in source.replace("\\", "/").lower() and any(
        c in frame.columns for c in ("free_speed_kmh", "capacity_vph", "free_speed")
    ):
        raise PermissionError(
            "OSM links.csv free_speed/capacity is the triangle-breaking trap. "
            "Use fd_parameters.csv only."
        )
