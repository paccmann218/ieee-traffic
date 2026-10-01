"""TrafficFlowBench score-budget and differential-probe arithmetic.

This module contains no fitted values.  It makes every assumption explicit so
that a leaderboard total is never mistaken for a separately identified task
score.

Examples
--------
    python tfb/score_frontier.py
    python tfb/score_frontier.py --u1 0.807 --u2 0.829
    python tfb/score_frontier.py --state .955 --physics .74 --odme .994
"""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass


W_STATE = 0.35
W_QUEUE = 0.30
W_PHYSICS = 0.15
W_ODME = 0.20
TARGETS = (0.90, 0.92, 0.94)
PUBLIC_R0 = 0.71192
PUBLIC_U1 = 0.80579
PUBLIC_U2 = 0.82600
PUBLIC_J1 = 0.84461
PUBLIC_L0 = 0.81733
ORGANIZER_QUEUE = 0.3017


@dataclass(frozen=True)
class Components:
    state: float
    physics: float
    odme: float

    @property
    def nonqueue(self) -> float:
        return W_STATE * self.state + W_PHYSICS * self.physics + W_ODME * self.odme

    def total(self, queue: float) -> float:
        return self.nonqueue + W_QUEUE * queue

    def queue_required(self, target: float) -> float:
        return (target - self.nonqueue) / W_QUEUE


SCENARIOS = {
    # The public R0 score minus the documented organizer-persistence queue
    # baseline.  This identifies only the weighted non-queue bundle.
    "current_inferred": {"nonqueue": 0.71192 - W_QUEUE * 0.3017},
    # CHANGELOG §5j working ceiling.  These are estimates, not hard bounds.
    "section_5j": {"components": Components(0.955, 0.740, 0.994)},
    # Published exact-answer benchmark.  It is an unattainable predictor
    # reference because it includes the held-out measurement noise.
    "exact_answer": {"components": Components(1.0, 0.9636, 1.0)},
    # Algebraic all-ones bound, useful only for detecting impossible claims.
    "all_ones": {"components": Components(1.0, 1.0, 1.0)},
}


def scenario_rows() -> list[dict[str, float | str | bool]]:
    rows: list[dict[str, float | str | bool]] = []
    for name, spec in SCENARIOS.items():
        if "components" in spec:
            c = spec["components"]
            assert isinstance(c, Components)
            nonqueue = c.nonqueue
        else:
            nonqueue = float(spec["nonqueue"])
        row: dict[str, float | str | bool] = {
            "scenario": name,
            "nonqueue": nonqueue,
            "max_total_at_queue_1": nonqueue + W_QUEUE,
        }
        for target in TARGETS:
            required = (target - nonqueue) / W_QUEUE
            row[f"queue_for_{target:.2f}"] = required
            row[f"{target:.2f}_feasible_with_queue_le_1"] = required <= 1.0
        rows.append(row)
    return rows


def probe_deltas(r0: float, u1: float | None, u2: float | None) -> dict[str, float]:
    """Decode the pre-registered U1/U2 one-condition probes.

    U1 changes onset only, so total delta = .30 * .5 * delta_onset.
    U2 changes ongoing only relative to U1, with the same .15 multiplier.
    """
    out: dict[str, float] = {}
    if u1 is not None:
        out["u1_minus_r0"] = u1 - r0
        out["onset_iou_delta"] = (u1 - r0) / (W_QUEUE / 2)
    if u2 is not None:
        if u1 is None:
            raise ValueError("--u2 requires --u1; U2-R0 mixes onset and ongoing")
        out["u2_minus_u1"] = u2 - u1
        out["ongoing_iou_delta"] = (u2 - u1) / (W_QUEUE / 2)
    if u1 is not None and u2 is not None:
        # The published organizer baseline predicts no onset queues, so its
        # aggregate 0.3017 means ongoing IoU = 2 * 0.3017.
        onset = out["onset_iou_delta"]
        ongoing = 2 * ORGANIZER_QUEUE + out["ongoing_iou_delta"]
        out.update(
            {
                "u1_onset_iou": onset,
                "u2_ongoing_iou": ongoing,
                "u2_queue_iou": (onset + ongoing) / 2,
                "identified_nonqueue": r0 - W_QUEUE * ORGANIZER_QUEUE,
            }
        )
    return out


def later_probes(
    r0: float,
    u1: float | None,
    u2: float,
    j1: float | None,
    l0: float | None,
) -> dict[str, float]:
    """J1 changes onset only vs U2; L0 changes state/physics only vs U2."""
    out: dict[str, float] = {}
    base = probe_deltas(r0, u1, u2)
    if j1 is not None:
        out["j1_minus_u2"] = j1 - u2
        out["j1_onset_iou_delta"] = (j1 - u2) / (W_QUEUE / 2)
        if "u1_onset_iou" in base:
            out["j1_onset_iou"] = base["u1_onset_iou"] + out["j1_onset_iou_delta"]
            out["j1_queue_iou"] = (out["j1_onset_iou"] + base["u2_ongoing_iou"]) / 2
    if l0 is not None:
        out["l0_minus_u2"] = l0 - u2
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--r0", type=float, default=PUBLIC_R0)
    ap.add_argument("--u1", type=float, default=PUBLIC_U1)
    ap.add_argument("--u2", type=float, default=PUBLIC_U2)
    ap.add_argument("--j1", type=float, default=PUBLIC_J1)
    ap.add_argument("--l0", type=float, default=PUBLIC_L0)
    ap.add_argument("--state", type=float)
    ap.add_argument("--physics", type=float)
    ap.add_argument("--odme", type=float)
    args = ap.parse_args()

    rows = scenario_rows()
    if any(v is not None for v in (args.state, args.physics, args.odme)):
        if any(v is None for v in (args.state, args.physics, args.odme)):
            ap.error("--state, --physics and --odme must be supplied together")
        c = Components(args.state, args.physics, args.odme)
        rows.append(
            {
                "scenario": "user_components",
                **asdict(c),
                "nonqueue": c.nonqueue,
                "max_total_at_queue_1": c.total(1.0),
                **{f"queue_for_{t:.2f}": c.queue_required(t) for t in TARGETS},
            }
        )

    payload = {
        "formula": "0.35*state + 0.30*queue + 0.15*physics + 0.20*odme",
        "scenarios": rows,
        "probe_deltas": {
            **probe_deltas(args.r0, args.u1, args.u2),
            **later_probes(args.r0, args.u1, args.u2, args.j1, args.l0),
        },
        "warnings": [
            "A total score identifies no component by itself.",
            "section_5j is a working ceiling, not a hard information bound.",
            "exact_answer includes held-out observations and is not a predictor ceiling.",
            "Never decode U2-R0 as one condition; use U2-U1.",
        ],
    }
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
