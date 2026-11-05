"""Regenerate every figure and generated table used by the final report and README.

    python docs/scripts/make_figures.py [--network-root data/small]

Inputs
------
docs/data/submissions.csv                     all 31 Kaggle submissions of the team
docs/data/public_leaderboard_scores_2026-11-05.csv
                                              public leaderboard, rank and score only
--network-root (optional)                     a copy of the competition release that
                                              contains corridors/<PANEL>/network/; used
                                              only for the fundamental-diagram figure
deliverables/ES4_20261009/gap_ladder.json     per-panel February gap-shell statistics
                                              (optional; used for the per-panel dots)

Every other number below is a recorded measurement. Each constant carries the
research-ledger entry it comes from. Local numbers are TRAIN or visible-cell
proxies; only the submission CSV holds Kaggle scores.

Outputs
-------
docs/report/figures/*.pdf      vector figures for the LaTeX report
docs/figures/*.png             raster copies for the README
docs/report/generated/*.tex    tables generated from the same data
"""
from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "docs" / "data"
FIG_PDF = ROOT / "docs" / "report" / "figures"
FIG_PNG = ROOT / "docs" / "figures"
GEN = ROOT / "docs" / "report" / "generated"

# ---------------------------------------------------------------- style --
# Validated categorical order (blue, orange, aqua); neutral greys for context.
BLUE, ORANGE, AQUA, RED = "#2a78d6", "#eb6834", "#1baf7a", "#e34948"
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS = "#e1e0d9", "#c3c2b7"
GREY, LIGHT = "#a8a7a0", "#d9d8d1"
SURFACE = "#ffffff"


def _register_fonts() -> str:
    """Use Libertinus Sans (the report's companion face) when TeX can find it."""
    family = "DejaVu Sans"
    kpse = shutil.which("kpsewhich")
    if not kpse:
        return family
    found = []
    for name in ("LibertinusSans-Regular.otf", "LibertinusSans-Bold.otf", "LibertinusSans-Italic.otf"):
        try:
            path = subprocess.run([kpse, name], capture_output=True, text=True, timeout=30).stdout.strip()
        except (OSError, subprocess.SubprocessError):
            path = ""
        if path and Path(path).is_file():
            font_manager.fontManager.addfont(path)
            found.append(path)
    return "Libertinus Sans" if found else family


def setup_style() -> None:
    plt.rcParams.update({
        "font.family": _register_fonts(),
        "font.size": 8.5,
        "axes.titlesize": 9,
        "axes.titleweight": "bold",
        "axes.titlelocation": "left",
        "axes.labelsize": 8.5,
        "axes.labelcolor": INK2,
        "axes.edgecolor": AXIS,
        "axes.linewidth": 0.6,
        "axes.facecolor": SURFACE,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": GRID,
        "grid.linewidth": 0.5,
        "grid.linestyle": "-",
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "xtick.labelcolor": INK2,
        "ytick.labelcolor": INK2,
        "xtick.labelsize": 7.8,
        "ytick.labelsize": 7.8,
        "xtick.major.width": 0.5,
        "ytick.major.width": 0.5,
        "xtick.major.size": 2.5,
        "ytick.major.size": 2.5,
        "legend.fontsize": 7.8,
        "legend.frameon": False,
        "figure.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "pdf.fonttype": 42,
        "svg.fonttype": "path",
        "text.color": INK,
    })


def save(fig, name: str) -> None:
    FIG_PDF.mkdir(parents=True, exist_ok=True)
    FIG_PNG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_PDF / f"{name}.pdf", bbox_inches="tight", pad_inches=0.04)
    fig.savefig(FIG_PNG / f"{name}.png", dpi=220, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)
    print("wrote", name)


def note(ax, text: str, y: float = -0.2) -> None:
    ax.text(0, y, text, transform=ax.transAxes, fontsize=7, color=MUTED, ha="left", va="top")


# ------------------------------------------------------------ data load --
def load_submissions() -> list[dict]:
    with open(DATA / "submissions.csv", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    for r in rows:
        r["index"] = int(r["index"])
        r["score"] = float(r["public_score"])
    return rows


def score_of(rows: list[dict], label: str) -> float:
    return next(r["score"] for r in rows if r["label"] == label)


# ------------------------------------------- additive score decomposition --
# The four tasks are scored on disjoint rows, so
#   S_total = B + 0.20*S_ODME + 0.15*IoU_onset + 0.15*IoU_ongoing,
#   B = 0.35*S_state + 0.15*S_physics.
# The organizer's baseline table (repo docs/BASELINES.md, validation split)
# anchors absolute values. Persistence predicts nothing on onset windows, so
# its S_queue 0.3017 is half of its ongoing IoU.
BASELINE = dict(S_state=0.6929, S_queue=0.3017, S_physics=0.3549, S_ODME=0.8359)


def decomposition(rows: list[dict]) -> dict:
    s = {r["label"]: r["score"] for r in rows}
    b0 = 0.35 * BASELINE["S_state"] + 0.15 * BASELINE["S_physics"]
    od = BASELINE["S_ODME"] + (s["T4-v1"] - s["BASE"]) / 0.20
    og_pers = 2 * BASELINE["S_queue"]
    b_v1 = s["P-persq"] - 0.20 * od - 0.15 * 0.0 - 0.15 * og_pers
    og_v1 = (s["P-noonset"] - b_v1 - 0.20 * od) / 0.15
    on_v1 = (s["V1"] - s["P-noonset"]) / 0.15
    on_u1 = (s["U1"] - s["P-persq"]) / 0.15
    og_u2 = og_pers + (s["U2"] - s["U1"]) / 0.15
    on_j1 = on_u1 + (s["J1"] - s["U2"]) / 0.15
    on_u3 = on_u1 + (s["U3"] - s["U2"]) / 0.15
    b_l0 = b_v1 + (s["L0"] - s["U2"])
    b_rs1 = b_v1 + (s["RC2"] - s["RC1"])
    og_rq2 = (s["CS1"] - b_rs1 - 0.20 * od - 0.15 * on_j1) / 0.15
    return dict(
        B_baseline=b0, ODME_baseline=BASELINE["S_ODME"], ODME_nnls=od,
        ongoing_persistence=og_pers, B_v1=b_v1, ongoing_v1=og_v1, onset_v1=on_v1,
        ongoing_v1_unfiltered=og_v1 + (s["V1-nh"] - s["V1"]) / 0.15,
        ongoing_slotT=og_v1 + (s["P-slotT"] - s["V1"]) / 0.15,
        onset_u1=on_u1, ongoing_u2=og_u2, onset_j1=on_j1, onset_u3=on_u3,
        B_interp=b_l0, B_rs1=b_rs1, ongoing_rq2=og_rq2,
        ongoing_cs2=og_rq2 + (s["CS2"] - s["CS1"]) / 0.15,
        onset_cs3=on_j1 + (s["CS3"] - s["CS2"]) / 0.15,
        d_ongoing_cf2=(s["CF2"] - s["CF1"]) / 0.15,
        d_ongoing_cf3=(s["CF3"] - s["CF2"]) / 0.15,
        d_ongoing_es1=(s["ES1"] - s["TF3"]) / 0.15,
        queue_sum_rc1=(s["RC1"] - b_v1 - 0.20 * od) / 0.15,
        check_j1=b_v1 + 0.20 * od + 0.15 * on_j1 + 0.15 * og_u2,
    )


# --------------------------------------------------------------- figures --
def fig_leaderboard_timeline(rows: list[dict]) -> None:
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(6.5, 2.75), gridspec_kw=dict(width_ratios=[1, 1.55], wspace=0.22))
    x = np.array([r["index"] for r in rows])
    y = np.array([r["score"] for r in rows])
    cat = [r["category"] for r in rows]
    legal = np.array([c in ("candidate", "selected", "anchor") for c in cat])
    best = np.maximum.accumulate(np.where(legal, y, 0))

    def draw(ax, lo, hi, labels):
        ax.step(x, best, where="post", color=BLUE, lw=1.2, alpha=0.55, zorder=1)
        for xi, yi, c in zip(x, y, cat):
            if not lo <= yi <= hi:
                continue
            if c == "selected":
                ax.scatter(xi, yi, s=58, color=BLUE, edgecolor=SURFACE, linewidth=1.4, zorder=4)
                ax.scatter(xi, yi, s=150, facecolor="none", edgecolor=BLUE, linewidth=0.9, zorder=3)
            elif c == "candidate":
                ax.scatter(xi, yi, s=26, color=BLUE, edgecolor=SURFACE, linewidth=1.0, zorder=3)
            else:
                ax.scatter(xi, yi, s=24, color=GREY, edgecolor=SURFACE, linewidth=1.0, zorder=3, marker="D" if c == "diagnostic" else "o")
        for lab, dx, dy, ha in labels:
            r = next(r for r in rows if r["label"] == lab)
            ax.annotate(lab, (r["index"], r["score"]), xytext=(dx, dy), textcoords="offset points",
                        fontsize=7, color=INK2, ha=ha, va="center")
        ax.set_ylim(lo, hi)
        ax.set_xlabel("Submission number (chronological)")

    draw(a1, 0.45, 0.89, [("BASE", 6, -2, "left"), ("T4-v1", 5, 4, "left"), ("V1", -4, 9, "right"),
                           ("P-persq", 6, -1, "left")])
    a1.set_xlim(0, 32)
    a1.set_ylabel("Kaggle public score")
    a1.set_title("All 31 submissions")
    a1.axhspan(0.79, 0.89, color=GRID, alpha=0.35, lw=0, zorder=0)

    draw(a2, 0.79, 0.877, [("U1", -5, -8, "right"), ("U2", 7, -4, "left"), ("J1", -6, 6, "right"), ("U3", 6, 0, "left"),
                            ("L0", 6, -1, "left"), ("P-slotT", 6, 2, "left"), ("RC1", 6, 0, "left"), ("CS1", -6, 4, "right"),
                            ("CS3", 6, -2, "left"), ("CT2", -6, 5, "right"), ("CF1", -6, 5, "right"), ("CF2", -6, 6, "right"),
                            ("TF3", 0, -11, "center"), ("ES4", -7, 8, "right"), ("ES5", 4, -8, "left")])
    a2.set_xlim(9.2, 32)
    a2.set_title("Zoom: the legal pipeline, 1 to 10 October")
    a2.set_yticks([0.80, 0.82, 0.84, 0.86, 0.88])

    handles = [
        Line2D([], [], color=BLUE, marker="o", ls="", ms=5, mec=SURFACE, label="Candidate pipeline"),
        Line2D([], [], color=BLUE, marker="o", ls="", ms=6.5, mec=BLUE, mfc=BLUE, label="Selected final (TF3, ES4)"),
        Line2D([], [], color=GREY, marker="o", ls="", ms=5, mec=SURFACE, label="Probe / anchor"),
        Line2D([], [], color=GREY, marker="D", ls="", ms=4.5, mec=SURFACE, label="Diagnostic (reads slot T; never selected)"),
        Line2D([], [], color=BLUE, lw=1.2, alpha=0.55, label="Best legal score so far"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.17), handletextpad=0.4, columnspacing=1.2)
    save(fig, "fig_leaderboard_timeline")


def fig_waterfall(rows: list[dict]) -> None:
    s = {r["label"]: r["score"] for r in rows}
    steps = [
        ("Organizer baselines (anchor)", s["BASE"], "base"),
        ("Exact split-specific NNLS demand", s["T4-v1"] - s["BASE"], "demand"),
        ("Residual GBDT state reconstruction", s["P-persq"] - s["T4-v1"], "state"),
        ("Onset: legal classifier (U1)", s["U1"] - s["P-persq"], "queue"),
        ("Ongoing: legal forward-fill (U2)", s["U2"] - s["U1"], "queue"),
        ("Onset: joint-mask analogue (J1)", s["J1"] - s["U2"], "queue"),
        ("RS1 state model (RC2 - RC1)", s["RC2"] - s["RC1"], "state"),
        ("RQ2 ongoing analogue decoder (CS1)", s["CS1"] - (s["J1"] + s["RC2"] - s["RC1"]), "queue"),
        ("S5 speed + Q19 ongoing (CT1)", s["CT1"] - s["CS1"], "mixed"),
        ("S10/S12 factor + covariance (CT2)", s["CT2"] - s["CT1"], "state"),
        ("FS1 gate + Q46 first-crossing (CF1)", s["CF1"] - s["CT2"], "mixed"),
        ("RQ48 quarter-prior decoder (CF2)", s["CF2"] - s["CF1"], "queue"),
        ("Three-member sample ensemble (TF3)", s["TF3"] - s["CF2"], "state"),
        ("All-month ongoing classifier (ES1)", s["ES1"] - s["TF3"], "queue"),
        ("Gap-1 speed recentering (ES2)", s["ES2"] - s["ES1"], "state"),
        ("Gap-2 speed recentering (ES3)", s["ES3"] - s["ES2"], "state"),
        ("Gap 3-6 speed recentering (ES4)", s["ES4"] - s["ES3"], "state"),
    ]
    colors = dict(base=GREY, demand=AQUA, state=BLUE, queue=ORANGE, mixed=LIGHT)
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    level = 0.0
    n = len(steps)
    for i, (name, delta, kind) in enumerate(steps):
        yy = n - 1 - i
        if kind == "base":
            ax.barh(yy, delta, left=0, height=0.62, color=colors[kind], lw=0)
            level = delta
            ax.text(delta + 0.004, yy, f"{delta:.5f}", va="center", fontsize=7.2, color=INK2)
        else:
            ax.barh(yy, delta, left=level, height=0.62, color=colors[kind], lw=0)
            level += delta
            ax.text(max(level, level - delta) + 0.004, yy, f"+{delta:.5f}" if delta >= 0 else f"{delta:.5f}",
                    va="center", fontsize=7.2, color=INK2)
    ax.barh(-1.25, level, height=0.62, color=BLUE, lw=0)
    ax.text(level + 0.004, -1.25, f"{level:.5f}  (ES4, Kaggle public)", va="center", fontsize=7.4, color=INK, fontweight="bold")
    ax.set_yticks(list(range(n - 1, -1, -1)) + [-1.25])
    ax.set_yticklabels([name for name, _, _ in steps] + ["Selected final ES4"], fontsize=7.6)
    ax.set_xlim(0.50, 0.905)
    ax.set_ylim(-1.9, n - 0.4)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Kaggle public score (each bar starts where the previous one ends)")
    ax.set_title("From the organizer baselines to ES4, one public measurement per step")
    handles = [Patch(color=colors[k], label=l) for k, l in
               (("state", "State / physics"), ("queue", "Queue"), ("demand", "Demand (ODME)"), ("mixed", "Two components at once"))]
    ax.legend(handles=handles, loc="lower left", ncol=1, bbox_to_anchor=(0.01, 0.1))
    save(fig, "fig_waterfall")


def fig_lb_distribution(rows: list[dict]) -> None:
    with open(DATA / "public_leaderboard_scores_2026-11-05.csv", encoding="utf-8") as fh:
        lb = list(csv.DictReader(fh))
    scores = np.array([float(r["public_score"]) for r in lb])
    ours = next(r for r in lb if r["is_powerbuff_girls"] == "1")
    ours_score, ours_rank = float(ours["public_score"]), int(ours["rank"])
    top10 = np.sort(scores)[::-1][9]
    fig, ax = plt.subplots(figsize=(6.5, 2.35))
    bins = np.arange(0.70, 0.94, 0.005)
    clipped = np.clip(scores, 0.70, 0.9399)
    ax.hist(clipped, bins=bins, color=LIGHT, edgecolor=SURFACE, linewidth=0.8)
    ax.axvspan(ours_score - 0.0126, ours_score + 0.0126, color=BLUE, alpha=0.08, lw=0)
    ymax = ax.get_ylim()[1]
    for xv, lab, col, dy in ((np.median(scores), f"median {np.median(scores):.4f}", GREY, -0.93),
                             (top10, f"10th place {top10:.4f}", GREY, 0.93),
                             (scores.max(), f"leader {scores.max():.4f}", GREY, 0.70)):
        ax.axvline(xv, color=col, lw=0.9)
        if dy < 0:
            ax.text(xv - 0.0012, ymax * -dy, lab, fontsize=7, color=INK2, va="top", ha="right")
        else:
            ax.text(xv + 0.0012, ymax * dy, lab, fontsize=7, color=INK2, va="top")
    ax.axvline(ours_score, color=BLUE, lw=1.6)
    ax.text(ours_score + 0.0012, ymax * 0.62, f"ES4 {ours_score:.5f}\nrank {ours_rank} of {len(scores)}",
            fontsize=7.2, color=INK, va="top", fontweight="bold")
    ax.set_xlim(0.70, 0.94)
    ax.set_xlabel("Public score (scores below 0.70 are pooled into the first bin)")
    ax.set_ylabel("Teams")
    ax.set_title("Public leaderboard, 5 November 2026 12:26 UTC")
    note(ax, "Shaded band: plus or minus 0.0126, the organizers' standard deviation of S_total from queue-window sampling alone.", y=-0.27)
    save(fig, "fig_lb_distribution")


def fig_state_regimes() -> None:
    # RESEARCH_20261004_SECOND.md 14.351: RS1 development model (fit June-November),
    # February targets held out; equal weight over the cells of each regime.
    regimes = ["R1 (20 %)", "R2 (30 %)", "R3 (50 %)"]
    cells = [555_352, 902_718, 1_985_874]
    v_model, v_interp = [1.460, 1.553, 1.661], [2.748, 2.839, 2.975]
    q_model, q_interp = [30.4, 30.9, 31.1], [44.5, 46.2, 47.6]
    p_model, p_interp = [0.9451, 0.9428, 0.9403], [0.9066, 0.9032, 0.8993]
    fig, axes = plt.subplots(1, 3, figsize=(6.5, 2.3), gridspec_kw=dict(wspace=0.42))
    x = np.arange(3)
    w = 0.34
    for ax, (m, i, title, unit, fmt) in zip(axes, (
            (v_model, v_interp, "Speed RMSE", "km/h", "{:.2f}"),
            (q_model, q_interp, "Per-lane flow RMSE", "veh/h/lane", "{:.1f}"),
            (p_model, p_interp, "State piece S_state(r)", "score", "{:.3f}"))):
        ax.bar(x - w / 2 - 0.01, i, w, color=ORANGE, lw=0, label="Time interpolation")
        ax.bar(x + w / 2 + 0.01, m, w, color=BLUE, lw=0, label="Residual GBDT (RS1)")
        for xx, vv in zip(x, m):
            ax.text(xx + w / 2 + 0.01, vv, fmt.format(vv), ha="center", va="bottom", fontsize=6.6, color=INK2)
        for xx, vv in zip(x, i):
            ax.text(xx - w / 2 - 0.01, vv, fmt.format(vv), ha="center", va="bottom", fontsize=6.6, color=INK2)
        ax.set_xticks(x)
        ax.set_xticklabels(regimes, fontsize=7.2)
        ax.set_title(title)
        ax.set_ylabel(unit)
        ax.grid(axis="x", visible=False)
    axes[2].set_ylim(0.85, 0.96)
    axes[0].set_ylim(0, 3.4)
    axes[1].set_ylim(0, 55)
    h, lab = axes[0].get_legend_handles_labels()
    fig.legend(h, lab, loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.1))
    fig.text(0.0, -0.17, f"February 2031 targets held out from a model fitted on June to November 2030; "
             f"{sum(cells):,} cells (R1 {cells[0]:,}, R2 {cells[1]:,}, R3 {cells[2]:,}). TRAIN proxy, not a Kaggle score.",
             fontsize=7, color=MUTED)
    save(fig, "fig_state_regimes")


def fig_state_gaps() -> None:
    # 14.347: February official targets with both temporal neighbours.
    groups = ["1", "2", "3-6", "7-12", "13-30"]
    n = [804_794, 963_903, 1_371_721, 214_200, 34_936]
    model = [1.408, 1.519, 1.696, 1.972, 1.987]
    interp = [2.365, 2.677, 3.162, 3.815, 3.976]
    fig, ax = plt.subplots(figsize=(6.5, 2.35))
    x = np.arange(len(groups))
    ax.plot(x, interp, color=ORANGE, lw=2, marker="o", ms=5, mec=SURFACE, mew=1.2, label="Time interpolation")
    ax.plot(x, model, color=BLUE, lw=2, marker="o", ms=5, mec=SURFACE, mew=1.2, label="Residual GBDT (RS1)")
    ax.text(x[-1] + 0.08, interp[-1], f"{interp[-1]:.2f}", va="center", fontsize=7, color=INK2)
    ax.text(x[-1] + 0.08, model[-1], f"{model[-1]:.2f}", va="center", fontsize=7, color=INK2)
    ax.text(x[0] - 0.08, interp[0], f"{interp[0]:.2f}", va="center", ha="right", fontsize=7, color=INK2)
    ax.text(x[0] - 0.08, model[0], f"{model[0]:.2f}", va="center", ha="right", fontsize=7, color=INK2)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{g} slot{'' if g == '1' else 's'}\n{c:,} cells" for g, c in zip(groups, n)], fontsize=7.2)
    ax.set_xlim(-0.5, len(groups) - 0.4)
    ax.set_ylim(0, 4.6)
    ax.set_ylabel("Speed RMSE (km/h)")
    ax.set_xlabel("Distance to the farther observed temporal neighbour")
    ax.set_title("The learned residual halves interpolation error at every gap length")
    ax.legend(loc="upper left")
    save(fig, "fig_state_gaps")


def fig_gap_ladder(ladder_path: Path) -> None:
    # 14.316 (gap 1, about 0.02), 14.322 (gap 2), 14.323 (gaps 3 to 25-48).
    shells = ["1", "2", "3", "4", "5", "6", "7", "8", "9-12", "13-24", "25-48"]
    bias = [0.02, 0.02774, 0.039, 0.059, 0.082, 0.108, 0.143, 0.184, 0.196, 0.309, 1.451]
    written = {"1": "ES2", "2": "ES3", "3": "ES4", "4": "ES4", "5": "ES4", "6": "ES4"}
    fig, ax = plt.subplots(figsize=(6.5, 2.55))
    x = np.arange(len(shells))
    for xi, sh, b in zip(x, shells, bias):
        col = BLUE if sh in written else LIGHT
        ax.bar(xi - 0.1, min(b, 0.42), 0.46, color=col, lw=0, zorder=2)
        txt = f"{b:.3f}" if b < 0.42 else f"{b:.2f}*"
        ax.text(xi - 0.1, min(b, 0.42) + 0.008, ("~" if sh == "1" else "") + txt, ha="center", va="bottom", fontsize=6.6, color=INK2)
        if sh in written:
            ax.text(xi, -0.05, written[sh], ha="center", va="top", fontsize=6.8, color=BLUE, fontweight="bold")
    if ladder_path.is_file():
        d = json.loads(ladder_path.read_text())
        key = {"g3": 2, "g4": 3, "g5": 4, "g6": 5, "g7": 6, "g8": 7, "g9_12": 8, "g13_24": 9, "g25_48": 10}
        rng = np.random.default_rng(7)
        for r in d["february"]:
            if r["shell"] in key and r["mean_abs_bias"] is not None:
                ax.scatter(key[r["shell"]] + 0.27 + rng.uniform(-0.05, 0.05), min(r["mean_abs_bias"], 0.42), s=8, color=INK2,
                           alpha=0.55, lw=0, zorder=3)
    ax.axhline(0.15, color=INK, lw=0.9, zorder=4)
    ax.set_xticks(x)
    ax.set_xticklabels(shells)
    ax.set_ylim(-0.085, 0.46)
    ax.set_xlim(-0.6, len(shells) - 0.4)
    ax.grid(axis="x", visible=False)
    ax.set_xlabel("Gap shell: distance in 5-minute slots to the farther observed temporal neighbour")
    ax.set_ylabel("Mean |link bias| (km/h)")
    ax.set_title("February link bias of the same-day interpolant, by gap shell")
    handles = [Patch(color=BLUE, label="Written to March/April (submission named below)"),
               Patch(color=LIGHT, label="Not written"),
               Line2D([], [], color=INK, lw=0.9, label="Acceptance bar, 0.15 km/h"),
               Line2D([], [], color=INK2, marker="o", ls="", ms=3, alpha=0.6, label="One corridor panel")]
    ax.legend(handles=handles, loc="upper left")
    note(ax, "Gap 7 passed the February test but no March or April link reached 100 cells; gaps of 8 or more failed it. "
            "* clipped at 0.42.", y=-0.3)
    save(fig, "fig_gap_ladder")


def fig_physics_alternatives() -> None:
    # 14.318 and 14.319: D12_I5_N, October-February targets, 2,583,593 cells.
    a_lab = ["Time interpolation", "Corridor (spatial) interpolation", "Godunov LWR, 0.245 s step",
             "Triangular-diagram inversion of density"]
    a_val = [0.90445, 0.83075, 0.81273, 0.74493]
    # 14.352: February, all ten panels.
    b_lab = ["Deployed reconstruction", "Flow nudge to Newell envelope", "Congested-cell Newell blend",
             "Full Newell lower envelope"]
    b_val = [0.9456, 0.9408, 0.8292, 0.4248]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(6.5, 2.2), gridspec_kw=dict(wspace=1.15))
    for ax, lab, val, title in ((a1, a_lab, a_val, "D12_I5_N, Oct-Feb held out"),
                                (a2, b_lab, b_val, "February, ten panels")):
        y = np.arange(len(lab))[::-1]
        cols = [BLUE] + [ORANGE] * (len(lab) - 1)
        ax.barh(y, val, 0.55, color=cols, lw=0)
        for yy, v in zip(y, val):
            ax.text(v + 0.01, yy, f"{v:.4f}", va="center", fontsize=7, color=INK2)
        ax.set_yticks(y)
        ax.set_yticklabels(lab, fontsize=7.4)
        ax.set_xlim(0.3, 1.06)
        ax.set_xlabel("State piece (higher is better)")
        ax.set_title(title)
        ax.grid(axis="y", visible=False)
    save(fig, "fig_physics_alternatives")


def fig_queue_ceiling() -> None:
    # 14.328 (published TRAIN windows, latent proxy), 14.330 and 14.334 (odd March dates, visible proxy).
    groups = ["40 ongoing windows\n(published TRAIN)", "40 onset windows\n(published TRAIN)", "146 ongoing windows\n(visible March, odd dates)"]
    series = [
        ("Persistence of the last legal observation", ORANGE, [0.7551, 0.0250, 0.7184]),
        ("Analogue choice alone", GREY, [0.8357, 0.9375, 0.7178]),
        ("Deployed decoder (classifier + analogue)", BLUE, [np.nan, np.nan, 0.8129]),
        ("Best action in the candidate set (oracle)", AQUA, [0.9343, 1.0000, 0.8748]),
    ]
    fig, ax = plt.subplots(figsize=(6.5, 2.5))
    x = np.arange(len(groups))
    w = 0.19
    for k, (lab, col, vals) in enumerate(series):
        off = (k - 1.5) * (w + 0.015)
        for xi, v in zip(x, vals):
            if np.isnan(v):
                continue
            ax.bar(xi + off, v, w, color=col, lw=0)
            ax.text(xi + off, v + 0.012, f"{v:.3f}", ha="center", va="bottom", fontsize=6.4, color=INK2, rotation=0)
        ax.bar(0, 0, 0, color=col, label=lab)
    ax.set_xticks(x)
    ax.set_xticklabels(groups, fontsize=7.4)
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("Mean IoU (proxy)")
    ax.grid(axis="x", visible=False)
    ax.set_title("Queue forecasting: what the rule picks versus the best candidate it already holds")
    ax.legend(loc="upper center", ncol=2, bbox_to_anchor=(0.5, -0.24))
    save(fig, "fig_queue_ceiling")


def fig_odme(rows: list[dict]) -> None:
    s = {r["label"]: r["score"] for r in rows}
    base = BASELINE["S_ODME"]
    items = [
        ("NNLS, λ = 0.05, split's own counts (T4-v1)", s["T4-v1"]),
        ("KL / entropy fit on the raw prior (T4-v4)", s["T4-v4"]),
        ("Organizer baseline, TRAIN counts", s["BASE"]),
        ("KL from a fitted gravity seed (T4-v5)", s["T4-v5"]),
        ("Prior × smooth correction (T4-v2)", s["T4-v2"]),
    ]
    vals = [base + (v - s["BASE"]) / 0.20 for _, v in items]
    fig, ax = plt.subplots(figsize=(6.5, 1.95))
    y = np.arange(len(items))[::-1]
    cols = [BLUE] + [ORANGE if "baseline" in n else LIGHT for n, _ in items[1:]]
    ax.barh(y, vals, 0.55, color=cols, lw=0)
    for yy, v, (_, tot) in zip(y, vals, items):
        ax.text(v + 0.006, yy, f"{v:.3f}   (public total {tot:.5f})", va="center", fontsize=7, color=INK2)
    ax.set_yticks(y)
    ax.set_yticklabels([n for n, _ in items], fontsize=7.4)
    ax.set_xlim(0.4, 1.12)
    ax.set_xlabel("Implied S_ODME = 0.8359 + (public total - 0.55343) / 0.20")
    ax.set_title("Task 4 probes: only the demand rows differ from the organizer baseline")
    ax.grid(axis="y", visible=False)
    save(fig, "fig_odme")


def fig_transfer(rows: list[dict]) -> None:
    s = {r["label"]: r["score"] for r in rows}
    state = [
        ("RS1 state model (RC2 vs RC1)", s["RC2"] - s["RC1"]),
        ("S10/S12 corrections (CT2 vs CT1)", s["CT2"] - s["CT1"]),
        ("Sample ensemble (TF3 vs CF2)", s["TF3"] - s["CF2"]),
        ("Gap-1 recentering (ES2 vs ES1)", s["ES2"] - s["ES1"]),
        ("Gap-2 recentering (ES3 vs ES2)", s["ES3"] - s["ES2"]),
        ("Gap 3-6 recentering (ES4 vs ES3)", s["ES4"] - s["ES3"]),
        ("March-fit speed model (ES5 vs ES4)*", s["ES5"] - s["ES4"]),
    ]
    queue = [
        ("Ongoing unfiltered tree (V1-nh vs V1)", s["V1-nh"] - s["V1"]),
        ("Ongoing forward-fill (U2 vs U1)", s["U2"] - s["U1"]),
        ("Onset analogue (J1 vs U2)", s["J1"] - s["U2"]),
        ("Onset frequency prior (U3 vs U2)", s["U3"] - s["U2"]),
        ("RQ7/RQ12 queues (RC1 vs J1)", s["RC1"] - s["J1"]),
        ("Ongoing origin analogue (CS2 vs CS1)", s["CS2"] - s["CS1"]),
        ("Onset demand completion (CS3 vs CS2)", s["CS3"] - s["CS2"]),
        ("RQ48 decoder (CF2 vs CF1)", s["CF2"] - s["CF1"]),
        ("Occupancy classifier (CF3 vs CF2)", s["CF3"] - s["CF2"]),
        ("All-month classifier (ES1 vs TF3)", s["ES1"] - s["TF3"]),
    ]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(6.5, 3.0), gridspec_kw=dict(wspace=1.25))
    for ax, items, title in ((a1, state, "State-only changes"), (a2, queue, "Queue-only changes")):
        y = np.arange(len(items))[::-1]
        vals = np.array([v for _, v in items]) * 1000
        ax.barh(y, vals, 0.58, color=[BLUE if v >= 0 else RED for v in vals], lw=0)
        ax.axvline(0, color=AXIS, lw=0.8)
        for yy, v in zip(y, vals):
            ax.text(v + (0.6 if v >= 0 else -0.6), yy, f"{v:+.2f}", va="center", ha="left" if v >= 0 else "right",
                    fontsize=6.6, color=INK2)
        ax.set_yticks(y)
        ax.set_yticklabels([n for n, _ in items], fontsize=7.1)
        ax.set_xlabel("Change in public score (× 1000)")
        ax.set_title(title)
        ax.grid(axis="y", visible=False)
    a1.set_xlim(-11, 16)
    a2.set_xlim(-46, 32)
    fig.text(0.0, -0.07, "* ES5 is the only state change selected on a visible-cell proxy of the evaluation month (fit on odd March days, judged on even days); "
             "the others were selected on held-out TRAIN months.",
             fontsize=7, color=MUTED)
    save(fig, "fig_transfer")


def fig_fundamental_diagram(network_root: Path | None) -> None:
    import pandas as pd

    if network_root is None or not (network_root / "corridors").is_dir():
        print("skip fig_fundamental_diagram: no --network-root with corridors/")
        return
    frames = []
    for panel_dir in sorted((network_root / "corridors").iterdir()):
        path = panel_dir / "network" / "fd_parameters.csv"
        if path.is_file():
            fd = pd.read_csv(path).drop_duplicates("link_id")
            fd["district"] = panel_dir.name.split("_")[0]
            frames.append(fd)
    fd = pd.concat(frames, ignore_index=True)
    fd["cap"] = fd.capacity_vph / fd.lanes
    fd["kc"] = fd.critical_density / fd.lanes
    fd["kj"] = fd.k_jam / fd.lanes
    outlier = fd.cap > 3000
    keep = fd[~outlier]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(6.5, 2.45), gridspec_kw=dict(wspace=0.32, width_ratios=[1.3, 1]))
    for r in keep.itertuples():
        a1.plot([0, r.kc, r.kj], [0, r.cap, 0], color=GREY, lw=0.4, alpha=0.06)
    for d, col in (("D7", BLUE), ("D12", ORANGE)):
        m = keep[keep.district == d]
        vf, cap, kj = m.free_speed_kmh.median(), m.cap.median(), m.kj.median()
        kc = cap / vf
        a1.plot([0, kc, kj], [0, cap, 0], color=col, lw=2, solid_joinstyle="round",
                label=f"District {d[1:]} median ({len(m):,} links): v_f {vf:.0f} km/h, capacity {cap:,.0f} veh/h/lane, "
                      f"w {cap / (kj - kc):.1f} km/h")
        a2.scatter(m.free_speed_kmh, m.cap, s=7, color=col, alpha=0.45, lw=0)
    a1.set_xlim(0, 104)
    a1.set_ylim(0, 3000)
    a1.set_xlabel("Density k (veh/km/lane)")
    a1.set_ylabel("Flow q (veh/h/lane)")
    a1.set_title(f"Per-lane triangular diagrams of {len(keep):,} links")
    a1.text(30, 2780, "jam density 100 veh/km/lane on every link", fontsize=7, color=INK2)
    a2.set_ylim(1100, 3000)
    a2.set_xlabel("Free-flow speed v_f (km/h)")
    a2.set_ylabel("Capacity (veh/h/lane)")
    a2.set_title("Link parameters")
    h, lab = a1.get_legend_handles_labels()
    fig.legend(h, lab, loc="lower center", ncol=1, bbox_to_anchor=(0.5, -0.27))
    fig.text(0.0, -0.31, f"Released fd_parameters.csv, ten panels. {int(outlier.sum())} links whose stated capacity exceeds "
             "3,000 veh/h/lane are omitted from both panels.", fontsize=7, color=MUTED)
    save(fig, "fig_fundamental_diagram")


# -------------------------------------------------------- generated TeX --
def tex_escape(text: str) -> str:
    return text.replace("_", r"\_").replace("%", r"\%").replace("&", r"\&").replace("#", r"\#")


def write_tables(rows: list[dict], comp: dict) -> None:
    GEN.mkdir(parents=True, exist_ok=True)
    short = {
        "BASE": "Organizer baselines for all tasks", "T4-v2": "ODME: prior $\\times$ smooth correction",
        "T4-v4": "ODME: KL/entropy on raw prior", "T4-v5": "ODME: KL from gravity seed",
        "T4-v1": "ODME: exact NNLS on own split", "V1": "GBDT state, LightGBM queues",
        "P-persq": "V1 state + persistence queue", "P-noonset": "V1 with onset zeroed",
        "V1-nh": "V1, ongoing tree on unfiltered windows", "P-slotT": "Ongoing = state at slot $T$ (diagnostic)",
        "U1": "Legal onset classifier (12 slots)", "U2": "U1 + ongoing forward-fill",
        "J1": "U2 + joint-mask onset analogue", "L0": "U2 with time-interpolated state",
        "U3": "U2 + onset frequency prior", "RC1": "RQ7 ongoing + RQ12 onset", "RC2": "RC1 + RS1 state",
        "CS1": "RS1 state + RQ2 ongoing + J1 onset", "CS2": "CS1, ongoing origin analogue",
        "CS3": "CS2 + onset demand completion", "CT1": "S5 speed + Q19 ongoing", "CT2": "CT1 + S10/S12 state",
        "CF1": "FS1 gate + Q46 ongoing", "CF2": "CF1 + RQ48 decoder", "CF3": "CF2 + occupancy classifier",
        "TF3": "CF2 + 3-member sample ensemble", "ES1": "TF3 + all-month ongoing classifier",
        "ES2": "ES1 + gap-1 recentering", "ES3": "ES2 + gap-2 recentering", "ES4": "ES3 + gap 3--6 recentering",
        "ES5": "ES4 + March-fit speed model",
    }
    lines = [r"\begin{tabular}{@{}r l l l r r@{}}", r"\toprule",
             r"\# & ID & Content & Submitted (UTC) & Kaggle ref & Public \\", r"\midrule"]
    prev_day = None
    for r in rows:
        day = r["submitted_utc"][:10]
        if prev_day and day != prev_day:
            lines.append(r"\addlinespace[2pt]")
        prev_day = day
        lab = r["label"]
        score = f"{r['score']:.5f}"
        if r["category"] == "selected":
            lab, score = rf"\textbf{{{lab}}}", rf"\textbf{{{score}}}"
        when = r["submitted_utc"][5:16].replace("T", " ")
        lines.append(f"{r['index']} & {lab} & {short[r['label']]} & {when} & {r['ref']} & {score} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (GEN / "tab_submissions.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")

    def f(v, d=4):
        return f"{v:.{d}f}"
    comp_rows = [
        ("Demand", r"$S_{\mathrm{ODME}}$, organizer baseline (TRAIN counts)", f(comp["ODME_baseline"]), "documented"),
        ("Demand", r"$S_{\mathrm{ODME}}$, exact NNLS on the split's own counts", f(comp["ODME_nnls"]), "T4-v1 $-$ BASE"),
        ("State", r"$B$, weekday--slot historical mean (baseline)", f(comp["B_baseline"]), "documented"),
        ("State", r"$B$, time interpolation", f(comp["B_interp"]), "L0 $-$ U2"),
        ("State", r"$B$, first residual GBDT (V1)", f(comp["B_v1"]), "P-persq"),
        ("State", r"$B$, RS1 residual GBDT", f(comp["B_rs1"]), "RC2 $-$ RC1"),
        ("Onset", "IoU, persistence (predicts nothing)", "$\\approx 0$", "by construction"),
        ("Onset", r"IoU, first LightGBM, 13 slots incl.\ $T$ (V1)", f(comp["onset_v1"]), "V1 $-$ P-noonset"),
        ("Onset", "IoU, legal 12-slot classifier (U1)", f(comp["onset_u1"]), "U1 $-$ P-persq"),
        ("Onset", "IoU, train frequency prior (U3)", f(comp["onset_u3"]), "U3 $-$ U2"),
        ("Onset", "IoU, demand-history completion (CS3)", f(comp["onset_cs3"]), "CS3 $-$ CS2"),
        ("Onset", r"IoU, joint-mask analogue (J1, kept to ES4)", f(comp["onset_j1"]), "J1 $-$ U2"),
        ("Ongoing", "IoU, organizer persistence (eligible cells)", f(comp["ongoing_persistence"]), "documented"),
        ("Ongoing", r"IoU, first LightGBM, 13 slots incl.\ $T$ (V1)", f(comp["ongoing_v1"]), "P-noonset"),
        ("Ongoing", "IoU, same, trained on unfiltered windows", f(comp["ongoing_v1_unfiltered"]), "V1-nh $-$ V1"),
        ("Ongoing", r"IoU, state at slot $T$ (diagnostic only)", f(comp["ongoing_slotT"]), "P-slotT $-$ V1"),
        ("Ongoing", "IoU, forward-fill of the last legal value (U2)", f(comp["ongoing_u2"]), "U2 $-$ U1"),
        ("Ongoing", "IoU, RQ2 analogue decoder (CS1)", f(comp["ongoing_rq2"]), "CS1 identity"),
        ("Ongoing", "IoU, origin analogue (CS2)", f(comp["ongoing_cs2"]), "CS2 $-$ CS1"),
        ("Ongoing", r"$\Delta$IoU, RQ48 quarter-prior decoder", f"+{comp['d_ongoing_cf2']:.4f}", "CF2 $-$ CF1"),
        ("Ongoing", r"$\Delta$IoU, occupancy classifier RQ59", f"{comp['d_ongoing_cf3']:+.4f}", "CF3 $-$ CF2"),
        ("Ongoing", r"$\Delta$IoU, all-month classifier EQ1", f"+{comp['d_ongoing_es1']:.4f}", "ES1 $-$ TF3"),
    ]
    lines = [r"\begin{tabular}{@{}l l r l@{}}", r"\toprule",
             r"Task & Quantity & Value & Identified from \\", r"\midrule"]
    prev = None
    for task, q, v, src in comp_rows:
        if prev and task != prev:
            lines.append(r"\addlinespace[2pt]")
        lines.append(f"{task if task != prev else ''} & {q} & {v} & {src} \\\\")
        prev = task
    lines += [r"\bottomrule", r"\end{tabular}"]
    (GEN / "tab_components.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (GEN / "components.json").write_text(json.dumps(comp, indent=1), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--network-root", type=Path, default=ROOT / "data" / "small",
                        help="Release directory containing corridors/<PANEL>/network (optional)")
    parser.add_argument("--gap-ladder", type=Path, default=ROOT / "deliverables" / "ES4_20261009" / "gap_ladder.json")
    args = parser.parse_args()
    setup_style()
    rows = load_submissions()
    comp = decomposition(rows)
    print("decomposition check: J1 rebuilt", round(comp["check_j1"], 5), "Kaggle", score_of(rows, "J1"))
    for k, v in comp.items():
        print(f"  {k:24s} {v:.5f}")
    write_tables(rows, comp)
    fig_leaderboard_timeline(rows)
    fig_waterfall(rows)
    fig_lb_distribution(rows)
    fig_state_regimes()
    fig_state_gaps()
    fig_gap_ladder(args.gap_ladder)
    fig_physics_alternatives()
    fig_queue_ceiling()
    fig_odme(rows)
    fig_transfer(rows)
    fig_fundamental_diagram(args.network_root)


if __name__ == "__main__":
    main()
