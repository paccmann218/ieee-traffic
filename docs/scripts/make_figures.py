"""Regenerate the figures and generated tables of the final report and README.

    python docs/scripts/make_figures.py [--network-root PATH] [--gap-ladder PATH]

Inputs
    docs/data/submissions.csv      the team's Kaggle submissions (public scores only)
    --network-root (optional)      a copy of the competition release containing
                                   corridors/<PANEL>/network/; used only for the
                                   fundamental-diagram figure
    --gap-ladder (optional)        per-panel February gap-shell statistics

Every other number is a recorded measurement, and each constant names the research-ledger
entry it comes from. Local numbers are TRAIN or visible-cell proxies; only the submission
file holds Kaggle scores.

Outputs
    docs/report/figures/*.pdf      vector figures for the report
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

# Restrained palette: one accent, two greys, one colour for losses.
NAVY = "#1f3b5c"
GREY = "#8c8c8c"
LIGHT = "#d9d9d9"
RED = "#9e3a38"
BLACK = "#000000"
FULL_WIDTH = 6.3


# ------------------------------------------------------------------ style
def _register_fonts() -> str:
    """Use Libertinus Serif, the body face of the report, when TeX can locate it."""
    kpse = shutil.which("kpsewhich")
    found = False
    for name in ("LibertinusSerif-Regular.otf", "LibertinusSerif-Italic.otf", "LibertinusSerif-Bold.otf"):
        if not kpse:
            break
        try:
            path = subprocess.run([kpse, name], capture_output=True, text=True, timeout=30).stdout.strip()
        except (OSError, subprocess.SubprocessError):
            path = ""
        if path and Path(path).is_file():
            font_manager.fontManager.addfont(path)
            found = True
    return "Libertinus Serif" if found else "DejaVu Serif"


def setup_style() -> None:
    plt.rcParams.update({
        "font.family": _register_fonts(),
        "mathtext.fontset": "stix",
        "font.size": 9,
        "axes.labelsize": 9,
        "axes.titlesize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "axes.linewidth": 0.6,
        "axes.edgecolor": BLACK,
        "axes.labelcolor": BLACK,
        "axes.grid": False,
        "axes.spines.top": True,
        "axes.spines.right": True,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.top": True,
        "ytick.right": True,
        "xtick.major.size": 3,
        "ytick.major.size": 3,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.color": BLACK,
        "ytick.color": BLACK,
        "legend.frameon": False,
        "legend.handlelength": 1.6,
        "lines.linewidth": 1.2,
        "lines.markersize": 4,
        "hatch.linewidth": 0.5,
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
        "pdf.fonttype": 42,
    })


def save(fig, name: str) -> None:
    FIG_PDF.mkdir(parents=True, exist_ok=True)
    FIG_PNG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_PDF / f"{name}.pdf", bbox_inches="tight", pad_inches=0.02)
    fig.savefig(FIG_PNG / f"{name}.png", dpi=220, bbox_inches="tight", pad_inches=0.06)
    plt.close(fig)
    print("wrote", name)


def panel(ax, letter: str) -> None:
    ax.text(0.0, 1.03, f"({letter})", transform=ax.transAxes, ha="left", va="bottom", fontsize=9)


# -------------------------------------------------------------- data load
def load_submissions() -> list[dict]:
    with open(DATA / "submissions.csv", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    for r in rows:
        r["index"] = int(r["index"])
        r["score"] = float(r["public_score"])
    return rows


# --------------------------------------------- additive score decomposition
# The four tasks are scored on disjoint rows, so
#   S_total = B + 0.20 S_ODME + 0.15 IoU_onset + 0.15 IoU_ongoing,   B = 0.35 S_state + 0.15 S_physics.
# The organizer baseline table (validation split) anchors absolute values. Persistence
# predicts nothing on onset windows, so its S_queue of 0.3017 is half its ongoing IoU.
BASELINE = dict(S_state=0.6929, S_queue=0.3017, S_physics=0.3549, S_ODME=0.8359)


def decomposition(rows: list[dict]) -> dict:
    s = {r["label"]: r["score"] for r in rows}
    b0 = 0.35 * BASELINE["S_state"] + 0.15 * BASELINE["S_physics"]
    od = BASELINE["S_ODME"] + (s["T4-v1"] - s["BASE"]) / 0.20
    og_pers = 2 * BASELINE["S_queue"]
    b_v1 = s["P-persq"] - 0.20 * od - 0.15 * og_pers
    og_v1 = (s["P-noonset"] - b_v1 - 0.20 * od) / 0.15
    on_u1 = (s["U1"] - s["P-persq"]) / 0.15
    og_u2 = og_pers + (s["U2"] - s["U1"]) / 0.15
    on_j1 = on_u1 + (s["J1"] - s["U2"]) / 0.15
    b_rs1 = b_v1 + (s["RC2"] - s["RC1"])
    og_rq2 = (s["CS1"] - b_rs1 - 0.20 * od - 0.15 * on_j1) / 0.15
    return dict(
        B_baseline=b0, ODME_baseline=BASELINE["S_ODME"], ODME_nnls=od, ongoing_persistence=og_pers,
        B_v1=b_v1, ongoing_v1=og_v1, onset_v1=(s["V1"] - s["P-noonset"]) / 0.15,
        ongoing_v1_unfiltered=og_v1 + (s["V1-nh"] - s["V1"]) / 0.15,
        ongoing_slotT=og_v1 + (s["P-slotT"] - s["V1"]) / 0.15,
        onset_u1=on_u1, ongoing_u2=og_u2, onset_j1=on_j1,
        onset_u3=on_u1 + (s["U3"] - s["U2"]) / 0.15,
        B_interp=b_v1 + (s["L0"] - s["U2"]), B_rs1=b_rs1, ongoing_rq2=og_rq2,
        ongoing_cs2=og_rq2 + (s["CS2"] - s["CS1"]) / 0.15,
        onset_cs3=on_j1 + (s["CS3"] - s["CS2"]) / 0.15,
        d_ongoing_cf2=(s["CF2"] - s["CF1"]) / 0.15,
        d_ongoing_cf3=(s["CF3"] - s["CF2"]) / 0.15,
        d_ongoing_es1=(s["ES1"] - s["TF3"]) / 0.15,
        check_j1=b_v1 + 0.20 * od + 0.15 * on_j1 + 0.15 * og_u2,
    )


def steps(rows: list[dict]) -> list[tuple[str, str, float]]:
    """The measured steps from the organizer baselines to ES4, each a difference of two public scores."""
    s = {r["label"]: r["score"] for r in rows}
    return [
        ("Exact split-specific NNLS demand", "Demand", s["T4-v1"] - s["BASE"]),
        ("Residual GBDT state reconstruction", "State", s["P-persq"] - s["T4-v1"]),
        ("Legal onset classifier (U1)", "Onset", s["U1"] - s["P-persq"]),
        ("Forward-filled ongoing persistence (U2)", "Ongoing", s["U2"] - s["U1"]),
        ("Joint-mask onset analogue (J1)", "Onset", s["J1"] - s["U2"]),
        ("RS1 state model (RC2 $-$ RC1)", "State", s["RC2"] - s["RC1"]),
        ("RQ2 ongoing analogue decoder (CS1, by additivity)", "Ongoing", s["CS1"] - (s["J1"] + s["RC2"] - s["RC1"])),
        ("S5 speed correction and Q19 features (CT1)", "State, ongoing", s["CT1"] - s["CS1"]),
        ("S10 factor flow and S12 covariance speed (CT2)", "State", s["CT2"] - s["CT1"]),
        ("FS1 gate and Q46 first-crossing classifier (CF1)", "State, ongoing", s["CF1"] - s["CT2"]),
        ("RQ48 quarter-prior decoder (CF2)", "Ongoing", s["CF2"] - s["CF1"]),
        ("Three-member sample ensemble (TF3)", "State", s["TF3"] - s["CF2"]),
        ("All-month ongoing classifier (ES1)", "Ongoing", s["ES1"] - s["TF3"]),
        ("Gap-1 speed recentering (ES2)", "State", s["ES2"] - s["ES1"]),
        ("Gap-2 speed recentering (ES3)", "State", s["ES3"] - s["ES2"]),
        ("Gap 3--6 speed recentering (ES4)", "State", s["ES4"] - s["ES3"]),
    ]


# ----------------------------------------------------------------- figures
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
    keep = fd[fd.cap <= 3000]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(FULL_WIDTH, 2.35), gridspec_kw=dict(wspace=0.3))
    for r in keep.itertuples():
        a1.plot([0, r.kc, r.kj], [0, r.cap, 0], color=GREY, lw=0.3, alpha=0.08)
    styles = {"D7": dict(color=NAVY, ls="-"), "D12": dict(color=BLACK, ls="--")}
    for d in ("D7", "D12"):
        m = keep[keep.district == d]
        vf, cap, kj = m.free_speed_kmh.median(), m.cap.median(), m.kj.median()
        a1.plot([0, cap / vf, kj], [0, cap, 0], lw=1.4, label=f"District {d[1:]} median", **styles[d])
    a1.set_xlim(0, 105)
    a1.set_ylim(0, 3000)
    a1.set_xlabel("Density (veh/km/lane)")
    a1.set_ylabel("Flow (veh/h/lane)")
    a1.legend(loc="upper right")
    panel(a1, "a")
    a2.scatter(keep[keep.district == "D7"].free_speed_kmh, keep[keep.district == "D7"].cap, s=5, marker="o",
               facecolor="none", edgecolor=NAVY, linewidth=0.5, label="District 7")
    a2.scatter(keep[keep.district == "D12"].free_speed_kmh, keep[keep.district == "D12"].cap, s=5, marker="^",
               facecolor="none", edgecolor=BLACK, linewidth=0.5, label="District 12")
    a2.set_ylim(1100, 3000)
    a2.set_xlabel("Free-flow speed (km/h)")
    a2.set_ylabel("Capacity (veh/h/lane)")
    a2.legend(loc="lower right", bbox_to_anchor=(1.0, 1.0), ncol=2, markerscale=1.8, handletextpad=0.2, borderaxespad=0.2)
    panel(a2, "b")
    save(fig, "fig_fundamental_diagram")


def fig_state_error() -> None:
    # RESEARCH_20261004_SECOND.md 14.351 (regimes) and 14.347 (gaps): February targets held out
    # from a model fitted on June to November.
    regimes = ["R1 (20%)", "R2 (30%)", "R3 (50%)"]
    v_model, v_interp = [1.460, 1.553, 1.661], [2.748, 2.839, 2.975]
    gaps = ["1", "2", "3–6", "7–12", "13–30"]
    g_model = [1.408, 1.519, 1.696, 1.972, 1.987]
    g_interp = [2.365, 2.677, 3.162, 3.815, 3.976]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(FULL_WIDTH, 2.3), gridspec_kw=dict(wspace=0.28, width_ratios=[1, 1.25]))
    x = np.arange(3)
    w = 0.34
    a1.bar(x - w / 2, v_interp, w, color=LIGHT, edgecolor=BLACK, linewidth=0.5, label="Time interpolation")
    a1.bar(x + w / 2, v_model, w, color=NAVY, edgecolor=BLACK, linewidth=0.5, label="Residual model")
    a1.set_xticks(x)
    a1.set_xticklabels(regimes)
    a1.set_ylim(0, 4.6)
    a1.set_ylabel("Speed RMSE (km/h)")
    a1.set_xlabel("Masking regime")
    a1.legend(loc="upper left", ncol=1)
    panel(a1, "a")
    xg = np.arange(len(gaps))
    a2.plot(xg, g_interp, color=GREY, marker="s", markerfacecolor="white", label="Time interpolation")
    a2.plot(xg, g_model, color=NAVY, marker="o", label="Residual model")
    a2.set_xticks(xg)
    a2.set_xticklabels(gaps)
    a2.set_xlim(-0.4, len(gaps) - 0.6)
    a2.set_ylim(0, 4.6)
    a2.set_xlabel("Gap to the farther temporal anchor (slots)")
    a2.set_ylabel("Speed RMSE (km/h)")
    a2.legend(loc="upper left")
    panel(a2, "b")
    save(fig, "fig_state_error")


def fig_gap_ladder(ladder_path: Path) -> None:
    # 14.316 (shell 1, about 0.02), 14.322 (shell 2) and 14.323 (shells 3 to 25-48).
    shells = ["1", "2", "3", "4", "5", "6", "7", "8", "9–12", "13–24", "25–48"]
    bias = [0.02, 0.02774, 0.039, 0.059, 0.082, 0.108, 0.143, 0.184, 0.196, 0.309, 1.451]
    written = {0, 1, 2, 3, 4, 5}
    top = 0.40
    fig, ax = plt.subplots(figsize=(FULL_WIDTH, 2.25))
    x = np.arange(len(shells))
    for xi, b in zip(x, bias):
        ax.bar(xi, min(b, top), 0.55, color=NAVY if xi in written else LIGHT, edgecolor=BLACK, linewidth=0.5, zorder=2)
    ax.annotate(f"{bias[-1]:.2f}", (x[-1], top), xytext=(0, 2), textcoords="offset points", ha="center", va="bottom",
                fontsize=7.5)
    if ladder_path.is_file():
        key = {"g3": 2, "g4": 3, "g5": 4, "g6": 5, "g7": 6, "g8": 7, "g9_12": 8, "g13_24": 9, "g25_48": 10}
        rng = np.random.default_rng(7)
        for r in json.loads(ladder_path.read_text())["february"]:
            if r["shell"] in key and r["mean_abs_bias"] is not None and r["mean_abs_bias"] <= top:
                ax.scatter(key[r["shell"]] + rng.uniform(-0.14, 0.14), r["mean_abs_bias"], s=7, marker="o",
                           facecolor="white", edgecolor=BLACK, linewidth=0.5, zorder=3)
    ax.axhline(0.15, color=BLACK, lw=0.8, ls="--", zorder=1)
    ax.set_xticks(x)
    ax.set_xticklabels(shells)
    ax.set_xlim(-0.6, len(shells) - 0.4)
    ax.set_ylim(0, top + 0.06)
    ax.set_xlabel("Gap shell (slots to the farther temporal anchor)")
    ax.set_ylabel("Mean |link bias| (km/h)")
    handles = [Patch(facecolor=NAVY, edgecolor=BLACK, linewidth=0.5, label="Shell written"),
               Patch(facecolor=LIGHT, edgecolor=BLACK, linewidth=0.5, label="Shell not written"),
               Line2D([], [], color=BLACK, marker="o", markerfacecolor="white", markersize=3.5, lw=0, label="Single panel"),
               Line2D([], [], color=BLACK, lw=0.8, ls="--", label="Threshold 0.15 km/h")]
    ax.legend(handles=handles, loc="upper left")
    save(fig, "fig_gap_ladder")


def fig_queue_selection() -> None:
    # 14.328 (published TRAIN windows, latent labels), 14.330 and 14.334 (odd March dates, visible proxy).
    groups = ["Ongoing, published\nTRAIN windows (40)", "Onset, published\nTRAIN windows (40)",
              "Ongoing, visible March\nodd dates (146)"]
    series = [
        ("Persistence", dict(color="white", edgecolor=BLACK), [0.7551, 0.0250, 0.7184]),
        ("Analogue choice", dict(color=LIGHT, edgecolor=BLACK), [0.8357, 0.9375, 0.7178]),
        ("Deployed decoder", dict(color=NAVY, edgecolor=BLACK), [np.nan, np.nan, 0.8129]),
        ("Best candidate (oracle)", dict(color="white", edgecolor=BLACK, hatch="////"), [0.9343, 1.0, 0.8748]),
    ]
    fig, ax = plt.subplots(figsize=(FULL_WIDTH, 2.3))
    x = np.arange(len(groups))
    w = 0.19
    for k, (label, style, vals) in enumerate(series):
        offset = (k - 1.5) * w
        for xi, v in zip(x, vals):
            if not np.isnan(v):
                ax.bar(xi + offset, v, w * 0.92, linewidth=0.5, **style)
        ax.bar(np.nan, np.nan, 0, linewidth=0.5, label=label, **style)
    ax.set_xticks(x)
    ax.set_xticklabels(groups)
    ax.set_ylim(0, 1.18)
    ax.set_ylabel("Mean IoU (proxy)")
    ax.legend(loc="upper center", ncol=4, columnspacing=1.2, handletextpad=0.4)
    save(fig, "fig_queue_selection")


def fig_transfer(rows: list[dict]) -> None:
    s = {r["label"]: r["score"] for r in rows}
    state = [
        ("RS1 state model", s["RC2"] - s["RC1"]),
        ("S10/S12 corrections", s["CT2"] - s["CT1"]),
        ("Sample ensemble (TF3)", s["TF3"] - s["CF2"]),
        ("Gap-1 recentering", s["ES2"] - s["ES1"]),
        ("Gap-2 recentering", s["ES3"] - s["ES2"]),
        ("Gap 3–6 recentering", s["ES4"] - s["ES3"]),
        ("March-fit speed model (ES5)", s["ES5"] - s["ES4"]),
    ]
    queue = [
        ("Ongoing tree, unfiltered windows", s["V1-nh"] - s["V1"]),
        ("Ongoing forward-fill (U2)", s["U2"] - s["U1"]),
        ("Onset analogue (J1)", s["J1"] - s["U2"]),
        ("Onset frequency prior (U3)", s["U3"] - s["U2"]),
        ("RQ7/RQ12 queue bundle (RC1)", s["RC1"] - s["J1"]),
        ("Ongoing origin analogue (CS2)", s["CS2"] - s["CS1"]),
        ("Onset demand completion (CS3)", s["CS3"] - s["CS2"]),
        ("RQ48 decoder (CF2)", s["CF2"] - s["CF1"]),
        ("Occupancy classifier (CF3)", s["CF3"] - s["CF2"]),
        ("All-month classifier (ES1)", s["ES1"] - s["TF3"]),
    ]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(FULL_WIDTH, 2.75), gridspec_kw=dict(wspace=1.05))
    for ax, items, letter, lim in ((a1, state, "a", (-6, 12)), (a2, queue, "b", (-30, 24))):
        y = np.arange(len(items))[::-1]
        vals = np.array([v for _, v in items]) * 1000
        ax.barh(y, vals, 0.6, color=[NAVY if v >= 0 else RED for v in vals], edgecolor=BLACK, linewidth=0.5)
        ax.axvline(0, color=BLACK, lw=0.6)
        ax.set_yticks(y)
        ax.set_yticklabels([n for n, _ in items])
        ax.tick_params(axis="y", length=0)
        ax.set_xlim(*lim)
        ax.set_xlabel(r"Change in validation score ($\times 10^{-3}$)")
        panel(ax, letter)
    save(fig, "fig_transfer")


# ------------------------------------------------------------ generated TeX
def write_tables(rows: list[dict], comp: dict) -> None:
    GEN.mkdir(parents=True, exist_ok=True)
    short = {
        "BASE": "Organizer baselines for all tasks", "T4-v2": r"ODME: prior $\times$ smooth correction",
        "T4-v4": "ODME: KL/entropy fit to the prior", "T4-v5": "ODME: KL from a gravity seed",
        "T4-v1": "ODME: exact NNLS on the split's own data", "V1": "First GBDT state, LightGBM queues",
        "P-persq": "V1 state with persistence queues", "P-noonset": "V1 with onset rows set to zero",
        "V1-nh": "V1, ongoing tree on unfiltered windows", "P-slotT": r"Ongoing set from slot $T$ (diagnostic)",
        "U1": "Legal onset classifier (12 slots)", "U2": "U1 with forward-filled ongoing persistence",
        "J1": "U2 with the joint-mask onset analogue", "L0": "U2 with time-interpolated state",
        "U3": "U2 with an onset frequency prior", "RC1": "RQ7 ongoing and RQ12 onset", "RC2": "RC1 with RS1 state",
        "CS1": "RS1 state, RQ2 ongoing, J1 onset", "CS2": "CS1 with origin-analogue ongoing",
        "CS3": "CS2 with onset demand completion", "CT1": "S5 speed correction, Q19 ongoing",
        "CT2": "CT1 with S10/S12 state", "CF1": "FS1 gate, Q46 ongoing", "CF2": "CF1 with RQ48 decoder",
        "CF3": "CF2 with occupancy classifier", "TF3": "CF2 with three-member ensemble",
        "ES1": "TF3 with all-month ongoing classifier", "ES2": "ES1 with gap-1 recentering",
        "ES3": "ES2 with gap-2 recentering", "ES4": "ES3 with gap 3--6 recentering", "ES5": "ES4 with March-fit speed model",
    }
    lines = [r"\begin{tabular}{@{}r l l r@{}}", r"\toprule", r"\# & ID & Content & Validation score \\", r"\midrule"]
    for r in rows:
        lab, score = r["label"], f"{r['score']:.5f}"
        if r["category"] == "selected":
            lab, score = rf"\textbf{{{lab}}}", rf"\textbf{{{score}}}"
        lines.append(f"{r['index']} & {lab} & {short[r['label']]} & {score} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (GEN / "tab_submissions.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")

    lines = [r"\begin{tabular}{@{}r l l r r@{}}", r"\toprule",
             r"Step & Change & Rows changed & $\Delta$ score & Cumulative \\", r"\midrule",
             rf"0 & Organizer baselines (anchor) & all & & {[r for r in rows if r['label'] == 'BASE'][0]['score']:.5f} \\\\"]
    level = [r for r in rows if r["label"] == "BASE"][0]["score"]
    for i, (name, task, delta) in enumerate(steps(rows), 1):
        level += delta
        lines.append(f"{i} & {name} & {task} & {delta:+.5f} & {level:.5f} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (GEN / "tab_steps.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")

    def f(v, d=4):
        return f"{v:.{d}f}"
    comp_rows = [
        ("Demand", r"$\Sodme$, organizer solve on TRAIN counts", f(comp["ODME_baseline"]), "documented"),
        ("", r"$\Sodme$, exact NNLS on the split's own data", f(comp["ODME_nnls"]), "T4-v1 $-$ BASE"),
        ("State", r"$B$, weekday $\times$ slot historical mean", f(comp["B_baseline"]), "documented"),
        ("", r"$B$, time interpolation", f(comp["B_interp"]), "L0 $-$ U2"),
        ("", r"$B$, first residual GBDT (V1)", f(comp["B_v1"]), "P-persq"),
        ("", r"$B$, RS1 residual GBDT", f(comp["B_rs1"]), "RC2 $-$ RC1"),
        ("Onset", "IoU, persistence (predicts nothing)", r"$\approx 0$", "by construction"),
        ("", r"IoU, first LightGBM, 13 slots incl.\ $T$ (V1)", f(comp["onset_v1"]), "V1 $-$ P-noonset"),
        ("", "IoU, legal 12-slot classifier (U1)", f(comp["onset_u1"]), "U1 $-$ P-persq"),
        ("", "IoU, training frequency prior (U3)", f(comp["onset_u3"]), "U3 $-$ U2"),
        ("", "IoU, demand-history completion (CS3)", f(comp["onset_cs3"]), "CS3 $-$ CS2"),
        ("", "IoU, joint-mask analogue (J1, kept to ES4)", f(comp["onset_j1"]), "J1 $-$ U2"),
        ("Ongoing", "IoU, organizer persistence", f(comp["ongoing_persistence"]), "documented"),
        ("", r"IoU, first LightGBM, 13 slots incl.\ $T$ (V1)", f(comp["ongoing_v1"]), "P-noonset"),
        ("", "IoU, same model, unfiltered training windows", f(comp["ongoing_v1_unfiltered"]), "V1-nh $-$ V1"),
        ("", r"IoU, state at slot $T$ (diagnostic only)", f(comp["ongoing_slotT"]), "P-slotT $-$ V1"),
        ("", "IoU, forward-fill of the last legal value (U2)", f(comp["ongoing_u2"]), "U2 $-$ U1"),
        ("", "IoU, RQ2 analogue decoder (CS1)", f(comp["ongoing_rq2"]), "CS1 identity"),
        ("", "IoU, origin analogue (CS2)", f(comp["ongoing_cs2"]), "CS2 $-$ CS1"),
        ("", r"$\Delta$IoU, RQ48 quarter-prior decoder", f"+{comp['d_ongoing_cf2']:.4f}", "CF2 $-$ CF1"),
        ("", r"$\Delta$IoU, occupancy classifier RQ59", f"{comp['d_ongoing_cf3']:+.4f}", "CF3 $-$ CF2"),
        ("", r"$\Delta$IoU, all-month classifier EQ1", f"+{comp['d_ongoing_es1']:.4f}", "ES1 $-$ TF3"),
    ]
    lines = [r"\begin{tabular}{@{}l l r l@{}}", r"\toprule", r"Task & Quantity & Value & Identified from \\", r"\midrule"]
    for i, (task, q, v, src) in enumerate(comp_rows):
        if task and i:
            lines.append(r"\addlinespace[2pt]")
        lines.append(f"{task} & {q} & {v} & {src} \\\\")
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
    print("decomposition check: J1 rebuilt", round(comp["check_j1"], 5),
          "Kaggle", next(r["score"] for r in rows if r["label"] == "J1"))
    write_tables(rows, comp)
    fig_fundamental_diagram(args.network_root)
    fig_state_error()
    fig_gap_ladder(args.gap_ladder)
    fig_queue_selection()
    fig_transfer(rows)


if __name__ == "__main__":
    main()
