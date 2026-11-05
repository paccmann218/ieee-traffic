"""Build the final TrafficFlowBench report.

Fonts are the Windows Segoe UI and Georgia files. The output is
docs/TrafficFlowBench_Final_Report.pdf.
"""
from pathlib import Path

from reportlab.lib.colors import HexColor, white
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    PageBreak, KeepTogether, HRFlowable, ListFlowable, ListItem,
)


NAVY = HexColor("#102033")
TEAL = HexColor("#1C6B64")
SAND = HexColor("#F4F1EA")
INK = HexColor("#1C1C1C")
MUTED = HexColor("#5E6872")
LINE = HexColor("#D9D3C7")
ROW = HexColor("#F7F5F1")

pdfmetrics.registerFont(TTFont("Segoe", r"C:\Windows\Fonts\segoeui.ttf"))
pdfmetrics.registerFont(TTFont("Segoe-Bold", r"C:\Windows\Fonts\segoeuib.ttf"))
pdfmetrics.registerFont(TTFont("Georgia", r"C:\Windows\Fonts\georgia.ttf"))
pdfmetrics.registerFont(TTFont("Georgia-Bold", r"C:\Windows\Fonts\georgiab.ttf"))
pdfmetrics.registerFont(TTFont("Consolas", r"C:\Windows\Fonts\consola.ttf"))
pdfmetrics.registerFontFamily("Segoe", normal="Segoe", bold="Segoe-Bold")
pdfmetrics.registerFontFamily("Georgia", normal="Georgia", bold="Georgia-Bold")


def styles():
    return {
        "h1": ParagraphStyle("h1", fontName="Georgia-Bold", fontSize=16, leading=20, textColor=NAVY, spaceBefore=8, spaceAfter=6),
        "h2": ParagraphStyle("h2", fontName="Segoe-Bold", fontSize=11.5, leading=15, textColor=TEAL, spaceBefore=9, spaceAfter=3),
        "body": ParagraphStyle("body", fontName="Segoe", fontSize=9.5, leading=13.2, textColor=INK, alignment=TA_LEFT, spaceAfter=7),
        "bullet": ParagraphStyle("bullet", fontName="Segoe", fontSize=9.5, leading=12.8, textColor=INK, leftIndent=2),
        "caption": ParagraphStyle("caption", fontName="Segoe", fontSize=8, leading=10.5, textColor=MUTED, spaceBefore=1, spaceAfter=9),
        "th": ParagraphStyle("th", fontName="Segoe-Bold", fontSize=8, leading=10, textColor=white),
        "td": ParagraphStyle("td", fontName="Segoe", fontSize=8, leading=10.5, textColor=INK),
        "code": ParagraphStyle("code", fontName="Consolas", fontSize=7.6, leading=10, textColor=NAVY, backColor=SAND, leftIndent=4, rightIndent=4, spaceBefore=2, spaceAfter=8),
    }


S = styles()


def P(text, kind="body"):
    return Paragraph(text, S[kind])


def bullets(items):
    return ListFlowable(
        [ListItem(Paragraph(item, S["bullet"]), leftIndent=8, value="•") for item in items],
        bulletType="bullet", start="•", leftIndent=12, spaceBefore=0, spaceAfter=8,
    )


def table(header, rows, widths):
    data = [[P(cell, "th") for cell in header]]
    for row in rows:
        data.append([P(cell, "td") for cell in row])
    chart = Table(data, colWidths=widths, repeatRows=1)
    chart.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), NAVY),
        ("TEXTCOLOR", (0, 0), (-1, 0), white),
        ("BACKGROUND", (0, 1), (-1, -1), white),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [white, ROW]),
        ("LINEBELOW", (0, 0), (-1, 0), 0, TEAL),
        ("LINEBELOW", (0, 1), (-1, -2), 0.2, LINE),
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 5),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
    ]))
    return chart


def cover(canvas, doc):
    canvas.saveState()
    width, height = A4
    canvas.setFillColor(NAVY)
    canvas.rect(0, 0, width, height, fill=1, stroke=0)
    canvas.setFillColor(TEAL)
    canvas.rect(0, height - 22 * mm, width, 5 * mm, fill=1, stroke=0)
    canvas.setFillColor(white)
    canvas.setFont("Segoe", 9)
    canvas.drawString(18 * mm, height - 42 * mm, "2026 IEEE BIG DATA CUP")
    canvas.setFont("Georgia", 28)
    canvas.drawString(18 * mm, height - 62 * mm, "TrafficFlowBench")
    canvas.setFont("Georgia", 16)
    canvas.drawString(18 * mm, height - 74 * mm, "Final report")
    canvas.setFillColor(HexColor("#D5E4E1"))
    canvas.setFont("Segoe", 11)
    text = canvas.beginText(18 * mm, height - 96 * mm)
    text.setLeading(16)
    for line in (
        "Two selected submissions",
        "TF3  ·  public score 0.86991",
        "ES4  ·  public score 0.87187",
    ):
        text.textLine(line)
    canvas.drawText(text)
    canvas.setFillColor(HexColor("#C5CDD4"))
    canvas.setFont("Segoe", 9)
    canvas.drawString(18 * mm, 42 * mm, "Team Powerbuff Girls")
    canvas.drawString(18 * mm, 36 * mm, "Kaggle account naomiandreapereira")
    canvas.drawString(18 * mm, 30 * mm, "November 2026")
    canvas.setFillColor(TEAL)
    canvas.rect(18 * mm, 24 * mm, 36 * mm, 1.2 * mm, fill=1, stroke=0)
    canvas.restoreState()


def later(canvas, doc):
    canvas.saveState()
    width, height = A4
    canvas.setFillColor(TEAL)
    canvas.rect(0, height - 7 * mm, width, 2.2 * mm, fill=1, stroke=0)
    canvas.setFillColor(MUTED)
    canvas.setFont("Segoe", 8)
    canvas.drawString(16 * mm, 9 * mm, "Powerbuff Girls   ·   TrafficFlowBench")
    canvas.drawRightString(width - 16 * mm, 9 * mm, str(doc.page - 1))
    canvas.restoreState()


def build():
    out = Path(__file__).resolve().parent / "TrafficFlowBench_Final_Report.pdf"
    doc = SimpleDocTemplate(
        str(out), pagesize=A4,
        leftMargin=16 * mm, rightMargin=16 * mm,
        topMargin=16 * mm, bottomMargin=16 * mm,
        title="TrafficFlowBench final report",
        author="Powerbuff Girls",
        subject="Selected submissions TF3 and ES4",
    )
    usable = A4[0] - 32 * mm
    story = [PageBreak()]

    story.append(P("1.  The two files we ask to be scored", "h1"))
    story.append(P(
        "This report describes the two submissions selected on the Kaggle dashboard for "
        "the 2026 IEEE Big Data Cup, TrafficFlowBench. The team is Powerbuff Girls. "
        "The Kaggle account is naomiandreapereira. Both files contain the same "
        "6,980,503 rows. They were compared cell by cell. Flow and path flow are identical. "
        "They differ in 5,825,667 speed cells and in all 87,000 ongoing-queue cells. "
        "Onset queues are identical."
    ))
    story.append(table(
        ["File", "Kaggle ref", "Submitted (UTC)", "Public score"],
        [
            ["submission_TF3.zip", "56825669", "4 Oct 2026, 13:51", "0.86991"],
            ["submission_ES4.zip", "56998048", "9 Oct 2026, 07:13", "0.87187"],
        ],
        [42 * mm, 32 * mm, 48 * mm, 32 * mm],
    ))
    story.append(P("Public scores were read from the Kaggle submission list on 5 November 2026. No private score has been returned.", "caption"))
    story.append(P(
        "ES4 is the stronger public file, by 0.00196. On 5 November 2026 the account's "
        "public rank was 197 among 765 teams. The private leaderboard, which decides the "
        "Kaggle ranking, is released after the 6 November deadline. Awards then combine "
        "that ranking with this report."
    ))

    story.append(P("2.  What the score is", "h1"))
    story.append(P(
        "One number ranks a submission. It is a weighted sum of four tasks, each on a scale from 0 to 1."
    ))
    story.append(P("S_total = 0.35 S_state + 0.30 S_queue + 0.15 S_physics + 0.20 S_ODME", "code"))
    story.append(bullets([
        "<b>State, weight 0.35.</b> On blanked eligible cells, speed error is divided by 25 km/h and per-lane flow error by 600 veh/h. The blend is 0.54 speed and 0.46 flow. The three mask rates, 20, 30 and 50 percent, are averaged with equal weight.",
        "<b>Queue, weight 0.30.</b> For each of six future steps, from T+5 through T+30, a link is queued when its latent speed is at or below 0.60 of free-flow speed. The score is the intersection-over-union of the predicted and true queued cells. Onset and ongoing windows have equal weight. The two D12 I-405 directions are not part of this task.",
        "<b>Physics, weight 0.15.</b> There is no separate physics file. Density is flow divided by speed. One third of the score checks the triangular fundamental diagram. Two thirds check whether the change in vehicles on a link equals inflow minus outflow, including the released ramp counts. All ten corridors are coverage mode A.",
        "<b>Demand, weight 0.20.</b> Path flows are compared with the withheld reference, with the released link counts, with the distance moved from the released prior, and with the destination-attraction pattern.",
    ]))
    story.append(P(
        "Directions are averaged inside a corridor family, and the five families are then averaged equally. "
        "A missing task scores 0. The public number uses the validation month. The private number uses a "
        "different month, generated separately, with different incidents. Fitting the validation month "
        "is not the same as learning the traffic."
    ))

    story.append(P("3.  What both files have in common", "h1"))
    story.append(P(
        "Both files answer the four tasks with the same division of labor. State is a reconstruction "
        "from the masked detectors. Queue is a forecast from the legal hour of history. Demand is a "
        "split-specific fit to the released counts. Physics is whatever that state field implies."
    ))
    story.append(P("State reconstruction", "h2"))
    story.append(P(
        "The blanked cells are not filled by copying a weekday profile. The profile is only a feature. "
        "For each target cell the inputs are built from the masked speed, flow and occupancy, the "
        "nearest observed neighbors in time, a Treiber–Helbing smoothing of those observations, "
        "ramp flows, link position, lane count, and the weekday-and-time profile estimated from "
        "training days. The model predicts the residual from time interpolation, separately for speed "
        "and for per-lane flow. It is a gradient-boosted tree: 800 rounds, 255 leaves, learning rate 0.05, "
        "minimum leaf size 150, feature fraction 0.9, row fraction 0.8, and an L2 penalty of 2. "
        "There is no early stopping."
    ))
    story.append(P(
        "A density is then formed from the two predictions and projected halfway toward the value implied "
        "by a triangular diagram. A further correction, fixed before the final fit, adjusts free-flow "
        "cells using flow factors estimated from that split's own masked observations. Target speed, "
        "flow and occupancy are not inputs. March and April were not used to fit the trees or to choose "
        "the ensemble."
    ))
    story.append(P("Queue forecast", "h2"))
    story.append(P(
        "Each window provides twelve observations, from T−60 through T−5. The origin cell at T is "
        "released by the current data page and may be used. Neither selected file reads it. "
        "The forecast is six binary fields, T+5 through T+30."
    ))
    story.append(P(
        "The predictor looks up the twenty training episodes whose recent speed and flow most resemble "
        "the query, inside a time-of-day and weekend window. It chooses a whole future mask from that "
        "neighborhood: the persisted queue, a neighbor's future, or a threshold on the neighbor vote. "
        "A frozen classifier, fit on training windows, supplies a probability that is mixed with the "
        "empirical vote. The ongoing rule uses a quarter of the empirical vote and three quarters of "
        "the classifier. Onset uses the neighbor mask itself, and in these files the positive calls "
        "sit on the last step, which is where the training onset windows first cross the cutoff. "
        "The queue threshold is 0.60 times the measured free-flow speed in fd_parameters.csv."
    ))
    story.append(P("Demand", "h2"))
    story.append(P(
        "Path flows solve a nonnegative least-squares problem on the path-link incidence of that split:"
    ))
    story.append(P("Squared error against the released counts, plus 0.05 times the squared distance from the released prior. Flows stay nonnegative.", "code"))
    story.append(P(
        "Here c is the released link count and b is the released prior. The same solver and the same "
        "penalty are used for validation and for the private month, each on its own counts and its own prior. "
        "Training counts are not substituted. Both selected files contain this solution. Their path-flow "
        "columns match exactly."
    ))

    story.append(P("4.  TF3: three reconstructions, one average", "h1"))
    story.append(P(
        "A single boosted tree is fit on a random sample of eligible training cells. A different sample "
        "produces a different field. TF3 asks whether the average of three such fields is steadier on a "
        "month the trees did not fit."
    ))
    story.append(P(
        "There are exactly three members. One is the frozen earlier model. The other two use seeds "
        "20261104 and 20261204. Each new member draws up to 600,000 eligible rows per panel, without "
        "replacement. The draw is deterministic given the seed and the panel. The samples overlap. "
        "They are not new measurements. For each of speed and flow, the three raw predictions are "
        "averaged with equal weight, and only then are the density and free-flow corrections applied. "
        "No member is dropped after seeing a score, and the weights are not tuned."
    ))
    story.append(P(
        "The choice was locked before the final fit. On December and on January, using models fit only "
        "through November, the joint local objective 0.35 S_state + 0.15 S_physics had to rise by at "
        "least 0.0001, the state term could not fall, at least seven of ten panels had to improve, and "
        "no panel could lose more than 0.001. Both months passed on all ten panels. February was then "
        "scored once, with the same rule, and also passed on all ten panels. The joint gains were "
        "0.000372, 0.000385 and 0.000318. These are training-month proxies, not leaderboard scores."
    ))
    story.append(P(
        "The accepted rule was then refit on all nine training months and applied to the validation "
        "and private months. A clean run from the released files, using the saved weights and rebuilding "
        "every cache, reproduced all 6,980,503 rows. Queues matched exactly. The largest continuous "
        "difference was 1.82e−12. That check is saved-weight inference. It is not a second training run."
    ))
    story.append(P(
        "TF3 keeps the queue and the demand of the file it started from. Kaggle returned a public score "
        "of 0.86991 on reference 56825669. The archive SHA256 is "
        "2ae94d422399c7248297487fc8f45d737de0e024fbad32b6a360c290bbabca7a."
    ))

    story.append(P("5.  ES4: take a link bias out of the short gaps", "h1"))
    story.append(P(
        "On a February target cell, the speed obtained by linear interpolation between the nearest "
        "observed samples is nearly unbiased by link. The mean absolute link bias of that interpolant, "
        "on the short-gap shells, is a few hundredths of a kilometre per hour. The reconstructed field "
        "is not. Its link means sit away from the interpolant. ES4 subtracts that link mean, and only "
        "on shells where the February test says the interpolant itself is safe."
    ))
    story.append(P(
        "A shell is defined by the farther of the two neighbors. Both neighbors must exist. The February "
        "rule, fixed before any validation or private cell is edited, accepts a shell when the mean "
        "absolute link bias of the interpolant is under 0.15 km/h and the bias correction does not raise "
        "February speed error by more than 0.02 km/h. The offset applied to March and April is the "
        "link mean of (submitted speed minus interpolant) on that shell, using only masked observations. "
        "A link needs at least 100 such cells. The result is clipped to the interval from 5 to 140 km/h. "
        "A shell is written only when the root-mean-square link shift is at least 0.40 km/h and no link "
        "shift exceeds 15 km/h."
    ))
    story.append(P(
        "Three shells passed and were applied, in order, on top of the TF3 speeds. The one-slot shell "
        "changed 1,715,267 speeds. The exactly-two-slot shell changed 1,918,477. The three-to-six-slot "
        "shell changed 2,191,923, of which 1,186,051 are gap 3, 671,422 are gap 4, 319,161 are gap 5, "
        "and 15,289 are gap 6. The seven-slot shell passed the February test and then had no validation "
        "or private cells under the writing rule, so it changed nothing. Gaps of eight slots and more "
        "failed the February bias test and were left alone."
    ))
    story.append(table(
        ["Shell", "Speeds changed", "What February said"],
        [
            ["At most 1 slot", "1,715,267", "Interpolant bias about 0.02 km/h"],
            ["Exactly 2 slots", "1,918,477", "Same test, then written"],
            ["3 to 6 slots", "2,191,923", "Mean absolute bias 0.039 to 0.108 km/h by shell"],
            ["7 slots", "0", "February passed; no cells met the writing rule"],
            ["8 slots and more", "0", "February bias test failed"],
        ],
        [38 * mm, 32 * mm, usable - 70 * mm],
    ))
    story.append(P(
        "The last shell moved those 2,191,923 speeds by 0.954 km/h on average. The root-mean-square "
        "change was 1.246 km/h and the largest change was 6.06 km/h. Adding the earlier shells, "
        "ES4 and TF3 differ in 5,825,667 speed cells.",
        "caption",
    ))
    story.append(P(
        "Between TF3 and those speed edits, the ongoing queue was replaced by the same decoder fit on "
        "all nine training months rather than on the earlier training window. The replacement was "
        "accepted only after a control model reproduced TF3's ongoing labels. The comparison of the "
        "two selected archives shows that all 87,000 ongoing cells differ and that no onset cell differs. "
        "Flow does not differ. Path flow does not differ."
    ))
    story.append(P(
        "Kaggle returned a public score of 0.87187 on reference 56998048. Relative to TF3 that is "
        "+0.00196. The successive public scores on the way there were 0.87030 after the queue "
        "replacement, 0.87069 after the one-slot edit, and 0.87113 after the two-slot edit. "
        "The archive SHA256 is ce03386b42e33bc6c726b09c0b64b791f294ea3abeb88d7e310552bb080c04bb."
    ))

    story.append(P("6.  What the numbers say, and what they do not", "h1"))
    story.append(P("The public gains are real and small", "h2"))
    story.append(P(
        "Each of the four public steps above was one change, confirmed by a separate read of the "
        "Kaggle submission list. The ensemble, the ongoing-queue refit, and the three gap shells "
        "each moved the public score by a few ten-thousandths. Together they are the difference "
        "between TF3 and ES4. They are not a claim that the field is close to a perfect reconstruction."
    ))
    story.append(P(
        "On February target cells the saved reconstruction, before the gap edits, has speed errors of "
        "1.46, 1.55 and 1.66 km/h on the 20, 30 and 50 percent masks. Time interpolation on the same "
        "cells is at 2.75, 2.84 and 2.98 km/h. Per-lane flow error is about 31 veh/h against about "
        "45 to 48 for interpolation. The state piece, averaged across the three masks, is 0.943 "
        "against 0.903. That is a training-month proxy. It says the trees beat interpolation on every "
        "mask rate, including the half-blanked days. It does not say what March or April will score."
    ))
    story.append(P("Where the remaining error sits", "h2"))
    story.append(P(
        "State error is no longer the large term on the training months. A cell whose neighbors "
        "disagree by 20 km/h or more, or whose interpolant is below 60 percent of free-flow, is "
        "where interpolation fails: on 138,195 such February cells the interpolant is at 10.57 km/h "
        "and the saved model is at 4.62. The model is already the thing being submitted on those "
        "cells. Replacing it with the interpolant there would be the wrong edit, which is why the "
        "gap rule only removes a link mean and does not overwrite the cell."
    ))
    story.append(P(
        "Same-time neighbors are not a substitute. On 735,668 February holes where both adjacent "
        "links are observed, their average is at 6.37 km/h and the model is at 1.44. A spatial fill "
        "loses to the temporal model even inside a gap."
    ))
    story.append(P(
        "The queue is the open term. On the official training windows, with the query month removed "
        "from the library, the deployed analogue scores about 0.87 latent overlap on held-out ongoing "
        "windows. The best action inside the same set, chosen with the label, is higher. Rules that "
        "have to be frozen before seeing the label, including a different neighbor count, a vote on "
        "the latent neighbor field, and a learned reranker, did not beat the deployed rule on the "
        "held-out dates. ES4 therefore keeps that rule, with the one ongoing refit described above. "
        "It does not claim the oracle."
    ))
    story.append(P(
        "A March-only speed model makes the same point from the other direction. Fit on odd March "
        "days and judged on even March days, it lowered visible speed error from 2.545 to 2.006 km/h "
        "on every corridor. Written onto the hidden March holes, that file scored 0.86725 in public, "
        "0.00462 below ES4. Visible sensors are not a stand-in for the masked holes. ES4 does not "
        "contain that model."
    ))
    story.append(P("Physics", "h2"))
    story.append(P(
        "The local conservation score of the saved field on February is about 0.74. An exact answer "
        "is about 0.96, because the organizer's boundary flows are close to, but not identical to, "
        "the observed change in accumulation. A Newell lower envelope between the bounding detectors, "
        "and a bounded nudge of the flow toward that envelope, were scored on February after the "
        "selected files were frozen. Both raised speed error or lowered the local physics score. "
        "They are not in TF3 or ES4. The diagram check itself is already close to saturated for a "
        "field whose speed and flow are jointly plausible. The conservation term moves only when the "
        "state field moves."
    ))

    story.append(P("7.  How to reproduce the files", "h1"))
    story.append(P(
        "The repository entry point is reproduce.py. With no arguments it checks the SHA256 of both "
        "selected archives and of the TF3 code bundle. The competition release is not in the "
        "repository. Point the commands below at the directory that contains submission_key.csv, "
        "corridors and task2."
    ))
    story.append(P("python reproduce.py", "code"))
    story.append(P(
        "TF3 is regenerated from the saved weights. The command rebuilds the caches and the "
        "predictions. It does not submit anything. The output directory must be new. On the clean "
        "run of 4 October this route matched every row, with exact queues and a maximum continuous "
        "difference of 1.82e−12. Use several cores and enough disk for the release and the caches. "
        "The pinned environment is numpy 2.5.3, pandas 3.0.6, scipy 1.18.1, pyarrow 25.0.1, "
        "lightgbm 4.7.0 and scikit-learn 1.9.1."
    ))
    story.append(P(
        "python reproduce.py --submission tf3 --release-root PATH --out PATH --threads 6",
        "code",
    ))
    story.append(P(
        "The last ES4 shell, gaps of three to six slots, is code/recenter_short_gaps.py. "
        "It reads a parent archive and the release, and it copies every column except the speeds "
        "in that shell. The parent of that shell is the two-slot file, not TF3. Running it on TF3 "
        "would recenter the right speeds and would leave TF3's ongoing queues in place, so the "
        "result would not be ES4. The uploaded ES4 archive is the file in submissions/."
    ))
    story.append(P(
        "python reproduce.py --submission es4 --release-root PATH --parent PARENT.zip --out PATH",
        "code",
    ))

    story.append(P("8.  Limits", "h1"))
    story.append(bullets([
        "There is no private score for either file. The public difference of 0.00196 is not a forecast of the private difference.",
        "The February and December checks are proxies. The conservation proxy uses observed accumulation because the organizer boundary flow is withheld.",
        "The three ensemble members share features and a sample of the same roads. Averaging reduces draw-to-draw noise. It does not remove a bias they share.",
        "The gap edit is a link mean. It is not a new model of a shock. Gaps of eight or more slots failed the February test and remain the uncorrected field.",
        "The queue still cannot see which of its candidate futures is the right one without the label. That selection problem is open.",
        "Slot T is available in the release and is not used.",
    ]))
    story.append(Spacer(1, 4 * mm))
    story.append(HRFlowable(width="100%", thickness=0.4, color=LINE, spaceAfter=4))
    story.append(P(
        "Selected archives: submission_TF3.zip, reference 56825669, public 0.86991; "
        "submission_ES4.zip, reference 56998048, public 0.87187. "
        "Team Powerbuff Girls. November 2026.",
        "caption",
    ))

    doc.build(story, onFirstPage=cover, onLaterPages=later)
    print(out)


if __name__ == "__main__":
    build()
