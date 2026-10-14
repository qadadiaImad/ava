"""Build the PDF report: docs/totem_repo_control.pdf.

Usage:  PYTHONPATH=. python scripts/build_report.py [--benchmark results/benchmark.json.gz]
"""
from __future__ import annotations

import argparse
import io
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image as PILImage  # noqa: E402
from reportlab.lib import colors  # noqa: E402
from reportlab.lib.enums import TA_LEFT  # noqa: E402
from reportlab.lib.pagesizes import A4  # noqa: E402
from reportlab.lib.styles import ParagraphStyle  # noqa: E402
from reportlab.lib.units import mm  # noqa: E402
from reportlab.pdfbase import pdfmetrics  # noqa: E402
from reportlab.pdfbase.ttfonts import TTFont  # noqa: E402
from reportlab.platypus import (  # noqa: E402
    BaseDocTemplate,
    CondPageBreak,
    Frame,
    Image,
    KeepTogether,
    ListFlowable,
    ListItem,
    PageBreak,
    PageTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
)

sys.path.insert(0, str(Path(__file__).resolve().parent))
import report_analysis as ra  # noqa: E402

from repo_control.benchmark import DETECTOR_TITLES, DETECTORS  # noqa: E402
from repo_control.synthetic import CATEGORIES, EXPECTED_KIND, SCENARIO_TITLES, SCENARIOS  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
BUILD = ROOT / "docs" / "_build"
OUT = ROOT / "docs" / "totem_repo_control.pdf"

# ------------------------------------------------------------------ fonts
FONT_DIR = Path(matplotlib.get_data_path()) / "fonts" / "ttf"
pdfmetrics.registerFont(TTFont("Sans", str(FONT_DIR / "DejaVuSans.ttf")))
pdfmetrics.registerFont(TTFont("Sans-Bold", str(FONT_DIR / "DejaVuSans-Bold.ttf")))
pdfmetrics.registerFont(TTFont("Sans-Italic", str(FONT_DIR / "DejaVuSans-Oblique.ttf")))
pdfmetrics.registerFont(TTFont("Sans-BoldItalic", str(FONT_DIR / "DejaVuSans-BoldOblique.ttf")))
pdfmetrics.registerFont(TTFont("Mono", str(FONT_DIR / "DejaVuSansMono.ttf")))
pdfmetrics.registerFontFamily("Sans", normal="Sans", bold="Sans-Bold", italic="Sans-Italic", boldItalic="Sans-BoldItalic")

INK = colors.HexColor("#0b0b0b")
INK2 = colors.HexColor("#52514e")
MUTED = colors.HexColor("#898781")
GRID = colors.HexColor("#e1e0d9")
AXIS = colors.HexColor("#c3c2b7")
WASH = colors.HexColor("#f4f3ef")
ACCENT = colors.HexColor("#2a78d6")

BODY = ParagraphStyle("body", fontName="Sans", fontSize=9.2, leading=13.4, textColor=INK, alignment=TA_LEFT, spaceAfter=5.5)
SMALL = ParagraphStyle("small", parent=BODY, fontSize=8, leading=11, spaceAfter=3)
CAPTION = ParagraphStyle("caption", parent=BODY, fontSize=7.8, leading=10.6, textColor=INK2, spaceBefore=3, spaceAfter=10)
TITLE = ParagraphStyle("title", parent=BODY, fontName="Sans-Bold", fontSize=19, leading=24, spaceAfter=6)
SUBTITLE = ParagraphStyle("subtitle", parent=BODY, fontSize=10.5, leading=14.5, textColor=INK2, spaceAfter=4)
META = ParagraphStyle("meta", parent=BODY, fontSize=8.2, leading=11.5, textColor=MUTED, spaceAfter=14)
H1 = ParagraphStyle("h1", parent=BODY, fontName="Sans-Bold", fontSize=13.5, leading=17, spaceBefore=14, spaceAfter=7,
                    keepWithNext=1)
H2 = ParagraphStyle("h2", parent=BODY, fontName="Sans-Bold", fontSize=10.5, leading=14, spaceBefore=9, spaceAfter=4,
                    keepWithNext=1)
CELL = ParagraphStyle("cell", parent=BODY, fontSize=7.6, leading=9.8, spaceAfter=0)
CELL_B = ParagraphStyle("cellb", parent=CELL, fontName="Sans-Bold")
CODE = ParagraphStyle("code", parent=BODY, fontName="Mono", fontSize=7.8, leading=10.5, textColor=INK, backColor=WASH,
                      borderPadding=(5, 6, 5, 6), spaceBefore=4, spaceAfter=10)
CALLOUT = ParagraphStyle("callout", parent=BODY, backColor=WASH, borderPadding=(7, 8, 7, 8), spaceBefore=6, spaceAfter=12)

PAGE_W, PAGE_H = A4
MARGIN_L = MARGIN_R = 19 * mm
TEXT_W = PAGE_W - MARGIN_L - MARGIN_R


# --------------------------------------------------------------- helpers
def P(text, style=BODY):
    return Paragraph(text, style)


def bullets(items, style=BODY, numbered=False):
    flow = [ListItem(Paragraph(t, style), leftIndent=12, value=i + 1 if numbered else None) for i, t in enumerate(items)]
    return ListFlowable(
        flow,
        bulletType="1" if numbered else "bullet",
        start=1 if numbered else None,
        bulletFontName="Sans",
        bulletFontSize=8 if not numbered else 8.6,
        bulletColor=INK2,
        leftIndent=13,
        bulletDedent=11 if not numbered else 13,
        spaceAfter=6,
    )


_EQ_COUNTER = [0]


def equation(tex: str, number: bool = True, size: float = 12.5):
    """Render a display equation with matplotlib mathtext; returns a right-numbered row."""
    fig = plt.figure(figsize=(0.01, 0.01))
    fig.text(0, 0, f"${tex}$", fontsize=size, color="#0b0b0b")
    buf = io.BytesIO()
    fig.savefig(buf, dpi=400, bbox_inches="tight", pad_inches=0.02, transparent=True)
    plt.close(fig)
    buf.seek(0)
    w, h = PILImage.open(buf).size
    buf.seek(0)
    width, height = w * 72 / 400, h * 72 / 400
    avail = TEXT_W - (24 * mm if number else 0)
    if width > avail:
        width, height = avail, height * avail / width
    img = Image(buf, width=width, height=height)
    if not number:
        t = Table([[img]], colWidths=[TEXT_W])
        t.setStyle(TableStyle([("ALIGN", (0, 0), (-1, -1), "CENTER"), ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                               ("TOPPADDING", (0, 0), (-1, -1), 3), ("BOTTOMPADDING", (0, 0), (-1, -1), 5)]))
        return t
    _EQ_COUNTER[0] += 1
    t = Table([["", img, P(f"({_EQ_COUNTER[0]})", ParagraphStyle("eqn", parent=BODY, textColor=INK2, alignment=2))]],
              colWidths=[12 * mm, TEXT_W - 24 * mm, 12 * mm])
    t.setStyle(TableStyle([("ALIGN", (1, 0), (1, 0), "CENTER"), ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                           ("TOPPADDING", (0, 0), (-1, -1), 3), ("BOTTOMPADDING", (0, 0), (-1, -1), 5)]))
    return t


_FIG_COUNTER = [0]
_TAB_COUNTER = [0]


def figure(path: Path, caption: str, width: float = TEXT_W):
    _FIG_COUNTER[0] += 1
    w, h = PILImage.open(path).size
    img = Image(str(path), width=width, height=width * h / w)
    return KeepTogether([img, P(f"<b>Figure {_FIG_COUNTER[0]}.</b> {caption}", CAPTION)])


def table(rows, col_widths, caption=None, header_rows=1, align_right_from=1, zebra=False, font_size=None):
    """Hairline table; first ``header_rows`` rows bold; numeric columns right-aligned."""
    st = CELL if font_size is None else ParagraphStyle("c2", parent=CELL, fontSize=font_size, leading=font_size * 1.3)
    stb = CELL_B if font_size is None else ParagraphStyle("c2b", parent=st, fontName="Sans-Bold")
    data = []
    for i, row in enumerate(rows):
        out = []
        for j, cell in enumerate(row):
            if isinstance(cell, (Paragraph, Image, Table)):
                out.append(cell)
                continue
            s = stb if i < header_rows else st
            if j >= align_right_from:
                s = ParagraphStyle(f"r{i}{j}", parent=s, alignment=2)
            out.append(Paragraph(str(cell), s))
        data.append(out)
    t = Table(data, colWidths=col_widths, repeatRows=header_rows)
    style = [
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LINEBELOW", (0, header_rows - 1), (-1, header_rows - 1), 0.8, AXIS),
        ("LINEBELOW", (0, header_rows), (-1, -1), 0.4, GRID),
        ("TOPPADDING", (0, 0), (-1, -1), 2.6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 2.6),
        ("LEFTPADDING", (0, 0), (-1, -1), 3),
        ("RIGHTPADDING", (0, 0), (-1, -1), 3),
    ]
    if zebra:
        for r in range(header_rows, len(rows), 2):
            style.append(("BACKGROUND", (0, r), (-1, r), WASH))
    t.setStyle(TableStyle(style))
    items = []
    if caption:
        _TAB_COUNTER[0] += 1
        items.append(P(f"<b>Table {_TAB_COUNTER[0]}.</b> {caption}", ParagraphStyle("tc", parent=CAPTION, spaceBefore=4, spaceAfter=4)))
    items.append(t)
    items.append(Spacer(1, 9))
    return KeepTogether(items)


def pct(x, digits=0):
    return "–" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{100 * x:.{digits}f}%"


def num(x, digits=2):
    return f"{x:,.{digits}f}".replace("-", "−")


def _is_heading(item) -> bool:
    return isinstance(item, Paragraph) and item.style.name in ("h1", "h2")


def _height(f, frame_h) -> float:
    """Height of a flowable before the build (KeepTogether needs a canvas, so measure its parts)."""
    if isinstance(f, KeepTogether):
        return sum(_height(c, frame_h) for c in f._content)
    try:
        return f.wrap(TEXT_W, frame_h)[1] + f.getSpaceBefore() + f.getSpaceAfter()
    except Exception:
        return 30 * mm


def _keep_headings(story):
    """Insert a conditional page break before each heading so it never ends a page.

    The space required is the heading (and any heading right after it) plus the next block: its full height when it
    cannot split (tables, figures), about 30mm of it otherwise.
    """
    frame_h = PAGE_H - 36 * mm
    out = []
    for i, item in enumerate(story):
        if _is_heading(item) and not (i > 0 and _is_heading(story[i - 1])):
            need, j = 0.0, i
            while j < len(story) and _is_heading(story[j]):
                need += _height(story[j], frame_h)
                j += 1
            if j < len(story):
                nxt = story[j]
                h = _height(nxt, frame_h)
                need += h if isinstance(nxt, (KeepTogether, Table, Image)) else min(h, 30 * mm)
            out.append(CondPageBreak(min(need, 0.92 * frame_h)))
        out.append(item)
    return out


# ------------------------------------------------------------ page setup
def _decorate(canvas, doc):
    canvas.saveState()
    canvas.setFont("Sans", 7.2)
    canvas.setFillColor(MUTED)
    if doc.page > 1:
        canvas.drawString(MARGIN_L, PAGE_H - 12 * mm, "Totem-implied repo: div-agnostic control and bad-curve detection")
        canvas.drawRightString(PAGE_W - MARGIN_R, PAGE_H - 12 * mm, "Draft for discussion")
        canvas.setStrokeColor(GRID)
        canvas.setLineWidth(0.5)
        canvas.line(MARGIN_L, PAGE_H - 13.5 * mm, PAGE_W - MARGIN_R, PAGE_H - 13.5 * mm)
    canvas.drawRightString(PAGE_W - MARGIN_R, 10 * mm, f"{doc.page}")
    canvas.restoreState()


def build(benchmark_path: Path) -> Path:
    BUILD.mkdir(parents=True, exist_ok=True)
    res = ra.load_benchmark(benchmark_path)

    # ---------------------------------------------------- numbers & figures
    gap_rows = ra.panel_gap_table()
    cancel_rows = ra.cancellation_table()
    book_rows = ra.book_lines()
    jump_rows = ra.jump_table()
    f_gap = ra.figure_panel_gap(BUILD / "fig_panel_gap.png")
    f_cancel = ra.figure_cancellation(BUILD / "fig_cancellation.png")
    f_gallery, gallery_notes = ra.figure_gallery(res, BUILD / "fig_gallery.png")
    f_decomp, decomp_note = ra.figure_decomposition(res, BUILD / "fig_decomposition.png")
    f_det = ra.figure_detection(res, BUILD / "fig_detection.png")
    f_power = ra.figure_power(res, BUILD / "fig_power.png")
    f_attr = ra.figure_attribution(res, BUILD / "fig_attribution.png")

    anomaly_scen = list(EXPECTED_KIND)
    mean_det = {d: float(np.mean([res.detection_rate(s, d) for s in anomaly_scen])) for d in DETECTORS}
    clean_fpr = {d: res.detection_rate("clean", d) for d in DETECTORS}
    attr_rows = ra.attribution_table(res)
    flagged_anom = [r for r in res.records if r["scenario"] in EXPECTED_KIND and r["flags"]["structural"]]
    attr_pooled = float(np.mean([r["attribution_ok"] for r in flagged_anom])) if flagged_anom else float("nan")
    fo_flag_struct = res.detection_rate("fo_repo_mismark", "structural")
    fo_flag = {d: res.detection_rate("fo_repo_mismark", d) for d in DETECTORS}
    fo_mismark_rate = res.fo_mismark_rate("fo_repo_mismark")
    fo_mismark_clean = res.fo_mismark_rate("clean")
    power = {d: res.power_curve(d) for d in DETECTORS}

    def pw(d, lo):
        for a, b, rate, n in power[d]:
            if a == lo:
                return rate
        return float("nan")

    n_names = len(res.records)
    n_cat = len(CATEGORIES)
    det = res.detection_rate
    A_book = [r for r in book_rows if r[0].startswith("A") and r[1] == "materiality"][0]
    B_book = [r for r in book_rows if r[0].startswith("B") and r[1] == "materiality"][0]
    C_book = [r for r in book_rows if r[0].startswith("C") and r[1] == "materiality"][0]

    story = []
    # ======================================================== title block
    story += [
        P("Totem-implied repo: div-agnostic control and bad-curve detection", TITLE),
        P("Methodology, prototype and synthetic benchmark for the independent price verification of equity repo", SUBTITLE),
        P("14 October 2026 · Draft for discussion · Code: <font name='Mono'>repo_control</font> (Python), "
          "branch <font name='Mono'>claude/repo-curve-anomaly-detection-uqiy1k</font>", META),
    ]

    # ===================================================== executive summary
    story += [P("Executive summary", H1)]
    story += [P(
        "Repo parameters for control are implied from Totem: forwards come from consensus call and put prices, "
        "dividends from the separate Totem dividend service. The two services are fed by different contributors, "
        "and the number of contributors changes by tenor. The implied repo curve therefore inherits noise and "
        "artefacts from both panels, and from time to time shows shapes or jumps that the market does not justify. "
        "This note explains where those shapes come from and shows that the control does not need a Totem repo curve "
        "for most exposures. It then proposes a detector that flags bad curves, names the likely cause and repairs the curve. "
        "Each claim is checked with a Python prototype and a benchmark on a synthetic Totem market with explicit "
        "contributor panels.")]
    story += [P("<b>Findings</b>", BODY)]
    story += [bullets([
        "<b>The implied repo is a residual of two panels.</b> To first order the implied repo equals the option panel's "
        "average repo plus a <i>panel gap</i>: the difference between the option contributors' dividend view (never "
        "observed) and the dividend panel's consensus, divided by <i>T</i>(<i>S</i> − <i>D</i>). With realistic dispersions "
        f"the gap alone is {gap_rows[2][5]:.0f}bp of repo noise at 1Y and {gap_rows[-1][5]:.0f}bp at 5Y for a single stock, "
        "more on thin panels. An ex-date that part of the option panel puts on the other side of an expiry moves one pillar "
        "by hundreds of bp, and a 0.1% spot snapshot mismatch reads as 120bp of repo at 1M. Options identify total carry "
        "only, so no statistical model can split repo from dividends using consensus data alone.",
        "<b>The total control is div-agnostic; only the split is not.</b> For exposure that depends on the underlying "
        "through its forwards, the dividend line plus the repo line equals a <i>forward line</i> (Totem forwards repriced "
        "with FO dividends) to first order. In the worked example the total stays at "
        f"{num(book_rows[1][4])} while the two lines swing by more than ±6 across dividend panels, matching full revaluation. "
        "Dividends reach the total only through the non-forward exposure <i>g</i>, which comes from sensitivities alone: "
        f"{pct(A_book[4], 1)} of the dividend-line uncertainty cancels for a forward-only book, {pct(B_book[4])} once a dividend "
        "swap is added.",
        "<b>Bad-curve detection should be structural.</b> Work on the Totem − FO difference in carry space (<i>b·T</i>), "
        "weight it by a noise model built from Totem's own dispersions and contributor counts, and test one matched-filter "
        "template per mechanism (level, step, spike, season) after removing a smooth repo difference. One variance scale "
        "per category sets the false-alarm rate. Every alarm comes with an explanation in natural units and a repaired "
        "curve, and a smooth level difference is reported as a candidate FO mismark rather than a bad Totem curve.",
        f"<b>Benchmark</b> ({n_names:,} synthetic test names over {n_cat} categories and {len(SCENARIOS)} scenarios, every "
        f"detector calibrated to {pct(res.fpr_target)} false alarms on clean history; "
        f"{pct(min(clean_fpr.values()))}–{pct(max(clean_fpr.values()))} realised on fresh clean names). Averaged over the six Totem-side "
        f"anomaly types, the structural detector flags {pct(mean_det['structural'])} of names, the naive gap threshold "
        f"{pct(mean_det['naive_level'])}, and B-spline + GMM {pct(mean_det['spline_gmm_curve'])} on the curve and "
        f"{pct(mean_det['spline_gmm_diff'])} on the Totem − FO difference. Raw detection does not separate the best "
        f"methods; three things do. Forward outliers: {pct(det('fwd_outlier', 'structural'))} against "
        f"{pct(min(det('fwd_outlier', x) for x in DETECTORS if x != 'structural'))}–"
        f"{pct(max(det('fwd_outlier', x) for x in DETECTORS if x != 'structural'))}. Genuine FO repo mismarks: a bad-curve "
        f"alarm in {pct(fo_flag_struct)} of names (the clean rate) against {pct(fo_flag['naive_level'])} for the naive gap "
        f"and {pct(fo_flag['spline_gmm_diff'])} for GMM on the difference, with {pct(fo_mismark_rate)} reported as "
        f"FO-mismark candidates instead. Explanations: the first one names the injected mechanism in {pct(attr_pooled)} "
        "of the names flagged.",
    ])]
    story += [P("<b>Recommendations</b>", BODY)]
    story += [bullets([
        "Control: report the forward line and the non-forward line; use the dividend/repo split only where the "
        "non-forward exposure is material.",
        "Detection: deploy the structural detector; calibrate its variance scale per region × category on 12–24 "
        "months of Totem history; review alarms with the attribution text.",
        "Inputs: synchronise the Totem reference spot with the FO spot; allocate dividend buckets with ex-date "
        "uncertainty around expiries; keep the parity diagnostics (implied discount factor, residuals, dispersion spikes).",
        "Anchors: where they trade, use listed index futures, dividend futures and total-return futures for indices, "
        "and securities-lending fees for the short end of single stocks.",
        "Keep B-spline + GMM and small neural networks as secondary layers on the structural outputs, not as the "
        "primary detector.",
    ], numbered=True)]

    # ====================================================== 1. setting
    story += [CondPageBreak(60 * mm), P("1. Setting and notation", H1)]
    story += [P("For each underlying and Totem date the controller has three inputs:")]
    story += [bullets([
        "<b>Totem options.</b> For each expiry <i>T<sub>k</sub></i> (here eleven expiries from 1M to 5Y): consensus "
        "call and put prices on a strike grid, the number of accepted contributors <i>n<sub>A,k</sub></i>, the dispersion "
        "of contributor forwards <i>σ<sub>F,k</sub></i> and the reference spot.",
        "<b>Totem dividends.</b> For each period <i>y</i> (calendar years): the consensus sum of dividends going ex in "
        "the period, its dispersion <i>σ<sub>D,y</sub></i> and contributor count <i>n<sub>B,y</sub></i>.",
        "<b>FO marks.</b> Spot <i>S</i>, the rate curve, the repo zero curve at the Totem expiries, the dividend forecast "
        "(ex-dates <i>t<sub>i</sub></i>, cash amounts <i>d<sub>i</sub></i>) and bucketed sensitivities: repo by pillar, "
        "dividends by period.",
    ])]
    story += [P("With discrete cash dividends and continuously compounded rate and repo zero curves, the forward is")]
    story += [equation(r"F(T) = G(T)\,\left[S - \sum_{t_i \leq T} \frac{d_i}{G(t_i)}\right],\qquad G(t) = e^{R(t) - B(t)},\quad R(t) = r(t)\,t,\quad B(t) = b(t)\,t")]
    story += [P(
        "<i>D</i>(<i>T</i>) = Σ<sub><i>t<sub>i</sub></i> ≤ <i>T</i></sub> <i>d<sub>i</sub></i>/<i>G</i>(<i>t<sub>i</sub></i>) is the "
        "carry-discounted cumulative dividend. Detection works in <b>carry space</b>, on the cumulative repo "
        "<i>B</i>(<i>T</i>) = <i>b</i>(<i>T</i>)·<i>T</i>, and on its Totem − FO difference:")]
    story += [equation(r"y_k = \Delta B_k = B_{\mathrm{Totem}}(T_k) - B_{\mathrm{FO}}(T_k) = \left[b_{\mathrm{Totem}}(T_k) - b_{\mathrm{FO}}(T_k)\right]T_k")]
    story += [P(
        "Carry is where the mechanisms add up and nothing is amplified: a constant error <i>ε</i> in carry is an error "
        "<i>ε/T</i> in repo, which is why the short end of an implied repo curve looks so noisy.")]
    story += [P(
        "<b>Implied repo.</b> Pillars are solved in order with flat forward repo between pillars. The model forward at "
        "pillar <i>k</i> is strictly decreasing in the unknown <i>B<sub>k</sub></i> (∂<i>F</i>/∂<i>B<sub>k</sub></i> ≤ −<i>F</i>), "
        "so a safeguarded Newton step on ln <i>F</i> converges in a few iterations. The bootstrap reproduces a known repo "
        "curve to 10<super>−12</super> in the tests, including ex-dates on pillars and negative repo.")]
    story += [P("<b>Forwards from consensus prices.</b> For European options")]
    story += [equation(r"C(K) - P(K) = DF\,\left(F - K\right)")]
    story += [P(
        "so <i>F</i> is the strike where the call and put prices cross, whatever the discount factor. The fitted <i>DF</i> is "
        "kept as a diagnostic (an implied funding spread) and the controller's own rate curve is used on both sides, so "
        "rates cancel in Δ<i>B</i>. Totem forwards are expressed on the FO spot before the bootstrap.")]
    story += [P(
        "<b>Dividend buckets.</b> Totem dividend levels are per period. The controller spreads each level over the FO "
        "ex-date profile pro rata; the allocation matrix <i>A<sub>ky</sub></i> = ∂<i>D<sub>k</sub></i>/∂<i>L<sub>y</sub></i> "
        "records how much of period <i>y</i> goes ex before expiry <i>k</i>.")]

    # ================================================== 2. why irregular
    story += [CondPageBreak(60 * mm), P("2. Why Totem-implied repo curves are irregular", H1)]
    story += [P("2.1 The panel gap", H2)]
    story += [P(
        "Each contributor <i>j</i> is internally consistent: ln <i>F<sub>j</sub></i> = ln(<i>S</i> − <i>D<sub>j</sub></i>) + "
        "<i>R</i> − <i>B<sub>j</sub></i>. The forward consensus averages the option panel <i>A</i>; the dividend consensus "
        "averages a different panel <i>B</i>. Linearising around a common centre:")]
    story += [equation(r"b_{\mathrm{implied}}(T) \approx \overline{b}_A(T) + \frac{\overline{D}_A(T) - \overline{D}_B(T)}{T\,(S - D(T))}")]
    story += [P(
        "The implied repo is the option panel's average repo plus the gap between two dividend views, divided by "
        "<i>T</i>. For one contributor, forward and dividend errors cancel in the implied repo; across two panels they "
        "do not. For disjoint panels the gap has standard deviation")]
    story += [equation(r"\mathrm{sd}\left[b_{\mathrm{implied}} - \overline{b}_A\right] = \frac{\sigma_D(T)\,\sqrt{1/n_A + 1/n_B}}{T\,(S - D(T))}")]
    rows = [["Tenor", "D(T)", "σ<sub>D</sub>(T)", "n<sub>A</sub>", "n<sub>B</sub>", "Formula (bp)", "Simulation (bp)"]]
    for r in gap_rows:
        rows.append([r[0], num(r[1], 1), num(r[2], 2), r[3], r[4], num(r[5], 1), num(r[6], 1)])
    story += [table(rows, [18 * mm, 18 * mm, 20 * mm, 14 * mm, 14 * mm, 26 * mm, 28 * mm],
                    caption="Panel-gap noise for an EU single stock (spot 100, 4% yield, contributor dispersion "
                            "1% + 3%·T of the cumulative dividend, panel sizes of the synthetic EU stock category). "
                            "Closed form (5) against a contributor-level simulation with no repo disagreement.")]
    story += [figure(f_gap, "(a) Repo noise from the panel gap alone, by panel size. (b) A spot snapshot mismatch "
                            "<i>ε</i> is constant in carry, so it reads as <i>ε/T</i> in repo: 120bp at 1M for a "
                            "0.1% mismatch.")]
    story += [P("2.2 Contributor drop-off", H2)]
    story += [P(
        "Panels are nested: contributors who quote a long expiry also quote the shorter ones, and fewer quote the long "
        "end. When the panel shrinks from <i>n</i> to <i>m</i> contributors between two expiries, the consensus jumps by "
        "itself with")]
    story += [equation(r"\mathrm{sd}\left[\overline{x}_m - \overline{x}_n\right] = \sigma\,\sqrt{\frac{1}{m} - \frac{1}{n}}")]
    story += [P(
        "in units of the contributor dispersion <i>σ</i>: "
        + ", ".join(f"{a}→{b}: {num(c, 2)}σ" for a, b, c, _ in jump_rows)
        + ". A jump of that size at a panel drop is noise, not information; the noise model of Section 4 reproduces it "
        "exactly.")]
    story += [P("2.3 Mechanism catalogue", H2)]
    story += [P(
        "The structured mechanisms each leave their own signature. The signature is clearest in carry space, which is "
        "what the detector exploits:")]
    mech = [
        ["Mechanism", "What happens", "Signature in repo", "Signature in carry ΔB", "Template"],
        ["Panel gap", "Different option and dividend panels", "Noise growing with dividend uncertainty, ×1/T", "Smooth part plus increments at ex-dates", "Noise model"],
        ["Contributor drop-off", "Panel shrinks between two expiries", "Jump at the drop", "Step", "Step"],
        ["Ex-date timing split", "Part of the option panel puts an ex-date on the other side of an expiry", "Spike of p·d / (T(S−D)) at one pillar", "Spike", "Spike (timing)"],
        ["Stale ex-date profile", "Bucket allocation uses a calendar that differs from contributors'", "Zigzag at in-season pillars", "Spikes at pillars near ex-dates", "Season"],
        ["Dividend bucket error", "One Totem period off", "Jump after the ex-dates, decaying as 1/T", "Step from the first pillar after the ex-date", "Step"],
        ["Spot snapshot, front dividend", "Reference spot differs from FO spot; dividend before the first expiry", "ε/T hyperbola", "Constant", "Level"],
        ["Forward outlier", "Bad consensus call/put at one expiry", "Spike", "Spike", "Spike (outlier)"],
        ["Gross/net definition", "Dividend service and forwards use different dividend bases", "Steps at every ex-date", "Ramp of steps; close to a repo slope for an index", "Step / null"],
        ["Parity extraction", "American exercise, imposed DF, poor strike coverage", "Pillar-specific", "Spike", "Spike + diagnostics"],
        ["Genuine FO mismark", "Desk repo differs from consensus", "Smooth level or slope", "≈ β·T", "Null model (reported, not flagged)"],
    ]
    story += [table(mech, [27 * mm, 41 * mm, 37 * mm, 37 * mm, 30 * mm], caption="Mechanisms and their signatures.",
                    align_right_from=99, zebra=False)]
    story += [P("2.4 Identifiability", H2)]
    story += [P(
        "Option prices identify the forward, that is the total carry; a dividend source is needed to split repo out of "
        "it. The panel gap is the difference between an observed dividend consensus and an <i>unobserved</i> dividend "
        "view, the option panel's. It cannot be estimated from consensus data, so no spline, clustering or neural network "
        "can remove it. Three things can be done: model its size (the noise model), recognise the structured mechanisms by "
        "their signature (the templates), and avoid needing the split where the exposure does not (Section 3).")]

    # ============================================== 3. control identity
    story += [CondPageBreak(60 * mm), P("3. Repo control from forward and dividend controls", H1)]
    story += [P("3.1 The identity", H2)]
    story += [P(
        "For a position whose value depends on the underlying only through its forwards (European options, forwards, "
        "futures), a dividend change and a repo change act through the same channel. Let <i>s<sub>D</sub></i> and "
        "<i>s<sub>b</sub></i> be the FO sensitivities to the dividend periods and the repo pillars, Δ<i>L</i> the "
        "Totem − FO dividend levels, and <i>J<sub>ky</sub></i> = ∂<i>b<sub>k</sub></i>/∂<i>L<sub>y</sub></i> the "
        "response of the implied repo to the dividend levels with the Totem forwards held fixed (from the bootstrap). "
        "With subscripts T for Totem:")]
    story += [equation(r"\mathrm{div\ line} = s_D \cdot \Delta L,\quad \mathrm{repo\ line} = s_b \cdot \left[b(F_T,D_T) - b_{FO}\right],\quad \mathrm{forward\ line} = s_b \cdot \left[b(F_T,D_{FO}) - b_{FO}\right]", size=10.5)]
    story += [equation(r"\mathrm{div\ line} + \mathrm{repo\ line} \;\approx\; \mathrm{forward\ line} + g \cdot \Delta L,\qquad g = s_D + s_b\,J")]
    story += [bullets([
        "For forward-only exposure <i>g</i> = 0: the total is the forward line. It uses Totem forwards and FO "
        "dividends, so it needs neither the Totem dividends nor a Totem repo curve.",
        "Dividend-panel noise reaches the total only through <i>g</i>. <i>g</i> comes from the FO sensitivities and the "
        "bootstrap Jacobian; no repricing is needed.",
        "The split between the dividend line and the repo line still depends on the dividend panel. Large offsetting "
        "lines with a small total (an offset ratio near 1) point to a split artefact, not a mismark.",
    ])]
    story += [P("3.2 Verification", H2)]
    rows = [["Totem dividend", "Implied repo (bp)", "Div line", "Repo line", "Total", "Full revaluation"]]
    for r in cancel_rows:
        rows.append([num(r[0], 1), num(r[1], 0), num(r[2], 3), num(r[3], 3), num(r[4], 3), num(r[5], 3)])
    story += [table(rows, [28 * mm, 30 * mm, 24 * mm, 24 * mm, 24 * mm, 30 * mm],
                    caption="1Y at-the-money call (spot 100, rate 3%, vol 25%, FO dividend 3.5, FO repo 60bp), "
                            "Totem forward 20bp below FO, three dividend panels. The implied repo moves by 125bp; the "
                            "total does not.")]
    rows = [["Book", "Div panel shift", "Div line", "Repo line", "Total", "Forward line", "Non-fwd line", "Full reval"]]
    for r in book_rows:
        if r[1] == "materiality":
            continue
        rows.append([r[0].split(":")[0], num(r[1], 1), num(r[2]), num(r[3]), num(r[4]), num(r[5]), num(r[6]), num(r[7])])
    story += [table(rows, [14 * mm, 24 * mm, 20 * mm, 20 * mm, 20 * mm, 24 * mm, 26 * mm, 24 * mm],
                    caption="Three books priced with the prototype pricer (eleven repo pillars, six dividend periods, "
                            "Totem forwards 20bp below FO): A = calls, puts and a forward; B = A + a short 2028 dividend "
                            "swap; C = A + a 1Y stock-loan fee. The Totem dividend panel moves the 2027 and 2028 periods. "
                            "Total = forward line + non-forward line, and both match full revaluation.")]
    story += [figure(f_cancel, "Dividend line, repo line and their total as the Totem dividend panel moves. For "
                               "forward-only exposure (left) the total is flat and equal to the forward line; a dividend "
                               "swap (right) adds a genuine dividend exposure, which is exactly the non-forward line.")]
    story += [P("3.3 Materiality of the split", H2)]
    story += [P(
        "With Σ<sub><i>L</i></sub> the covariance of the Totem dividend consensus, the 1-sd impact of dividend-panel "
        "noise on the total is √(<i>g</i>′Σ<sub><i>L</i></sub><i>g</i>), against √(<i>s<sub>D</sub></i>′Σ<sub><i>L</i></sub>"
        "<i>s<sub>D</sub></i>) for the dividend line alone. The ratio tells how much of the split risk is real for a "
        "given book:")]
    rows = [["Book", "Div line sd", "Total sd (split risk)", "Cancels"]]
    for r in (A_book, B_book, C_book):
        rows.append([r[0], num(r[2], 3), num(r[3], 3), pct(r[4], 1)])
    story += [table(rows, [70 * mm, 30 * mm, 40 * mm, 30 * mm],
                    caption="Split materiality per book (dividend consensus dispersion 5% of each period, 10 "
                            "contributors, correlation e<super>−|Δy|/2</super>).")]
    story += [P("3.4 Practical notes", H2)]
    story += [bullets([
        "<b>First order.</b> When |Δ<i>F</i>/<i>F</i>| is large (hard-to-borrow names, long tenors), take the total from "
        "full revaluation; the decomposition remains valid as an attribution.",
        "<b>Bucket alignment.</b> FO repo pillars should coincide with the Totem expiries, or the repo sensitivities be "
        "mapped linearly onto them; dividend sensitivities should use the same periods as the Totem buckets.",
        "<b>Between pillars.</b> Positions expiring between Totem expiries close to an ex-date depend on the allocation "
        "profile; the season and timing templates watch exactly that.",
        "<b>Non-forward exposure.</b> Dividend swaps and futures are pure dividend; stock borrow and TRS financing legs "
        "are pure repo; American exercise and exotics with discrete-dividend modelling carry both in their own proportions.",
        "<b>Prudent valuation.</b> If dividend and repo uncertainties are aggregated without netting (market price "
        "uncertainty per valuation input), split artefacts are counted twice. Defining the input as the forward where "
        "<i>g</i> ≈ 0 removes the double count.",
    ])]

    # ================================================== 4. detection
    story += [CondPageBreak(60 * mm), P("4. Structural detection of bad curves", H1)]
    story += [P("4.1 Observation and null model", H2)]
    story += [P(
        "The detector observes <i>y</i> = Δ<i>B</i> at the Totem expiries. Without an anomaly, Totem and FO may still "
        "disagree on repo, smoothly; the null model is a smooth repo difference "
        "δ<i>b</i>(<i>T</i>) = <i>β</i><sub>0</sub> + <i>β</i><sub>1</sub> ln <i>T</i> plus consensus noise:")]
    story += [equation(r"y = X\beta + \varepsilon,\qquad X = \left[\,T,\ T\ln T\,\right],\qquad \varepsilon \sim \mathcal{N}(0,\ \kappa\Sigma)")]
    story += [P(
        "<i>β</i><sub>0</sub> is the repo difference at 1Y. It is estimated and reported with its standard error but never "
        "flagged as a Totem anomaly.")]
    story += [P("4.2 Noise model from Totem statistics", H2)]
    story += [P(
        "Σ = Σ<super>F</super> + Σ<super>D</super> uses only what Totem publishes. For the option panel, the contributor "
        "forward dispersion <i>s<sub>k</sub></i> = <i>σ<sub>F,k</sub></i>/<i>F<sub>k</sub></i> is split into a persistent repo "
        "part <i>σ<sub>b</sub>T<sub>k</sub></i> (with <i>σ<sub>b</sub></i> read off the shortest expiries, where dividend "
        "uncertainty is negligible) and a cumulative dividend part: a persistent factor plus independent increments at "
        "ex-dates.")]
    story += [equation(r"\Sigma^F_{kl} = \frac{\sigma_b^2\,T_k T_l + w\,s'_k s'_l + (1-w)\,\min(s'_k, s'_l)^2}{\max(n_{A,k},\ n_{A,l})},\qquad s'^2_k = s_k^2 - \sigma_b^2 T_k^2", size=13)]
    story += [equation(r"\Sigma^D = J_A\,\Sigma_L\,J_A^{\prime},\qquad (J_A)_{ky} = \frac{A_{ky}}{S - D_k},\qquad (\Sigma_L)_{yz} = \frac{\rho_{yz}\,\sigma_{D,y}\,\sigma_{D,z}}{\max(n_{B,y},\ n_{B,z})}", size=13)]
    story += [bullets([
        "Dividing by the larger panel is the covariance of nested panel means; it reproduces the drop-off jump variance (6).",
        "An isolated spike in forward dispersion is capped at 1.5× the larger neighbour: contributors who disagree "
        "about an ex-date inflate the dispersion at exactly the pillar where the mean moves, and leaving it in would hide "
        "the anomaly. The ratio is reported with the explanation.",
        "<i>κ</i> is the only parameter the anomaly test learns from history, one per region × category. It is set so "
        "that the alarm rate on clean history equals the target. The test statistics scale as <i>κ</i><super>−1/2</super> "
        "and <i>κ</i><super>−1</super> while the family-wise critical value does not move, so the calibration is an exact "
        "one-dimensional search. The FO-mismark test (4.4) gets its own standard-error scale, calibrated the same way.",
    ])]
    story += [P("4.3 Templates and test", H2)]
    tmpl = [
        ["Template", "Vector in carry space", "Mechanisms", "Amplitude <i>a</i> in natural units"],
        ["Level", "1 at every expiry", "Spot or reference mismatch; dividend before the first expiry", "Spot error ≈ −a; front dividend error ≈ −a(S − D<sub>1</sub>)"],
        ["Step k", "1 from T<sub>k</sub> on", "Dividend amount in (T<sub>k−1</sub>, T<sub>k</sub>]; panel change at T<sub>k</sub>", "Dividend error ≈ −a(S − D<sub>k</sub>)"],
        ["Spike k", "1 at T<sub>k</sub> only", "Forward outlier; ex-date timing across T<sub>k</sub>", "ln F error ≈ −a; share of the nearby dividend ≈ −a(S − D<sub>k</sub>)/d<sub>i</sub>"],
        ["Season", "Σ<sub>i</sub> d<sub>i</sub> K<sub>h</sub>(T<sub>k</sub> − t<sub>i</sub>)/(S − D<sub>k</sub>)", "Ex-date profile shift in the bucket allocation", "Shift ≈ 365·a days"],
    ]
    story += [table(tmpl, [20 * mm, 42 * mm, 56 * mm, 54 * mm], caption="Anomaly templates (K<sub>h</sub>: Gaussian kernel, h = 20 days).",
                    align_right_from=99)]
    story += [P(
        "With <i>κ</i>Σ = <i>LL</i>′ and <i>P</i> the projection that removes the null model in whitened space, each "
        "template gets a matched-filter (generalised likelihood ratio) statistic, and the residual gets a global "
        "chi-square:")]
    story += [equation(r"z_t = \frac{\tilde{t}^{\prime}\,r}{\Vert\tilde{t}\Vert},\qquad r = P\,L^{-1}y,\quad \tilde{t} = P\,L^{-1}t,\qquad Q = r^{\prime}r \sim \chi^2_{K-2}")]
    story += [P(
        "A curve is flagged when max<sub><i>t</i></sub>|<i>z<sub>t</sub></i>| exceeds the family-wise critical value, simulated "
        "from the exact correlation of the <i>z<sub>t</sub></i>, or when <i>Q</i> exceeds its chi-square quantile (half the "
        "false-alarm budget each). <i>Q</i> catches roughness that no template explains.")]
    story += [P("4.4 Attribution, repair and the FO-mismark test", H2)]
    story += [P(
        "Templates are selected greedily (matching pursuit) while the best remaining one exceeds the critical value, then "
        "refitted jointly with the null model. Each amplitude is translated into natural units using the context: "
        "ex-dates near the pillar, contributor counts on both panels, dispersion ratios. The repaired curve removes the "
        "selected components, <i>B</i><sub>repaired</sub> = <i>B</i><sub>Totem</sub> − Σ<i>a<sub>t</sub>t</i>. Separately, "
        "|<i>β</i><sub>0</sub>| above its critical value and above 10bp reports the FO repo as a candidate mismark against "
        "the Totem consensus. That standard error rests on dispersions estimated from a handful of contributors, so the "
        "test uses a Student-t critical value with <i>n<sub>A</sub></i> − 1 degrees of freedom.")]
    story += [P("4.5 Rules of thumb for reviewers", H2)]
    story += [bullets([
        "Rough differences in carry come from Totem; smooth, persistent differences are candidate FO mismarks.",
        "Irregularity aligned with the dividend calendar points to the dividend side: timing, allocation or definition.",
        "A 1/T-shaped repo difference at every tenor is a spot or snapshot problem.",
        "A jump at a contributor drop-off smaller than σ√(1/m − 1/n) is noise.",
        "A dispersion spike at one expiry means contributors disagree, usually about an ex-date.",
    ])]

    story += [figure(f_decomp, "Carry-space view of the Totem dividend bucket error shown in Figure 4: observed Totem − FO "
                               "carry, the null fit with its noise band, and the null fit plus the selected template. "
                               f"Explanation: {decomp_note[1]}")]
    story += [figure(f_gallery, "Detected and explained anomalies on synthetic EU single stocks (same detector settings "
                                "as the benchmark). Each panel shows the FO repo, the Totem-implied repo and, when an anomaly "
                                "is found, the implied repo with the anomaly removed; the text is the detector's first "
                                "explanation. In the last panel the Totem curve is fine and the FO repo is shifted: no "
                                "anomaly is raised and the FO mismark is reported. The band is per pillar; the FO-mismark "
                                "test pools all pillars, so it can be significant while single points sit inside the band.")]
    # ================================================== 5. benchmark
    story += [CondPageBreak(60 * mm), P("5. Benchmark on a synthetic Totem market", H1)]
    story += [P("5.1 Design", H2)]
    story += [P(
        "The synthetic market is built bottom-up from contributors so that every mechanism of Section 2 arises the way "
        "it does in Totem, not as an added curve shape.")]
    cat_rows = [["Category", "Dividends", "Yield", "Repo (bp, short → long)", "Option panel", "Dividend panel"]]
    cal = {"eu_annual": "Annual, Apr–May (35% with a Nov interim)", "us_quarterly": "Quarterly", "mixed": "Annual or quarterly",
           "index_eu": "Monthly stream, 60% in Apr–Jun"}
    for name, spec in CATEGORIES.items():
        cat_rows.append([name, cal[spec.calendar], f"{spec.div_yield[0]:.1%}–{spec.div_yield[1]:.1%}",
                         f"{spec.repo_short_bp[0]}–{spec.repo_short_bp[1]} → {spec.repo_long_bp[0]}–{spec.repo_long_bp[1]}",
                         f"{spec.n_opt[0]} → {spec.n_opt[-1]}", f"{spec.n_div[0]} → {spec.n_div[-1]}"])
    story += [table(cat_rows, [22 * mm, 46 * mm, 20 * mm, 34 * mm, 24 * mm, 26 * mm],
                    caption="Synthetic categories. Panels shrink with tenor (nested); eleven Totem expiries from 1M to 5Y, "
                            "six dividend periods (rest of 2026 to 2031), valuation date 14 October 2026.",
                    align_right_from=2)]
    story += [bullets([
        "<b>Contributors.</b> Option and dividend panels are drawn independently. Each contributor holds internally "
        "consistent dividend views (a persistent factor plus per-dividend noise, dispersion growing with time) and a "
        "smooth repo view. Option contributors price calls and puts at five strikes with their own forward, a skewed "
        "vol surface, vol noise and price noise.",
        "<b>Consensus.</b> Mean after a median/MAD screen, with dispersion and counts reported as Totem would; the "
        "controller pipeline then runs unchanged (parity, allocation, bootstrap, noise model).",
        "<b>FO marks.</b> True dividends with 1% noise and true repo with a 3bp level error (one standard deviation).",
    ])]
    scen_rows = [["Scenario", "Injection", "Magnitude"],
                 ["Spot snapshot mismatch", "Option contributors price off a different spot", "±0.03% to ±0.15%"],
                 ["Totem dividend bucket error", "One period (2027–2029) off in the dividend consensus", "±0.15 to ±0.5 per 100 of spot"],
                 ["Ex-date timing split", "35–70% of the option panel moves the dividend closest to an expiry to its other side", "One dividend"],
                 ["Stale ex-date profile", "Allocation profile shifted so one dividend crosses an expiry", "3–20 days past the expiry"],
                 ["Forward outlier", "Consensus calls at one expiry shifted", "3–6 sd of the carry noise"],
                 ["Biased long-dated sub-panel", "Contributors who stay after the largest panel drop are biased on later dividends", "±8% to ±20%"],
                 ["FO repo mismark (genuine)", "FO repo shifted in parallel; Totem untouched", "±15 to ±50bp"],
                 ["Gross/net dividend bias", "Every Totem period scaled", "±8% to ±18%"]]
    story += [table(scen_rows, [45 * mm, 85 * mm, 42 * mm], caption="Scenarios. Each test name is simulated under every "
                    "scenario with the same seed, so differences between scenarios are paired.", align_right_from=99)]
    story += [P(
        f"<b>Detectors.</b> {DETECTOR_TITLES['naive_level']}: largest Totem − FO repo gap over the expiries. "
        f"{DETECTOR_TITLES['naive_shape']}: largest deviation of the Totem implied repo from the interpolation of its "
        "two neighbours. B-spline + GMM: least-squares cubic B-spline (six coefficients, knots in ln <i>T</i>) and a "
        "Gaussian mixture per category with the number of components chosen by BIC, applied to the Totem curve (the "
        "starter idea) and to the Totem − FO difference. Structural: Section 4.")]
    story += [P(
        f"<b>Protocol.</b> {res.n_train} clean names per category calibrate every detector to {pct(res.fpr_target)} "
        f"false alarms (a quantile for the baselines, <i>κ</i> for the structural detector); {res.n_test} fresh names per "
        "category are then simulated under each scenario. The signal-to-noise of an injection is the norm of the whitened, "
        "null-projected difference between the scenario and the clean carry of the same name.")]

    story += [P("5.2 Results", H2)]
    rows = [["Scenario"] + [DETECTOR_TITLES[d] for d in DETECTORS]]
    for s, rates in ra.detection_table(res):
        rows.append([SCENARIO_TITLES[s]] + [pct(x) for x in rates])
    rows.append(["<b>Mean over the six Totem anomalies</b>"] + [f"<b>{pct(mean_det[d])}</b>" for d in DETECTORS])
    story += [table(rows, [46 * mm, 24 * mm, 24 * mm, 26 * mm, 26 * mm, 26 * mm],
                    caption=f"Share of names flagged, all detectors at {pct(res.fpr_target)} false alarms on clean "
                            "history. The clean row is the realised false-alarm rate on fresh names. For the FO mismark "
                            "and the gross/net bias, a flag is not the desired outcome (see text).")]
    story += [figure(f_det, "Share of names flagged per scenario and detector (same data as Table 9).")]
    story += [figure(f_power, "Detection rate against the signal-to-noise of the injected anomaly, pooled over the six "
                              "Totem-side anomaly types; n is the number of names in each bin.")]
    rows = [["Scenario"] + [DETECTOR_TITLES[d] for d in DETECTORS]]
    for s, aucs in ra.auc_table(res):
        rows.append([SCENARIO_TITLES[s]] + [f"{x:.2f}" for x in aucs])
    story += [table(rows, [46 * mm, 24 * mm, 24 * mm, 26 * mm, 26 * mm, 26 * mm],
                    caption="Area under the ROC curve against the clean names (threshold-free ranking quality).")]
    story += [figure(f_attr, "Which mechanism the structural detector names first, per scenario. Level, step, spike "
                             "and season are the template families; roughness means the global test fired without a "
                             "template; no alarm means the name was not flagged.")]
    rows = [["Scenario", "Flagged", "Right mechanism (of flagged)", "FO-mismark flag"]]
    for r in ra.attribution_table(res):
        rows.append([SCENARIO_TITLES[r["scenario"]], pct(r["detected"]), pct(r["attr_cond"]), pct(r["fo_flag"])])
    story += [table(rows, [62 * mm, 30 * mm, 48 * mm, 32 * mm],
                    caption="Structural detector: detection, attribution (first explanation names the injected "
                            "mechanism and its location) and the FO-mismark test.")]
    rows = [["Category"] + [DETECTOR_TITLES[d] for d in DETECTORS] + ["κ"]]
    for c, fprs in ra.fpr_by_category(res):
        rows.append([c] + [pct(x, 1) for x in fprs] + [num(res.variance_scale[c], 2)])
    story += [table(rows, [24 * mm, 24 * mm, 24 * mm, 26 * mm, 26 * mm, 26 * mm, 18 * mm],
                    caption=f"Realised false-alarm rate on fresh clean names by category (target {pct(res.fpr_target)}) "
                            "and the calibrated variance scale κ.")]

    story += [P("5.3 Reading the results", H2)]
    story += [bullets(_reading_bullets(res, mean_det, fo_flag, fo_mismark_rate, fo_mismark_clean, pw, attr_pooled))]
    story += [P("5.4 Limits of the evidence", H2)]
    story += [P(
        "The synthetic market reproduces the mechanisms by construction; it does not prove that real Totem data contains "
        "only these. Real data will also carry stale contributors, corporate actions and special dividends, American "
        "exercise in single-stock quotes and data-entry errors. The variance scale absorbs a misspecified noise level but "
        "not a misspecified structure, and the magnitudes, while plausible, are not calibrated to your history. The "
        "benchmark shows that the method works when the mechanisms are present and is calibrated when they are not; "
        "validation on 12–24 months of real snapshots with labelled incidents is the next step.")]

    # ================================================= 6. alternatives
    story += [CondPageBreak(60 * mm), P("6. Where B-spline + GMM and neural networks fit", H1)]
    story += [bullets([
        "<b>B-spline coefficients as features.</b> With eleven expiries and six coefficients the spline mostly "
        "re-parameterises the curve, and least squares smears a one-pillar anomaly over several coefficients. Clustering "
        "by region × category then separates levels (general collateral, hard to borrow, index) more than anomalies. In "
        f"the benchmark GMM on the curve flags {pct(mean_det['spline_gmm_curve'])} of Totem anomalies on average, against "
        f"{pct(mean_det['structural'])} for the structural detector.",
        "<b>GMM on the difference</b> is close to the structural detector on average detection "
        f"({pct(mean_det['spline_gmm_diff'])} against {pct(mean_det['structural'])}), but it cannot tell a bad Totem curve "
        f"from an FO that disagrees: it flags {pct(fo_flag['spline_gmm_diff'])} of genuine FO repo mismarks as bad curves, "
        "and it gives no mechanism, no amplitude and no repaired curve.",
        "<b>Where they help.</b> Use a P-spline as the smooth part of the null model if <i>β</i><sub>0</sub> + "
        "<i>β</i><sub>1</sub> ln <i>T</i> proves too rigid on real data. Run a GMM on the structural outputs (<i>β</i>, "
        "template <i>z</i>-scores, <i>Q</i>, dispersion ratios) to learn the usual regimes per region × category. Use "
        "region × category as calibration groups for <i>κ</i>, not as clusters.",
        "<b>Neural networks.</b> In the linear-Gaussian limit, a small network trained to detect one anomaly type learns "
        "the matched filter of Section 4, which can be written down and calibrated analytically. Networks earn their "
        "place once there is a labelled incident history and patterns the templates do not cover; train them on residuals "
        "of the structural model and on synthetic injections, not on raw curves.",
    ])]

    # ================================================ 7. implementation
    story += [CondPageBreak(60 * mm), P("7. Implementation", H1)]
    story += [P("7.1 Package layout", H2)]
    layout = [["Module", "Content"],
              ["curves", "Zero curves (flat forwards), discrete dividends, forwards, exact implied-repo bootstrap"],
              ["parity", "Forward and discount factor from consensus call/put prices, diagnostics"],
              ["buckets", "Dividend periods, allocation of Totem levels to ex-dates, allocation matrix"],
              ["noise", "Panel-gap and drop-off formulas, nested-panel covariance, forward-panel and carry covariance, dispersion capping"],
              ["pipeline", "Totem and FO snapshots → implied repo (Totem/Totem and Totem/FO dividends), ΔB, Σ, detection context"],
              ["detect", "Structural detector: templates, matched filters, family-wise calibration, κ calibration, attribution, repair, FO-mismark test"],
              ["control", "Control lines (div, repo, forward, non-forward), bootstrap Jacobian, split materiality"],
              ["pricer", "Minimal pricer (European options, forwards, dividend swaps, stock-loan fee) and bucketed sensitivities"],
              ["baselines", "Naive gap, naive shape, B-spline + GMM"],
              ["synthetic, benchmark", "Contributor-level synthetic Totem market, scenarios, matched-FPR benchmark"]]
    story += [table(layout, [32 * mm, 140 * mm], caption="Package <font name='Mono'>repo_control</font> (numpy and scipy; "
                    "scikit-learn for the GMM baseline; matplotlib and reportlab for this report). 22 unit tests.",
                    align_right_from=99)]
    story += [P("7.2 Production inputs", H2)]
    schema = [["Input", "Fields", "Shape"],
              ["Totem options", "Expiries; strikes; consensus call and put prices; accepted contributors; dispersion of contributor forwards (or an estimate); reference spot", "K expiries × M strikes"],
              ["Totem dividends", "Period edges; consensus level; dispersion; contributors (net of dividends already paid in the current period)", "Y periods"],
              ["FO market", "Spot at the Totem snapshot time; rate curve; repo zero curve at the Totem expiries; dividend forecast (ex-dates, cash amounts)", "K pillars; dividend list"],
              ["FO risk", "Repo sensitivities by pillar (mapped to Totem expiries); dividend sensitivities by Totem period; per book", "K + Y per book"],
              ["History", "12–24 monthly snapshots of the above, plus any known incidents", "for κ and validation"]]
    story += [table(schema, [28 * mm, 112 * mm, 32 * mm], caption="Inputs for a production run.", align_right_from=99)]
    story += [P(
        "If the Totem service does not publish the dispersion of contributor forwards, estimate it from the dispersion of "
        "deep in-the-money calls and puts, or calibrate it on history; the detector needs the right scale per expiry, and "
        "<i>κ</i> corrects a common bias.")]
    story += [P("7.3 Monthly procedure", H2)]
    story += [bullets([
        "Load the Totem and FO snapshots at the same valuation time; check the reference spot.",
        "Imply forwards by parity; review the implied discount factor, residuals and dispersion ratios.",
        "Allocate the Totem dividend periods on the FO ex-date profile; imply repo with Totem dividends and with FO dividends.",
        "Build Σ and run the structural detector with the <i>κ</i> of the region × category.",
        "Compute the control lines per book: forward line, dividend line, non-forward line, repo line; check split materiality.",
        "Review alarms with their explanations and FO-mismark candidates; use the repaired curve where a Totem anomaly is confirmed.",
        "Record the decision for each alarm; the labels feed the next calibration.",
    ], numbered=True)]
    story += [P("7.4 Calibration and governance", H2)]
    story += [P(
        "<i>κ</i> per region × category on clean history, refreshed quarterly; an alarm budget of "
        f"{pct(res.fpr_target)} of names per month; monitoring of the realised alarm rate and of the share of "
        "explanations accepted by reviewers. Every component has a closed-form check in the test suite (bootstrap round "
        "trip, parity, the control identity against full revaluation, the panel-gap and drop-off formulas, null "
        "calibration of the detector, recovery of each template).")]
    story += [P("Reproduce:", BODY)]
    story += [P("pip install -e .[benchmark,report,dev]<br/>pytest<br/>python scripts/run_benchmark.py<br/>"
                "PYTHONPATH=. python scripts/build_report.py", CODE)]

    # ================================================== 8. next steps
    story += [P("8. Next steps", H1)]
    story += [bullets([
        "Run the pipeline on 12–24 months of Totem and FO snapshots (inputs of Table 14) and compare its alarms with "
        "past escalations.",
        "Calibrate <i>κ</i> and the noise-model settings (factor weight, repo dispersion estimate, dividend correlation "
        "length) per region × category.",
        "Label historical incidents (spot synchronisation, dividend errors, ex-date timing) to measure attribution on real data.",
        "Add exchange anchors: index futures, dividend futures and total-return futures for indices; securities-lending "
        "fees for single stocks.",
        "Add templates for recurring real-data patterns (special dividends, corporate actions) if they appear.",
    ], numbered=True)]

    # ================================================== appendix
    story += [CondPageBreak(80 * mm), P("Appendix A. Derivations", H1)]
    story += [P("<b>A.1 Panel gap.</b> For contributor <i>j</i>, ln <i>F<sub>j</sub></i> ≈ ln <i>F</i><sub>0</sub> − "
                "δ<i>D<sub>j</sub></i>/(<i>S</i> − <i>D</i>) − <i>T</i>δ<i>b<sub>j</sub></i>. Averaging over panel <i>A</i> "
                "gives the consensus forward; the controller then uses the panel-<i>B</i> dividend: <i>b</i><sub>implied</sub><i>T</i> "
                "= <i>R</i> − ln <i>F̄<sub>A</sub></i> + ln(<i>S</i> − <i>D̄<sub>B</sub></i>) ≈ <i>b</i><sub>0</sub><i>T</i> + "
                "<i>T</i>·mean<sub><i>A</i></sub>(δ<i>b</i>) + [mean<sub><i>A</i></sub>(δ<i>D</i>) − mean<sub><i>B</i></sub>(δ<i>D</i>)]/(<i>S</i> − <i>D</i>), "
                "which is (4). For independent panels the gap variance is σ<sub><i>D</i></sub><super>2</super>(1/<i>n<sub>A</sub></i> + "
                "1/<i>n<sub>B</sub></i>), giving (5); formula and contributor-level simulation agree within a few percent (Table 1).")]
    story += [P("<b>A.2 Nested panels.</b> If panel <i>m</i> ⊂ panel <i>n</i> with i.i.d. views of variance σ<super>2</super>, "
                "Var(mean<sub><i>m</i></sub>) = σ<super>2</super>/<i>m</i>, Cov(mean<sub><i>m</i></sub>, mean<sub><i>n</i></sub>) = "
                "σ<super>2</super>/<i>n</i>, so Var(mean<sub><i>m</i></sub> − mean<sub><i>n</i></sub>) = σ<super>2</super>(1/<i>m</i> − 1/<i>n</i>), "
                "which is (6); with persistent views across tenors the covariance of the two means is ρσ<sub><i>k</i></sub>σ<sub><i>l</i></sub>/max(<i>n<sub>k</sub></i>, <i>n<sub>l</sub></i>).")]
    story += [P("<b>A.3 Control identity.</b> For <i>V</i> = <i>V</i>(<i>F</i>(<i>T</i><sub>1</sub>), …), "
                "<i>s<sub>b</sub></i> = (∂<i>V</i>/∂ln <i>F</i>)·∂ln <i>F</i>/∂<i>b</i> and <i>s<sub>D</sub></i> = (∂<i>V</i>/∂ln <i>F</i>)·∂ln <i>F</i>/∂<i>L</i>. "
                "Holding <i>F</i> fixed while <i>L</i> moves requires d<i>b</i> = <i>J</i> d<i>L</i> with "
                "∂ln <i>F</i>/∂<i>b</i>·<i>J</i> = −∂ln <i>F</i>/∂<i>L</i>, hence <i>s<sub>D</sub></i> = −<i>s<sub>b</sub>J</i> and <i>g</i> = 0. "
                "In general, div line + repo line = <i>s<sub>D</sub></i>Δ<i>L</i> + <i>s<sub>b</sub></i>[Δ<i>b</i><sup>F</sup> + <i>J</i>Δ<i>L</i>] "
                "= forward line + <i>g</i>Δ<i>L</i> to first order, where Δ<i>b</i><sup>F</sup> = <i>b</i>(<i>F<sub>T</sub></i>, <i>D<sub>FO</sub></i>) − <i>b<sub>FO</sub></i>.")]
    story += [P("<b>A.4 Bootstrap.</b> With <i>B</i>(<i>t</i>) linear between pillars, "
                "<i>F</i>(<i>B<sub>k</sub></i>) = <i>Se</i><super><i>R<sub>k</sub></i>−<i>B<sub>k</sub></i></super> − Σ<i>d<sub>i</sub></i>"
                "<i>e</i><super>(<i>R<sub>k</sub></i>−<i>R<sub>i</sub></i>)−(<i>B<sub>k</sub></i>−<i>B<sub>i</sub></i>)</super> and "
                "∂<i>F</i>/∂<i>B<sub>k</sub></i> = −<i>Se</i><super><i>R<sub>k</sub></i>−<i>B<sub>k</sub></i></super> + Σ<i>c<sub>i</sub>d<sub>i</sub></i><i>e</i><super>…</super> "
                "with <i>c<sub>i</sub></i> ∈ [0, 1], so ∂<i>F</i>/∂<i>B<sub>k</sub></i> ≤ −<i>F</i> &lt; 0.")]
    story += [P("<b>A.5 Test statistic.</b> Under the null, <i>r</i> = <i>PL</i><super>−1</super><i>y</i> is Gaussian with "
                "covariance <i>P</i> (rank <i>K</i> − 2), so <i>z<sub>t</sub></i> ~ N(0, 1) for each template and "
                "Cov(<i>z<sub>t</sub></i>, <i>z<sub>u</sub></i>) = <i>t̃</i>′<i>ũ</i>/(‖<i>t̃</i>‖‖<i>ũ</i>‖). The family-wise critical "
                "value is the (1 − α/2) quantile of max|<i>z</i>| simulated from that correlation; <i>Q</i> ~ χ²<sub><i>K</i>−2</sub>.")]

    # ======================================================= build
    doc = BaseDocTemplate(str(OUT), pagesize=A4, leftMargin=MARGIN_L, rightMargin=MARGIN_R, topMargin=19 * mm,
                          bottomMargin=17 * mm, title="Totem-implied repo: div-agnostic control and bad-curve detection",
                          author="Independent price verification (draft)",
                          subject="Implied repo control and bad-curve detection from Totem consensus")
    frame = Frame(MARGIN_L, 17 * mm, TEXT_W, PAGE_H - 36 * mm, id="body", leftPadding=0, rightPadding=0, topPadding=0, bottomPadding=0)
    doc.addPageTemplates([PageTemplate(id="page", frames=[frame], onPage=_decorate)])
    doc.build(_keep_headings(story))
    return OUT


def _reading_bullets(res, mean_det, fo_flag, fo_mismark_rate, fo_mismark_clean, pw, attr_pooled):
    d = res.detection_rate
    out = [
        f"<b>Forward outliers</b> are where carry space and the noise model pay most: {pct(d('fwd_outlier', 'structural'))} "
        f"detected, against {pct(d('fwd_outlier', 'naive_level'))} for the naive gap, which cannot see a 3–6 sd outlier at a "
        "long expiry under the long-end noise.",
        f"<b>Spot mismatches</b> ({pct(d('spot_sync', 'structural'))}) and <b>stale ex-date profiles</b> "
        f"({pct(d('season_profile', 'structural'))}) are easy for every method; the structural detector adds the "
        "explanation (Figure 7).",
        f"<b>Dividend bucket errors</b> ({pct(d('div_bucket', 'structural'))} against {pct(d('div_bucket', 'naive_level'))} "
        "naive) are limited by size: an error of 0.15–0.5 per 100 of spot is small next to long-dated dividend dispersion, "
        "and Figure 6 shows the miss rate is a signal-to-noise effect.",
        f"<b>Ex-date timing splits</b>: {pct(d('div_timing', 'structural'))} detected; the naive shape check "
        f"({pct(d('div_timing', 'naive_shape'))}) is competitive here because these spikes are large in repo space.",
        f"<b>A biased long-dated sub-panel</b> is close to non-identifiable for every method "
        f"({pct(d('panel_shift', 'structural'))} structural, {pct(d('panel_shift', 'naive_level'))} naive gap): the bias "
        "builds up smoothly after the panel drop and mostly projects onto a repo slope. This is the identifiability limit "
        "of Section 2.4, not a tuning issue.",
        f"<b>Genuine FO mismarks</b> are kept apart from bad Totem curves: {pct(d('fo_repo_mismark', 'structural'))} raise "
        f"an anomaly (the clean rate) and {pct(fo_mismark_rate)} are reported as FO-mismark candidates, against "
        f"{pct(fo_mismark_clean)} on clean names. The naive gap and GMM on the difference raise a bad-curve alarm on "
        f"{pct(fo_flag['naive_level'])} and {pct(fo_flag['spline_gmm_diff'])} of them. The FO-mismark test finds "
        f"{pct(res.fo_mismark_rate('fo_repo_mismark', 'EU stock'))} of the mismarks on EU stocks and "
        f"{pct(res.fo_mismark_rate('fo_repo_mismark', 'EU index'))} on the index, but only "
        f"{pct(res.fo_mismark_rate('fo_repo_mismark', 'HTB stock'))} on hard-to-borrow names: there contributors' repo "
        "views differ by about 60bp, so a 15–50bp FO difference is within the uncertainty of the consensus itself.",
        f"<b>Gross/net dividend bias</b> is flagged in {pct(d('div_gross_net', 'structural'))} of names, mostly as steps at "
        "ex-dates for discrete payers; for an index dividend stream it looks like a repo slope and is reported through "
        "<i>β</i> instead. Without an external view of the dividend definition the two cannot be separated.",
        f"<b>Power</b>: above a signal-to-noise of 4 the structural detector finds "
        f"{pct(min(pw('structural', lo) for lo in (4, 6, 8, 12)))}–{pct(max(pw('structural', lo) for lo in (4, 6, 8, 12)))} "
        "of anomalies, more than any baseline. Below 3, the naive gap and GMM on the difference detect more: they also "
        "react to the part of an anomaly that looks like a smooth repo difference, which the structural detector leaves "
        "to the FO-mismark test on purpose. That same property is why they raise alarms on genuine FO mismarks.",
        f"<b>Attribution</b>: when the structural detector flags a Totem anomaly, its first explanation names the injected "
        f"mechanism and location in {pct(attr_pooled)} of cases (Table 11).",
    ]
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--benchmark", default=str(ROOT / "results" / "benchmark.json.gz"))
    args = ap.parse_args()
    path = build(Path(args.benchmark))
    print(f"wrote {path} ({os.path.getsize(path) / 1e6:.1f} MB)")
