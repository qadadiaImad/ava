"""Figures of the methodology note (French and English), typeset with LaTeX fonts.

All figures follow one running example (a synthetic European stock simulated under every scenario with the
same seed) plus the benchmark results.
"""
from __future__ import annotations

import sys
from functools import lru_cache
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from matplotlib.ticker import FuncFormatter  # noqa: E402
from scipy.stats import norm  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import note_examples as ne  # noqa: E402

from repo_control.benchmark import DETECTORS, BenchmarkResult  # noqa: E402
from repo_control.control import control_lines, implied_repo_jacobian  # noqa: E402
from repo_control.curves import bootstrap_repo  # noqa: E402
from repo_control.detect import tenor_label  # noqa: E402
from repo_control.noise import panel_gap_sd  # noqa: E402
from repo_control.pricer import book_value, sensitivities  # noqa: E402
from repo_control.synthetic import EXPECTED_KIND, PILLARS, simulate_case  # noqa: E402

RUNNING_SEED = 110127
RUNNING_CATEGORY = "EU stock"
GALLERY = ["spot_sync", "div_bucket", "div_timing", "season_profile", "fwd_outlier", "panel_shift",
           "fo_repo_mismark", "div_gross_net"]

BLUE, ORANGE, GREEN, YELLOW, PINK = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"
SERIES = [BLUE, ORANGE, GREEN, YELLOW, PINK]
INK, INK2, MUTED, GRID, AXIS = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
BLUE_RAMP = ["#f4f8fd", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]

# ----------------------------------------------------------------- text
TXT = {
    "fr": {
        "mat": "Maturité (années)",
        "repo_pb": "Repo zéro-coupon (pb)",
        "carry_pb": r"\'Ecart de portage $\delta_k$ (pb du spot)",
        "fo": r"Repo FO $q^{\mathrm{fo}}$",
        "tot": r"Repo implicite Totem $q^{\mathrm{tot}}$",
        "repaired": "Totem réparé (anomalie retirée)",
        "band": r"FO $\pm 2\sigma$ du bruit de consensus",
        "obs": r"$\delta_k$ observé",
        "band_null": r"Ajustement lisse $\pm 2\sigma$",
        "smooth": r"Ajustement lisse $x\beta$",
        "with_tmpl": "Ajustement lisse + forme type retenue",
        "clean_a": "(a) Espace de repo : la courbe Totem paraît irrégulière",
        "clean_b": r"(b) Espace de portage : tout reste dans la bande de bruit",
        "gap_title": r"(a) \'Ecart de panel seul (aucun désaccord sur le repo)",
        "gap_y": "Bruit du repo implicite (pb)",
        "cfg": ["Panels de l'exemple fil rouge", "Panels larges : 25 options / 20 div.", "Panels réduits : 8 / 4"],
        "spot_title": r"(b) Décalage du spot $\varepsilon$ : erreur $\varepsilon/T$ sur le repo",
        "spot_y": "Erreur de repo (pb, échelle log.)",
        "spot_lab": r"$\varepsilon = {x}\,\%$",
        "div_line": "Ligne dividendes",
        "repo_line": "Ligne repo",
        "total": "Total (dividendes + repo)",
        "full": "Total, réévaluation complète",
        "book_a": "Portefeuille A : options et forward",
        "book_b": "Portefeuille B : A + swap de dividendes 2028",
        "cancel_x": "Décalage du panel dividendes Totem sur 2027 et 2028 (montant par période)",
        "cancel_y": "Ligne de contrôle (unités de P\\&L)",
        "decomp_title": r"Erreur sur la période de dividendes 2027, vue en espace de portage",
        "snr_x": r"Rapport signal sur bruit $\lambda$ de l'anomalie injectée",
        "det_rate": "Taux de détection",
        "theory": "Théorie (forme type connue)",
        "attr_title": "Mécanisme cité en premier par le détecteur (part des sous-jacents)",
        "kinds": ["Niveau", "Marche", "Pic", "Saison", "Rugosité", "Pas d'alerte"],
        "gallery": {
            "spot_sync": "(a) Décalage du spot de référence",
            "div_bucket": "(b) Erreur sur la période 2027",
            "div_timing": "(c) Désaccord sur la date de détachement",
            "season_profile": "(d) Profil de détachement périmé",
            "fwd_outlier": "(e) Forward aberrant à 2 ans",
            "panel_shift": "(f) Sous-panel long biaisé (non détecté)",
            "fo_repo_mismark": "(g) Erreur de marquage du repo FO",
            "div_gross_net": "(h) Biais brut/net sur les dividendes",
        },
        "detectors": {
            "naive_level": r"\'Ecart naïf $\max|\Delta q|$",
            "naive_shape": "Forme naïve",
            "spline_gmm_curve": "B-spline + mélange (courbe)",
            "spline_gmm_diff": "B-spline + mélange (écart)",
            "structural": "Détecteur structurel",
        },
        "scenarios": {
            "spot_sync": "Décalage du spot",
            "div_bucket": "Erreur sur une période",
            "div_timing": "Désaccord de date",
            "season_profile": "Profil périmé",
            "fwd_outlier": "Forward aberrant",
            "panel_shift": "Sous-panel long biaisé",
            "fo_repo_mismark": "Erreur de marquage FO",
            "div_gross_net": "Biais brut/net",
        },
    },
    "en": {
        "mat": "Maturity (years)",
        "repo_pb": "Zero repo (bp)",
        "carry_pb": r"Carry difference $\delta_k$ (bp of spot)",
        "fo": r"FO repo $q^{\mathrm{fo}}$",
        "tot": r"Totem-implied repo $q^{\mathrm{tot}}$",
        "repaired": "Totem repaired (anomaly removed)",
        "band": r"FO $\pm 2\sigma$ of consensus noise",
        "obs": r"observed $\delta_k$",
        "band_null": r"Smooth fit $\pm 2\sigma$",
        "smooth": r"Smooth fit $x\beta$",
        "with_tmpl": "Smooth fit + selected shape",
        "clean_a": "(a) Repo space: the Totem curve looks irregular",
        "clean_b": "(b) Carry space: everything stays inside the noise band",
        "gap_title": "(a) Panel gap alone (no repo disagreement)",
        "gap_y": "Implied repo noise (bp)",
        "cfg": ["Running example panels", "Large panels: 25 option / 20 div.", "Thin panels: 8 / 4"],
        "spot_title": r"(b) Spot mismatch $\varepsilon$: repo error $\varepsilon/T$",
        "spot_y": "Repo error (bp, log scale)",
        "spot_lab": r"$\varepsilon = {x}\%$",
        "div_line": "Dividend line",
        "repo_line": "Repo line",
        "total": "Total (dividend + repo)",
        "full": "Total, full revaluation",
        "book_a": "Book A: options and a forward",
        "book_b": "Book B: A + 2028 dividend swap",
        "cancel_x": "Shift of the Totem dividend panel on 2027 and 2028 (amount per period)",
        "cancel_y": "Control line (P\\&L units)",
        "decomp_title": "Error on the 2027 dividend period, seen in carry space",
        "snr_x": r"Signal-to-noise ratio $\lambda$ of the injected anomaly",
        "det_rate": "Detection rate",
        "theory": "Theory (known shape)",
        "attr_title": "Mechanism named first by the detector (share of names)",
        "kinds": ["Level", "Step", "Spike", "Season", "Roughness", "No alarm"],
        "gallery": {
            "spot_sync": "(a) Reference spot mismatch",
            "div_bucket": "(b) Error on the 2027 period",
            "div_timing": "(c) Ex-date disagreement",
            "season_profile": "(d) Stale ex-date profile",
            "fwd_outlier": "(e) Forward outlier at 2 years",
            "panel_shift": "(f) Biased long-dated sub-panel (missed)",
            "fo_repo_mismark": "(g) FO repo mismark",
            "div_gross_net": "(h) Gross/net dividend bias",
        },
        "detectors": {
            "naive_level": r"Naive gap $\max|\Delta q|$",
            "naive_shape": "Naive shape",
            "spline_gmm_curve": "B-spline + mixture (curve)",
            "spline_gmm_diff": "B-spline + mixture (difference)",
            "structural": "Structural detector",
        },
        "scenarios": {
            "spot_sync": "Spot mismatch",
            "div_bucket": "Error on one period",
            "div_timing": "Ex-date disagreement",
            "season_profile": "Stale profile",
            "fwd_outlier": "Forward outlier",
            "panel_shift": "Biased long sub-panel",
            "fo_repo_mismark": "FO repo mismark",
            "div_gross_net": "Gross/net bias",
        },
    },
}


def setup(lang: str) -> None:
    preamble = r"\usepackage[utf8]{inputenc}\usepackage[T1]{fontenc}\usepackage{lmodern}\usepackage{amsmath}"
    if lang == "fr":
        preamble += r"\usepackage{icomma}"
    plt.rcParams.update(
        {
            "text.usetex": True,
            "text.latex.preamble": preamble,
            "font.family": "serif",
            "font.size": 9,
            "axes.titlesize": 9.5,
            "axes.labelsize": 9,
            "legend.fontsize": 8,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "axes.edgecolor": AXIS,
            "axes.linewidth": 0.8,
            "axes.labelcolor": INK2,
            "axes.titlecolor": INK,
            "axes.grid": True,
            "axes.axisbelow": True,
            "grid.color": GRID,
            "grid.linewidth": 0.6,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "xtick.labelcolor": INK2,
            "ytick.labelcolor": INK2,
            "legend.frameon": False,
            "lines.linewidth": 1.6,
            "lines.markersize": 5,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )


def _num(x: float, lang: str, fmt: str = "{:g}") -> str:
    s = fmt.format(x)
    return f"${s.replace('.', '{,}')}$" if lang == "fr" else f"${s}$"


def _axis_fmt(ax, lang: str, axis: str = "both") -> None:
    f = FuncFormatter(lambda x, pos: _num(x, lang))
    if axis in ("x", "both"):
        ax.xaxis.set_major_formatter(f)
    if axis in ("y", "both"):
        ax.yaxis.set_major_formatter(f)


def tenors(lang: str):
    labels = [tenor_label(t) for t in PILLARS]
    return [t.replace("Y", "A") for t in labels] if lang == "fr" else labels


def _mk(color):
    return dict(color=color, marker="o", markeredgecolor="white", markeredgewidth=1.0)


def _save(fig, path: Path) -> Path:
    fig.savefig(path, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    return path


# ------------------------------------------------------------ running example
@lru_cache(maxsize=None)
def running_cases(scale: float, level_scale: float):
    from repo_control.detect import StructuralDetector

    det = StructuralDetector(alpha=0.02, variance_scale=scale, level_scale=level_scale)
    out = {}
    for scen in ["clean"] + GALLERY:
        case = simulate_case(RUNNING_SEED, RUNNING_CATEGORY, scen)
        im = case.implied
        out[scen] = (case, det.detect(im.pillars, im.delta_carry, im.cov, im.context()))
    return det, out


def running(res: BenchmarkResult):
    return running_cases(res.variance_scale[RUNNING_CATEGORY], res.level_scale.get(RUNNING_CATEGORY, 1.0))


def fig_clean(res: BenchmarkResult, path: Path, lang: str) -> Path:
    t = TXT[lang]
    det, cases = running(res)
    case, d = cases["clean"]
    im = case.implied
    x = np.arange(PILLARS.size)
    sd = np.sqrt(np.diag(im.cov) * det.variance_scale)
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.6))
    ax = axes[0]
    q_fo, q_tot = im.repo_fo * 1e4, im.repo_totem.zeros * 1e4
    ax.fill_between(x, q_fo - 2 * sd / PILLARS * 1e4, q_fo + 2 * sd / PILLARS * 1e4, color=BLUE, alpha=0.10, lw=0,
                    label=t["band"])
    ax.plot(x, q_fo, label=t["fo"], **_mk(BLUE))
    ax.plot(x, q_tot, label=t["tot"], **_mk(ORANGE))
    ax.set_ylim(-60, 60)
    ax.set_xticks(x)
    ax.set_xticklabels(tenors(lang), fontsize=7)
    ax.set_ylabel(t["repo_pb"])
    ax.set_title(t["clean_a"], loc="left", fontsize=8.5)
    _axis_fmt(ax, lang, "y")
    ax = axes[1]
    ax.axhline(0, color=AXIS, lw=0.8)
    ax.fill_between(x, -2 * sd * 1e4, 2 * sd * 1e4, color=BLUE, alpha=0.10, lw=0)
    ax.plot(x, im.delta_carry * 1e4, **_mk(INK), linestyle="none")
    ax.set_xticks(x)
    ax.set_xticklabels(tenors(lang), fontsize=7)
    ax.set_ylabel(t["carry_pb"])
    ax.set_title(t["clean_b"], loc="left", fontsize=8.5)
    _axis_fmt(ax, lang, "y")
    handles, labels = axes[0].get_legend_handles_labels()
    order = [1, 2, 0]
    fig.legend([handles[i] for i in order], [labels[i] for i in order], loc="upper center", ncol=3,
               bbox_to_anchor=(0.5, 1.08))
    fig.tight_layout(w_pad=2.0)
    return _save(fig, path)


def fig_gallery(res: BenchmarkResult, path: Path, lang: str) -> Path:
    t = TXT[lang]
    det, cases = running(res)
    fig, axes = plt.subplots(4, 2, figsize=(6.6, 8.6))
    x = np.arange(PILLARS.size)
    for ax, scen in zip(axes.ravel(), GALLERY):
        case, d = cases[scen]
        im = case.implied
        q_fo, q_tot = im.repo_fo * 1e4, im.repo_totem.zeros * 1e4
        band = 2 * np.sqrt(np.diag(im.cov) * det.variance_scale) / PILLARS * 1e4
        ax.fill_between(x, q_fo - band, q_fo + band, color=BLUE, alpha=0.10, lw=0, label=t["band"])
        if d.explanations:
            repaired = (im.delta_carry - d.anomaly_fit) / PILLARS * 1e4 + q_fo
            ax.plot(x, repaired, color=GREEN, lw=2.4, alpha=0.9, label=t["repaired"])
        ax.plot(x, q_fo, label=t["fo"], **_mk(BLUE))
        ax.plot(x, q_tot, label=t["tot"], **_mk(ORANGE))
        lo, hi = min(q_fo.min(), q_tot.min()), max(q_fo.max(), q_tot.max())
        pad = 0.12 * (hi - lo + 1)
        ax.set_ylim(max(lo - pad, -400 if scen != "season_profile" else lo - pad), hi + pad)
        ax.set_xticks(x)
        ax.set_xticklabels(tenors(lang), fontsize=6.5)
        ax.set_title(t["gallery"][scen], loc="left", fontsize=8.5)
        ax.set_ylabel(t["repo_pb"], fontsize=7.5)
        _axis_fmt(ax, lang, "y")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    want = [t["fo"], t["tot"], t["repaired"], t["band"]]
    order = [labels.index(w) for w in want if w in labels]
    fig.legend([handles[i] for i in order], [labels[i] for i in order], loc="upper center", ncol=2,
               bbox_to_anchor=(0.5, 1.04))
    fig.tight_layout(h_pad=1.2, w_pad=1.6)
    return _save(fig, path)


def fig_decomposition(res: BenchmarkResult, path: Path, lang: str) -> Path:
    t = TXT[lang]
    det, cases = running(res)
    case, d = cases["div_bucket"]
    im = case.implied
    x = np.arange(PILLARS.size)
    sd = np.sqrt(np.diag(im.cov) * det.variance_scale)
    fig, ax = plt.subplots(figsize=(6.6, 2.6))
    ax.axhline(0, color=AXIS, lw=0.8)
    ax.fill_between(x, (d.smooth_fit - 2 * sd) * 1e4, (d.smooth_fit + 2 * sd) * 1e4, color=BLUE, alpha=0.10, lw=0,
                    label=t["band_null"])
    ax.plot(x, d.smooth_fit * 1e4, color=BLUE, label=t["smooth"])
    ax.plot(x, (d.smooth_fit + d.anomaly_fit) * 1e4, color=ORANGE, label=t["with_tmpl"])
    ax.plot(x, im.delta_carry * 1e4, linestyle="none", label=t["obs"], **_mk(INK))
    vals = np.concatenate([im.delta_carry, d.smooth_fit + d.anomaly_fit]) * 1e4
    span = vals.max() - vals.min()
    ax.set_ylim(vals.min() - 0.4 * span, vals.max() + 0.4 * span)
    ax.set_xticks(x)
    ax.set_xticklabels(tenors(lang))
    ax.set_ylabel(t["carry_pb"])
    ax.set_title(t["decomp_title"], loc="left")
    ax.legend(loc="lower left", ncol=2)
    _axis_fmt(ax, lang, "y")
    fig.tight_layout()
    return _save(fig, path)


# ------------------------------------------------------------ section 3 and 4
# Dividend views of the running example category: annual dividend of 4.2 going ex in early May, each contributor off
# by a relative error of standard deviation 0.01 + 0.03 t on the dividend going ex at t, 0.7 of it common to all of
# that contributor's dividends.
GAP_EX = 0.55 + np.arange(5)
GAP_DIV, GAP_RATE, GAP_RHO, GAP_DISP = 4.2, 0.03, 0.7, (0.01, 0.03)


def gap_dividends(T):
    """Carry-discounted cumulative dividend ``D(T)`` and the contributor dispersion ``sigma_D(T)`` of the view model."""
    T = np.atleast_1d(np.asarray(T, dtype=float))
    d = GAP_DIV * np.exp(-GAP_RATE * GAP_EX)
    s = d * (GAP_DISP[0] + GAP_DISP[1] * GAP_EX)
    m = GAP_EX[None, :] <= T[:, None]
    D = (m * d).sum(axis=1)
    var = GAP_RHO**2 * (m * s).sum(axis=1) ** 2 + (1 - GAP_RHO**2) * (m * s**2).sum(axis=1)
    return D, np.sqrt(var)


def running_panels(T):
    """Option and dividend panel sizes of the running example category at maturity ``T`` (step functions)."""
    from repo_control.synthetic import CATEGORIES, SCHEDULE

    spec = CATEGORIES[RUNNING_CATEGORY]
    T = np.atleast_1d(np.asarray(T, dtype=float))
    k = np.clip(np.searchsorted(PILLARS, T - 1e-9), 0, PILLARS.size - 1)
    p = np.clip(SCHEDULE.index(T), 0, len(spec.n_div) - 1)
    return np.asarray(spec.n_opt, dtype=float)[k], np.asarray(spec.n_div, dtype=float)[p]


def fig_panel_gap(path: Path, lang: str) -> Path:
    t = TXT[lang]
    S = 100.0
    T = np.linspace(1 / 12, 5, 600)
    D, sd = gap_dividends(T)
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.5))
    ax = axes[0]
    nA_run, nB_run = running_panels(T)
    for i, (nA, nB) in enumerate([(nA_run, nB_run), (25, 20), (8, 4)]):
        ax.plot(T, panel_gap_sd(T, S, D, sd, nA, nB) * 1e4, color=SERIES[i], label=t["cfg"][i], lw=1.4)
    ax.set_xlabel(t["mat"])
    ax.set_ylabel(t["gap_y"])
    ax.set_title(t["gap_title"], loc="left", fontsize=8.5)
    ax.set_ylim(0, 1.45 * ax.get_ylim()[1])
    ax.legend(loc="upper left", fontsize=7.2)
    _axis_fmt(ax, lang)
    ax = axes[1]
    for i, eps in enumerate([0.0005, 0.001, 0.003]):
        s = f"{eps * 100:.2f}".replace(".", "{,}") if lang == "fr" else f"{eps * 100:.2f}"
        ax.plot(T, eps / T * 1e4, color=SERIES[i], label=t["spot_lab"].replace("{x}", s))
    ax.set_yscale("log")
    ax.set_xlabel(t["mat"])
    ax.set_ylabel(t["spot_y"])
    ax.set_title(t["spot_title"], loc="left", fontsize=8.5)
    ax.legend(loc="upper right")
    _axis_fmt(ax, lang, "x")
    fig.tight_layout(w_pad=2.0)
    return _save(fig, path)


def fig_cancellation(path: Path, lang: str) -> Path:
    t = TXT[lang]
    m, books = ne.demo_market()
    T = PILLARS
    F_tot = m.forward(T) * (1 - 0.002)
    repo_fwd = bootstrap_repo(m.spot, m.rates, m.dividends, T, F_tot)
    J = implied_repo_jacobian(m.spot, m.rates, m.schedule, m.div_levels, m.profile, T, F_tot)
    bumps = np.linspace(-0.6, 0.6, 25)
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.6), sharey=True)
    keys = ["A: options and forwards", "B: A + short 2028 dividend swap"]
    for ax, key, title in zip(axes, keys, [t["book_a"], t["book_b"]]):
        book = books[key]
        sr, sd = sensitivities(book, m)
        rows = []
        for bump in bumps:
            lv = m.div_levels.copy()
            lv[1:3] += bump
            mt = m.with_levels(lv)
            rt = bootstrap_repo(m.spot, m.rates, mt.dividends, T, F_tot)
            cl = control_lines(sr, sd, rt.zeros - m.repo.zeros, repo_fwd.zeros - m.repo.zeros, lv - m.div_levels, J)
            rows.append((cl.div_line, cl.repo_line, cl.total, book_value(book, mt.with_repo(rt)) - book_value(book, m)))
        rows = np.array(rows)
        ax.axhline(0, color=AXIS, lw=0.8)
        ax.plot(bumps, rows[:, 0], color=ORANGE, label=t["div_line"])
        ax.plot(bumps, rows[:, 1], color=GREEN, label=t["repo_line"])
        ax.plot(bumps, rows[:, 2], color=BLUE, lw=2.2, label=t["total"])
        ax.plot(bumps[::4], rows[::4, 3], linestyle="none", label=t["full"], **_mk(BLUE))
        ax.set_title(title, loc="left", fontsize=8.5)
        _axis_fmt(ax, lang)
    axes[0].set_ylabel(t["cancel_y"])
    fig.supxlabel(t["cancel_x"], fontsize=8.5, color=INK2, y=0.02)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, bbox_to_anchor=(0.5, 1.07))
    fig.tight_layout(w_pad=1.2, rect=(0, 0.05, 1, 1))
    return _save(fig, path)


# ------------------------------------------------------------ benchmark
def theoretical_power(lam, c):
    lam = np.asarray(lam, dtype=float)
    return norm.cdf(lam - c) + norm.cdf(-lam - c)


def fig_power(res: BenchmarkResult, path: Path, lang: str, z_crit: float) -> Path:
    t = TXT[lang]
    edges = (0, 1, 2, 3, 4, 6, 8, 12, np.inf)
    labels = ["0--1", "1--2", "2--3", "3--4", "4--6", "6--8", "8--12", "12+"]
    mids = np.array([0.5, 1.5, 2.5, 3.5, 5.0, 7.0, 10.0, 14.0])
    fig, ax = plt.subplots(figsize=(6.6, 2.7))
    x = np.arange(len(labels))
    ax.plot(x, theoretical_power(mids, z_crit), color=MUTED, lw=1.2, label=t["theory"])
    for i, d in enumerate(DETECTORS):
        pc = res.power_curve(d, edges)
        ax.plot(x, [p[2] for p in pc], label=t["detectors"][d], lw=2.2 if d == "structural" else 1.3, **_mk(SERIES[i]))
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_xlabel(t["snr_x"])
    ax.set_ylabel(t["det_rate"])
    ax.set_ylim(-0.02, 1.02)
    ax.legend(loc="upper left", fontsize=7.2)
    _axis_fmt(ax, lang, "y")
    fig.tight_layout()
    return _save(fig, path)


def fig_attribution(res: BenchmarkResult, path: Path, lang: str) -> Path:
    t = TXT[lang]
    scen = list(EXPECTED_KIND) + ["fo_repo_mismark", "div_gross_net"]
    kinds = ["level", "step", "spike", "season", "roughness", "none"]
    M = np.zeros((len(scen), len(kinds)))
    for i, s in enumerate(scen):
        recs = [r for r in res.records if r["scenario"] == s]
        for r in recs:
            M[i, kinds.index(r["structural"]["top_kind"])] += 1
        M[i] /= max(len(recs), 1)
    fig, ax = plt.subplots(figsize=(6.6, 2.9))
    ax.imshow(M, cmap=LinearSegmentedColormap.from_list("b", BLUE_RAMP), vmin=0, vmax=1, aspect="auto")
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            if M[i, j] >= 0.01:
                s = f"{100 * M[i, j]:.0f}"
                ax.text(j, i, (s + r"\,\%") if lang == "fr" else (s + r"\%"), ha="center", va="center", fontsize=7.5,
                        color="white" if M[i, j] > 0.55 else INK)
    ax.set_xticks(range(len(kinds)))
    ax.set_xticklabels(t["kinds"])
    ax.set_yticks(range(len(scen)))
    ax.set_yticklabels([t["scenarios"][s] for s in scen])
    ax.grid(False)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.tick_params(length=0)
    ax.set_title(t["attr_title"], loc="left")
    fig.tight_layout()
    return _save(fig, path)


def build_all(res: BenchmarkResult, outdir: Path, lang: str, z_crit: float) -> None:
    setup(lang)
    outdir.mkdir(parents=True, exist_ok=True)
    fig_clean(res, outdir / "clean.pdf", lang)
    fig_gallery(res, outdir / "gallery.pdf", lang)
    fig_decomposition(res, outdir / "decomposition.pdf", lang)
    fig_panel_gap(outdir / "panel_gap.pdf", lang)
    fig_cancellation(outdir / "cancellation.pdf", lang)
    fig_power(res, outdir / "power.pdf", lang, z_crit)
    fig_attribution(res, outdir / "attribution.pdf", lang)
