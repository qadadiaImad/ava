"""Build the LaTeX theory notes (English and French).

Generates figures, value macros and tables from the library and the saved benchmark, then compiles
docs/latex/main_en.tex and docs/latex/main_fr.tex and copies the PDFs to docs/.

Usage:  PYTHONPATH=. python scripts/build_latex.py [--benchmark results/benchmark.json.gz] [--no-compile]
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import warnings
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import latex_assets as la  # noqa: E402
import report_analysis as ra  # noqa: E402

from repo_control.benchmark import DETECTORS, BenchmarkResult  # noqa: E402
from repo_control.synthetic import CATEGORIES, EXPECTED_KIND, SCENARIOS  # noqa: E402

warnings.filterwarnings("ignore", category=RuntimeWarning)

ROOT = Path(__file__).resolve().parents[1]
TEX = ROOT / "docs" / "latex"
OUTPUTS = {"en": ROOT / "docs" / "totem_repo_theory_en.pdf", "fr": ROOT / "docs" / "totem_repo_theorie_fr.pdf"}

UNICODE = {
    "−": "$-$", "Δ": "$\\Delta$", "±": "$\\pm$", "→": "$\\to$", "×": "$\\times$", "≈": "$\\approx$",
    "σ": "$\\sigma$", "κ": "$\\kappa$", "β": "$\\beta$", "ε": "$\\varepsilon$", "|": "$|$", "≤": "$\\le$",
}


def esc(s: str) -> str:
    """Escape plain text for a LaTeX table cell."""
    out = []
    for ch in str(s):
        if ch in UNICODE:
            out.append(UNICODE[ch])
        elif ch in "%&#_":
            out.append("\\" + ch)
        else:
            out.append(ch)
    return "".join(out).replace("$$", "")


def num(x, d=2) -> str:
    return f"\\num{{{x:.{d}f}}}"


def pct(x, d=0) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "--"
    return f"\\pct{{{100 * x:.{d}f}}}"


def macro(name: str, value: str) -> str:
    return f"\\newcommand{{\\{name}}}{{{value}}}\n"


def table(cols: str, header: list, rows: list, size: str = "\\small") -> str:
    lines = [f"{{{size}\\begin{{tabular}}{{{cols}}}", "\\toprule", " & ".join(header) + " \\\\", "\\midrule"]
    for r in rows:
        if r == "MID":
            lines.append("\\midrule")
        else:
            lines.append(" & ".join(r) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}}"]
    return "\n".join(lines) + "\n"


# ------------------------------------------------------------------ build
def generate(res: BenchmarkResult) -> None:
    t_en, t_fr = la.TXT["en"], la.TXT["fr"]
    cases = la.gallery_cases(res)
    z_crit = la.typical_z_crit(res)

    for lang in ("en", "fr"):
        fdir = TEX / "figures" / lang
        fdir.mkdir(parents=True, exist_ok=True)
        la.fig_panel_gap(fdir / "panel_gap.pdf", lang)
        la.fig_cancellation(fdir / "cancellation.pdf", lang)
        la.fig_detection(res, fdir / "detection.pdf", lang)
        la.fig_power(res, fdir / "power.pdf", lang, z_crit)
        la.fig_attribution(res, fdir / "attribution.pdf", lang)
        la.fig_gallery(res, fdir / "gallery.pdf", lang, cases)
        la.fig_decomposition(res, fdir / "decomposition.pdf", lang, cases)

    gap = ra.panel_gap_table()
    cancel = ra.cancellation_table()
    books = ra.book_lines()
    jumps = ra.jump_table()
    boot = la.example_bootstrap()
    par = la.example_parity()
    toy = la.example_toy_glrt()
    lev = la.example_leverage()
    anomaly = list(EXPECTED_KIND)
    mean_det = {d: float(np.mean([res.detection_rate(s, d) for s in anomaly])) for d in DETECTORS}
    clean_fpr = {d: res.detection_rate("clean", d) for d in DETECTORS}
    flagged = [r for r in res.records if r["scenario"] in EXPECTED_KIND and r["flags"]["structural"]]
    attr_pooled = float(np.mean([r["attribution_ok"] for r in flagged]))
    power = {d: {lo: rate for lo, hi, rate, n in res.power_curve(d)} for d in DETECTORS}
    det = res.detection_rate
    mat = {r[0][0]: r for r in books if r[1] == "materiality"}

    # ------------------------------------------------------- shared values
    v = ""
    v += macro("vNTest", f"\\num{{{len(res.records)}}}")
    v += macro("vNTrain", f"\\num{{{res.n_train}}}")
    v += macro("vNTestPerCat", f"\\num{{{res.n_test}}}")
    v += macro("vNCat", str(len(CATEGORIES)))
    v += macro("vNScen", str(len(SCENARIOS)))
    v += macro("vFPR", pct(res.fpr_target))
    v += macro("vFPRmin", pct(min(clean_fpr.values())))
    v += macro("vFPRmax", pct(max(clean_fpr.values())))
    for d, key in zip(DETECTORS, ["Naive", "Shape", "GmmCurve", "GmmDiff", "Struct"]):
        v += macro(f"vMean{key}", pct(mean_det[d]))
        v += macro(f"vFwd{key}", pct(det("fwd_outlier", d)))
        v += macro(f"vFo{key}", pct(det("fo_repo_mismark", d)))
        v += macro(f"vPanel{key}", pct(det("panel_shift", d)))
        v += macro(f"vBucket{key}", pct(det("div_bucket", d)))
        v += macro(f"vTiming{key}", pct(det("div_timing", d)))
        v += macro(f"vSpot{key}", pct(det("spot_sync", d)))
        v += macro(f"vSeason{key}", pct(det("season_profile", d)))
        v += macro(f"vGross{key}", pct(det("div_gross_net", d)))
        for lo, nm in [(0, "A"), (1, "B"), (2, "C"), (3, "D"), (4, "E"), (6, "F"), (8, "G"), (12, "H")]:
            v += macro(f"vPow{key}{nm}", pct(power[d][lo]))
    v += macro("vAttr", pct(attr_pooled))
    v += macro("vFoMismark", pct(res.fo_mismark_rate("fo_repo_mismark")))
    v += macro("vFoClean", pct(res.fo_mismark_rate("clean")))
    for c, key in zip(CATEGORIES, ["EuStock", "UsStock", "Htb", "Index"]):
        v += macro(f"vFoMismark{key}", pct(res.fo_mismark_rate("fo_repo_mismark", c)))
        v += macro(f"vKappa{key}", num(res.variance_scale[c]))
        v += macro(f"vLevelScale{key}", num(res.level_scale.get(c, 1.0)))
    v += macro("vZcrit", num(z_crit))
    v += macro("vPowThFour", pct(float(la.theoretical_power(4.0, z_crit)), 1))
    v += macro("vPowThSix", pct(float(la.theoretical_power(6.0, z_crit)), 1))
    v += macro("vPowThThree", pct(float(la.theoretical_power(3.5, z_crit))))
    v += macro("vNTemplates", str(la.template_count()))
    v += macro("vGapOneY", num(gap[2][5], 1))
    v += macro("vGapFiveY", num(gap[-1][5], 1))
    v += macro("vBookATotal", num(books[1][4]))
    v += macro("vMatA", pct(mat["A"][4], 1))
    v += macro("vMatB", pct(mat["B"][4], 1))
    v += macro("vMatC", pct(mat["C"][4], 1))
    v += macro("vParF", num(par["F"], 2))
    v += macro("vParDF", num(par["DF"], 4))
    v += macro("vParFtwo", num(par["F2"], 2))
    v += macro("vParDFtwo", num(par["DF2"], 4))
    v += macro("vToyC", num(toy["c"]))
    v += macro("vToyCsidak", num(toy["c_sidak"]))
    v += macro("vToyQ", num(toy["q"], 1))
    v += macro("vToyBeta", num(toy["beta_hat"] * 1e4, 1))
    v += macro("vToyM", str(toy["m"]))
    v += macro("vLevMin", num(lev[1:-1].min(), 2))
    v += macro("vLevMax", num(lev[1:-1].max(), 2))
    v += macro("vLevMean", num(lev.mean(), 3))
    v += macro("vNewtonA", f"\\num{{{abs(boot['steps'][0][1]):.1e}}}")
    v += macro("vNewtonB", f"\\num{{{abs(boot['steps'][1][1]):.1e}}}")
    for j, (a, b, c, _) in enumerate(jumps):
        v += macro(f"vJump{'ABCD'[j]}", num(c, 2))
    v += macro("vRuntime", f"\\num{{{res.runtime_s / 60:.0f}}}")

    for lang in ("en", "fr"):
        t = la.TXT[lang]
        gdir = TEX / "generated" / lang
        gdir.mkdir(parents=True, exist_ok=True)
        (gdir / "values.tex").write_text(v)
        T = {}

        # panel-gap table
        hdr = {"en": ["Tenor", "$D(T)$", "$\\sigma_D(T)$", "$n_A$", "$n_B$", "Formula (bp)", "Simulation (bp)"],
               "fr": ["Maturité", "$D(T)$", "$\\sigma_D(T)$", "$n_A$", "$n_B$", "Formule (pb)", "Simulation (pb)"]}[lang]
        rows = [[r[0].replace("Y", "A") if lang == "fr" else r[0], num(r[1], 1), num(r[2], 2), str(r[3]), str(r[4]),
                 num(r[5], 1), num(r[6], 1)] for r in gap]
        T["gap"] = table("lrrrrrr", hdr, rows)

        # cancellation (1Y call)
        hdr = {"en": ["Totem dividend", "Implied repo (bp)", "Div line", "Repo line", "Total", "Full revaluation"],
               "fr": ["Dividende Totem", "Repo implicite (pb)", "Ligne div.", "Ligne repo", "Total", "Réévaluation"]}[lang]
        rows = [[num(r[0], 1), num(r[1], 0), num(r[2], 3), num(r[3], 3), num(r[4], 3), num(r[5], 3)] for r in cancel]
        T["cancel"] = table("rrrrrr", hdr, rows)

        # book lines
        hdr = {"en": ["Book", "Panel shift", "Div line", "Repo line", "Total", "Fwd line", "Non-fwd line", "Full reval."],
               "fr": ["Port.", "Choc panel", "Ligne div.", "Ligne repo", "Total", "Ligne fwd", "Ligne non-fwd", "Rééval."]}[lang]
        rows = []
        for r in books:
            if r[1] == "materiality":
                continue
            rows.append([r[0][0], num(r[1], 1), num(r[2]), num(r[3]), num(r[4]), num(r[5]), num(r[6]), num(r[7])])
        T["books"] = table("lrrrrrrr", hdr, rows)

        # materiality
        hdr = {"en": ["Book", "sd of div line", "sd of total (split risk)", "Share cancelled"],
               "fr": ["Portefeuille", "Écart-type ligne div.", "Écart-type du total", "Part compensée"]}[lang]
        rows = [[k, num(mat[k][2], 3), num(mat[k][3], 3), pct(mat[k][4], 1)] for k in "ABC"]
        T["materiality"] = table("lrrr", hdr, rows)

        # bootstrap example
        hdr = {"en": ["Pillar $T_k$", "Forward $F_k$", "True $b_k$ (bp)", "Implied $b_k$ (bp)"],
               "fr": ["Pilier $T_k$", "Forward $F_k$", "$b_k$ vrai (pb)", "$b_k$ implicite (pb)"]}[lang]
        rows = [[num(boot["T"][k], 2), num(boot["F"][k], 4), num(boot["true"][k] * 1e4, 2), num(boot["implied"][k] * 1e4, 2)]
                for k in range(3)]
        T["boot"] = table("rrrr", hdr, rows)

        # toy GLRT
        names = {"level": ("Level", "Niveau"), "step@1": ("Step from $T_2$", "Marche dès $T_2$"),
                 "step@2": ("Step from $T_3$", "Marche dès $T_3$"), "step@3": ("Step at $T_4$", "Marche en $T_4$"),
                 "spike@0": ("Spike at $T_1$", "Pic en $T_1$"), "spike@1": ("Spike at $T_2$", "Pic en $T_2$"),
                 "spike@2": ("Spike at $T_3$", "Pic en $T_3$")}
        hdr = {"en": ["Template $t$", "$z_t$", "$\\hat a_t$ (bp)", "s.e. (bp)"],
               "fr": ["Gabarit $t$", "$z_t$", "$\\hat a_t$ (pb)", "é.-t. (pb)"]}[lang]
        rows = [[names[n][0 if lang == "en" else 1], num(z, 2), num(a * 1e4, 2), num(s * 1e4, 2)]
                for n, z, a, s in zip(toy["names"], toy["z"], toy["amp"], toy["se"])]
        T["toy"] = table("lrrr", hdr, rows)

        # leverage
        tl = la.tenors(lang)
        hdr = [{"en": "Pillar", "fr": "Pilier"}[lang]] + tl
        rows = [["$h_{kk}$"] + [num(x, 2) for x in lev], [{"en": "$1-h_{kk}$", "fr": "$1-h_{kk}$"}[lang]] + [num(1 - x, 2) for x in lev]]
        T["leverage"] = table("l" + "r" * len(tl), hdr, rows, size="\\footnotesize\\setlength{\\tabcolsep}{4pt}")

        # detection table (short detector names in headers; the caption explains them)
        short = {"en": ["\\makecell{Naive\\\\gap}", "\\makecell{Naive\\\\shape}", "\\makecell{GMM\\\\curve}",
                        "\\makecell{GMM\\\\diff.}", "\\makecell{Struc-\\\\tural}"],
                 "fr": ["\\makecell{Écart\\\\naïf}", "\\makecell{Forme\\\\naïve}", "\\makecell{GMM\\\\courbe}",
                        "\\makecell{GMM\\\\écart}", "\\makecell{Struc-\\\\turel}"]}[lang]
        hdr = [{"en": "Scenario", "fr": "Scénario"}[lang]] + short
        rows = []
        for s in SCENARIOS:
            rows.append([esc(t["scenarios"][s])] + [pct(det(s, d)) for d in DETECTORS])
        rows.append("MID")
        rows.append([{"en": "\\textbf{Mean over the six Totem anomalies}", "fr": "\\textbf{Moyenne sur les six anomalies Totem}"}[lang]]
                    + [f"\\textbf{{{pct(mean_det[d])}}}" for d in DETECTORS])
        first = "p{5.6cm}" if lang == "en" else "p{4.9cm}"
        T["detection"] = table(first + "rrrrr", hdr, rows, size="\\footnotesize")

        # AUC table
        rows = [[esc(t["scenarios"][s])] + [num(res.auc(s, d), 2) for d in DETECTORS] for s in SCENARIOS if s != "clean"]
        T["auc"] = table(first + "rrrrr", hdr, rows, size="\\footnotesize")

        # attribution table
        hdr2 = {"en": ["Scenario", "Flagged", "\\makecell{Right mechanism\\\\(of flagged)}", "\\makecell{FO-mismark\\\\flag}"],
                "fr": ["Scénario", "Signalés", "\\makecell{Bon mécanisme\\\\(parmi signalés)}", "\\makecell{Alerte\\\\marquage FO}"]}[lang]
        rows = []
        for r in ra.attribution_table(res):
            rows.append([esc(t["scenarios"][r["scenario"]]), pct(r["detected"]), pct(r["attr_cond"]), pct(r["fo_flag"])])
        T["attribution"] = table("p{6.2cm}rrr", hdr2, rows, size="\\footnotesize")

        # FPR by category + kappa
        hdr3 = [{"en": "Category", "fr": "Catégorie"}[lang]] + short + ["$\\kappa$"]
        rows = [[esc(t["categories"][c])] + [pct(res.detection_rate("clean", d, c), 1) for d in DETECTORS] + [num(res.variance_scale[c])]
                for c in CATEGORIES]
        T["fpr"] = table("lrrrrrr", hdr3, rows, size="\\footnotesize")

        # structural detection by category
        hdr4 = [{"en": "Scenario", "fr": "Scénario"}[lang]] + [esc(t["categories"][c]) for c in CATEGORIES]
        rows = [[esc(t["scenarios"][s])] + [pct(det(s, "structural", c)) for c in CATEGORIES] for s in SCENARIOS]
        T["bycat"] = table("p{5.6cm}rrrr", hdr4, rows, size="\\footnotesize")

        # gallery explanations
        dobj, items = cases
        hdr5 = {"en": ["Panel", "First explanation of the structural detector"],
                "fr": ["Panneau", "Première explication du détecteur structurel"]}[lang]
        rows = []
        for n, (scen, case, d) in enumerate(items):
            im = case.implied
            if d.explanations:
                txt = la.explanation_text(d.explanations[0], im.pillars, im.context(), dobj, lang)
            else:
                txt = la.fo_mismark_text(d, lang)
            rows.append([f"({'abcdef'[n]})", esc(txt)])
        T["gallery"] = table("lp{13.2cm}", hdr5, rows, size="\\footnotesize")

        # categories of the synthetic market
        cal = {"en": {"eu_annual": "Annual, Apr--May (35\\% with a Nov.\\ interim)", "us_quarterly": "Quarterly",
                      "mixed": "Annual or quarterly", "index_eu": "Monthly stream, 60\\% in Apr--Jun"},
               "fr": {"eu_annual": "Annuel, avr.--mai (35\\,\\% avec acompte en nov.)", "us_quarterly": "Trimestriel",
                      "mixed": "Annuel ou trimestriel", "index_eu": "Flux mensuel, 60\\,\\% en avr.--juin"}}[lang]
        hdr6 = {"en": ["Category", "Dividends", "Yield", "Repo (bp), short\\,$\\to$\\,long", "Option panel", "Div.\\ panel"],
                "fr": ["Catégorie", "Dividendes", "Rendement", "Repo (pb), court $\\to$ long", "Panel options", "Panel div."]}[lang]
        rows = []
        for c, spec in CATEGORIES.items():
            rows.append([esc(t["categories"][c]), cal[spec.calendar],
                         f"{pct(spec.div_yield[0], 1)}--{pct(spec.div_yield[1], 1)}",
                         f"$[{spec.repo_short_bp[0]},{spec.repo_short_bp[1]}]\\to[{spec.repo_long_bp[0]},{spec.repo_long_bp[1]}]$",
                         f"${spec.n_opt[0]}\\to{spec.n_opt[-1]}$", f"${spec.n_div[0]}\\to{spec.n_div[-1]}$"])
        T["categories"] = table("lp{3.1cm}lccc" if lang == "en" else "lp{2.9cm}lccc", hdr6, rows,
                                size="\\footnotesize\\setlength{\\tabcolsep}{4pt}")

        for k, body in T.items():
            (gdir / f"tab_{k}.tex").write_text(body)


def compile_all() -> None:
    for lang, out in OUTPUTS.items():
        main = f"main_{lang}.tex"
        cmd = ["latexmk", "-pdf", "-interaction=nonstopmode", "-halt-on-error", "-outdir=build", main]
        proc = subprocess.run(cmd, cwd=TEX, capture_output=True, text=True, encoding="utf-8", errors="replace")
        if proc.returncode != 0:
            print(proc.stdout[-4000:])
            raise SystemExit(f"latexmk failed for {main}")
        shutil.copy(TEX / "build" / f"main_{lang}.pdf", out)
        print(f"wrote {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--benchmark", default=str(ROOT / "results" / "benchmark.json.gz"))
    ap.add_argument("--no-compile", action="store_true")
    args = ap.parse_args()
    generate(BenchmarkResult.from_json(args.benchmark))
    if not args.no_compile:
        compile_all()
