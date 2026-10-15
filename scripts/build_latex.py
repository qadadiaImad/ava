"""Build the methodology notes (French and English).

Generates the figures, the value macros and the tables from the library and the saved benchmark, then compiles
docs/latex/main_fr.tex and docs/latex/main_en.tex and copies the PDFs to docs/.

Usage:  PYTHONPATH=. python scripts/build_latex.py [--benchmark results/benchmark.json.gz] [--no-compile] [--no-figures]
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
import note_figures as nf  # noqa: E402
import note_examples as ne  # noqa: E402
from latex_assets import NAMES, num, pct, tenor  # noqa: E402

from repo_control.benchmark import DETECTORS, BenchmarkResult  # noqa: E402
from repo_control.synthetic import CATEGORIES, EXPECTED_KIND, PILLARS, SCENARIOS  # noqa: E402

warnings.filterwarnings("ignore", category=RuntimeWarning)

ROOT = Path(__file__).resolve().parents[1]
TEX = ROOT / "docs" / "latex"
OUTPUTS = {"fr": ROOT / "docs" / "totem_repo_note_fr.pdf", "en": ROOT / "docs" / "totem_repo_note_en.pdf"}
DET_KEYS = dict(zip(DETECTORS, ["Naive", "Shape", "GmmCurve", "GmmDiff", "Struct"]))
CAT_KEYS = dict(zip(CATEGORIES, ["EuStock", "UsStock", "Htb", "Index"]))
SCEN_KEYS = {"fwd_outlier": "Fwd", "fo_repo_mismark": "Fo", "panel_shift": "Panel", "div_bucket": "Bucket",
             "div_timing": "Timing", "spot_sync": "Spot", "season_profile": "Season", "div_gross_net": "Gross"}


def macro(name: str, value: str) -> str:
    return f"\\newcommand{{\\{name}}}{{{value}}}\n"


def table(cols: str, header: list, rows: list, size: str = "\\small", env: str = "tabular") -> str:
    width = "{\\linewidth}" if env == "tabularx" else ""
    lines = [f"{{{size}\\begin{{{env}}}{width}{{{cols}}}", "\\toprule", " & ".join(header) + " \\\\", "\\midrule"]
    for r in rows:
        lines.append("\\midrule" if r == "MID" else " & ".join(r) + " \\\\")
    lines += ["\\bottomrule", f"\\end{{{env}}}}}"]
    return "\n".join(lines) + "\n"


# ------------------------------------------------------------------ values
def shared_values(res: BenchmarkResult, z_crit: float) -> str:
    gap = la.example_panel_gap()
    books = ne.book_lines()
    jumps = ne.jump_table()
    boot = la.example_bootstrap()
    par = la.example_parity()
    toy = la.example_toy_glrt()
    lev = la.example_leverage()
    anomaly = list(EXPECTED_KIND)
    det = res.detection_rate
    mean_det = {d: float(np.mean([det(s, d) for s in anomaly])) for d in DETECTORS}
    flagged = [r for r in res.records if r["scenario"] in EXPECTED_KIND and r["flags"]["structural"]]
    power = {d: {lo: rate for lo, hi, rate, n in res.power_curve(d)} for d in DETECTORS}
    mat = {r[0][0]: r for r in books if r[1] == "materiality"}

    v = macro("vNTest", f"\\num{{{len(res.records)}}}")
    v += macro("vNTrain", f"\\num{{{res.n_train}}}")
    v += macro("vNTestPerCat", f"\\num{{{res.n_test}}}")
    v += macro("vNCat", str(len(CATEGORIES)))
    v += macro("vNScen", str(len(SCENARIOS)))
    v += macro("vFPR", pct(res.fpr_target))
    for d, key in DET_KEYS.items():
        v += macro(f"vMean{key}", pct(mean_det[d]))
        v += macro(f"vClean{key}", pct(det("clean", d), 1))
        for s, sk in SCEN_KEYS.items():
            v += macro(f"v{sk}{key}", pct(det(s, d)))
        for lo, nm in [(0, "A"), (1, "B"), (2, "C"), (3, "D"), (4, "E"), (6, "F"), (8, "G"), (12, "H")]:
            v += macro(f"vPow{key}{nm}", pct(power[d][lo]))
    v += macro("vAttr", pct(float(np.mean([r["attribution_ok"] for r in flagged]))))
    v += macro("vFoMismark", pct(res.fo_mismark_rate("fo_repo_mismark")))
    v += macro("vFoClean", pct(res.fo_mismark_rate("clean")))
    for c, key in CAT_KEYS.items():
        v += macro(f"vFoMismark{key}", pct(res.fo_mismark_rate("fo_repo_mismark", c)))
        v += macro(f"vKappa{key}", num(res.variance_scale[c]))
        v += macro(f"vLevelScale{key}", num(res.level_scale.get(c, 1.0)))
    v += macro("vZcrit", num(z_crit))
    v += macro("vPowThFour", pct(float(la.theoretical_power(4.0, z_crit)), 0))
    v += macro("vPowThSix", pct(float(la.theoretical_power(6.0, z_crit)), 1))
    v += macro("vNTemplates", str(la.template_count()))
    v += macro("vGapOneY", num(gap[1][5], 1))
    v += macro("vGapFiveY", num(gap[-1][5], 0))
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
    b = toy["best"]
    v += macro("vToyZbest", num(toy["z"][b]))
    v += macro("vToyAmpBest", num(toy["amp"][b] * 1e4, 1))
    v += macro("vToySeBest", num(toy["se"][b] * 1e4, 1))
    v += macro("vToyZafter", num(toy["max_z_after"], 1))
    v += macro("vToyQafter", num(toy["q_after"], 1))
    v += macro("vLevMin", num(lev[1:-1].min(), 2))
    v += macro("vLevMax", num(lev[1:-1].max(), 2))
    v += macro("vNewtonA", f"\\num{{{abs(boot['steps'][0][1]):.1e}}}")
    v += macro("vNewtonB", f"\\num{{{abs(boot['steps'][1][1]):.1e}}}")
    for j, (a, b_, c, _) in enumerate(jumps):
        v += macro(f"vJump{'ABCD'[j]}", num(c, 2))
    v += macro("vRuntime", f"\\num{{{res.runtime_s / 60:.0f}}}")
    return v


def running_values(run: dict, lang: str) -> str:
    T, k6, k9 = run["pillars"], run["k6"], run["k9"]
    d0 = run["clean"]
    cases = run["cases"]
    v = macro("vRunDiv", num(run["div_mean"], 1))
    v += macro("vRunExDate", la.date_text(run["first_ex"], lang))
    v += macro("vRunNoptShort", str(int(run["n_opt"][0])))
    v += macro("vRunNoptLong", str(int(run["n_opt"][-1])))
    v += macro("vRunNdivShort", str(int(run["n_div"][0])))
    v += macro("vRunNdivLong", str(int(run["n_div"][-1])))
    v += macro("vRunQfoShort", num(run["q_fo"][0] * 1e4, 0))
    v += macro("vRunQfoLong", num(run["q_fo"][-1] * 1e4, 0))
    v += macro("vRunQtotLong", num(run["q_tot"][-1] * 1e4, 0))
    v += macro("vRunQtotSix", num(run["q_tot"][k6] * 1e4, 0))
    v += macro("vRunQtotNine", num(run["q_tot"][k9] * 1e4, 0))
    v += macro("vRunDrop", num((run["q_tot"][k6] - run["q_tot"][k9]) * 1e4, 0))
    v += macro("vRunGapLong", num(abs(run["q_tot"][-1] - run["q_fo"][-1]) * 1e4, 0))
    v += macro("vRunDeltaNine", num(run["delta"][k9] * 1e4, 0))
    v += macro("vRunSdNine", num(run["sd"][k9] * 1e4, 1))
    v += macro("vRunZNine", num(abs(run["delta"][k9] / run["sd"][k9]), 1))
    v += macro("vRunDeltaFive", num(run["delta"][-1] * 1e4, 0))
    v += macro("vRunSdFive", num(run["sd"][-1] * 1e4, 0))
    v += macro("vRunSdRepoShort", num(run["sd_repo"][0] * 1e4, 0))
    v += macro("vRunSdRepoLong", num(run["sd_repo"][-1] * 1e4, 0))
    v += macro("vRunDivGap", num(-run["div_gap_9m"], 2))
    v += macro("vRunDivGapPct", pct(-run["div_gap_9m"] / run["div_mean"], 0))
    v += macro("vRunMaxZ", num(d0.max_z))
    v += macro("vRunCrit", num(d0.z_crit))
    v += macro("vRunChiP", num(d0.q_p))
    v += macro("vRunKappa", num(run["det"].variance_scale))
    # scenario numbers quoted in the text
    c, d = cases["spot_sync"]
    v += macro("vRunSpotEps", pct(c.truth["eps"], 3, True))
    v += macro("vRunSpotEst", pct(-d.explanations[0].amplitude, 3, True))
    c, d = cases["div_bucket"]
    v += macro("vRunBucketDelta", num(c.truth["delta"], 2, True))
    ex = d.explanations[0]
    D9 = c.implied.carry_div[ex.pillar]
    v += macro("vRunBucketEst", num(-ex.amplitude * (c.implied.spot - D9), 2, True))
    v += macro("vRunBucketSe", num(ex.se * (c.implied.spot - D9), 2))
    v += macro("vRunBucketAmp", num(ex.amplitude * 1e4, 0))
    v += macro("vRunBucketZ", num(ex.z, 1))
    c, d = cases["div_timing"]
    v += macro("vRunTimingShare", pct(c.truth["share"]))
    v += macro("vRunTimingRatio", num(c.implied.dispersion_ratio[c.truth["pillar"]], 0))
    ex = d.explanations[0]
    k = ex.pillar
    i = int(np.argmin(np.abs(c.implied.profile.ex_times - T[k])))
    share = abs(ex.amplitude * (c.implied.spot - c.implied.carry_div[k]) / c.implied.profile.amounts[i])
    v += macro("vRunTimingEst", pct(share))
    v += macro("vRunTimingDays", num((c.true_dividends.ex_times[c.truth["div"]] - T[k]) * 365, 0))
    c, d = cases["season_profile"]
    v += macro("vRunSeasonShift", num(c.truth["shift_days"], 0, True))
    v += macro("vRunSeasonEst", num(d.explanations[0].amplitude * 365, 0, True))
    c, d = cases["fwd_outlier"]
    v += macro("vRunFwdShift", num(c.truth["ln_fwd_shift"] * 1e4, 0, True))
    v += macro("vRunFwdEst", num(-d.explanations[0].amplitude * 1e4, 0, True))
    v += macro("vRunFwdSe", num(d.explanations[0].se * 1e4, 0))
    v += macro("vRunFwdTenor", tenor(T[c.truth["pillar"]], lang))
    c, d = cases["panel_shift"]
    v += macro("vRunPanelDelta", pct(c.truth["delta"]))
    v += macro("vRunPanelTenor", tenor(T[c.truth["pillar"]], lang))
    v += macro("vRunPanelMaxZ", num(d.max_z))
    c, d = cases["fo_repo_mismark"]
    v += macro("vRunFoShift", num(c.truth["shift"] * 1e4, 0, True))
    v += macro("vRunFoLevel", num(d.repo_level * 1e4, 1))
    v += macro("vRunFoSe", num(d.repo_level_se * 1e4, 1))
    v += macro("vRunFoCrit", num(d.level_crit))
    v += macro("vRunFoThr", num(d.level_crit * d.repo_level_se * 1e4, 1))
    c, d = cases["div_gross_net"]
    v += macro("vRunGrossBias", pct(c.truth["bias"], 0, True))
    ex = d.explanations[0]
    v += macro("vRunGrossEst", num(-ex.amplitude * (c.implied.spot - c.implied.carry_div[ex.pillar]), 2, True))
    return v


# ------------------------------------------------------------------ tables
def tables(res: BenchmarkResult, run: dict, lang: str) -> dict:
    fr = lang == "fr"
    names = NAMES[lang]
    det = res.detection_rate
    out = {}

    gap = la.example_panel_gap()
    hdr = (["Maturité", "$D(T)$", "$\\sigma_D(T)$", "$n^{\\mathrm{opt}}$", "$n^{\\mathrm{div}}$", "Formule (pb)",
            "Simulation (pb)"] if fr else
           ["Maturity", "$D(T)$", "$\\sigma_D(T)$", "$n^{\\mathrm{opt}}$", "$n^{\\mathrm{div}}$", "Formula (bp)",
            "Simulation (bp)"])
    rows = [[tenor(r[0], lang), num(r[1], 2), num(r[2], 2), str(r[3]), str(r[4]), num(r[5], 1), num(r[6], 1)]
            for r in gap]
    out["gap"] = table("lrrrrrr", hdr, rows)

    cancel = ne.cancellation_table()
    hdr = (["$D^{\\mathrm{tot}}$", "$q^{\\mathrm{tot}}$ (pb)", "Ligne dividendes", "Ligne repo", "Total",
            "Réévaluation"] if fr else
           ["$D^{\\mathrm{tot}}$", "$q^{\\mathrm{tot}}$ (bp)", "Dividend line", "Repo line", "Total",
            "Full revaluation"])
    rows = [[num(r[0], 1), num(r[1], 0), num(r[2], 3), num(r[3], 3), num(r[4], 3), num(r[5], 3)] for r in cancel]
    out["cancel"] = table("rrrrrr", hdr, rows)

    books = ne.book_lines()
    hdr = (["", "$\\Delta\\ell$", "\\makecell{Ligne\\\\dividendes}", "\\makecell{Ligne\\\\repo}", "Total",
            "\\makecell{Ligne\\\\forward}", "\\makecell{Ligne\\\\non-forward}",
            "\\makecell{Rééva-\\\\luation}"] if fr else
           ["", "$\\Delta\\ell$", "\\makecell{Dividend\\\\line}", "\\makecell{Repo\\\\line}", "Total",
            "\\makecell{Forward\\\\line}", "\\makecell{Non-forward\\\\line}", "\\makecell{Full\\\\revaluation}"])
    rows, last = [], None
    for r in books:
        if r[1] == "materiality":
            continue
        if last is not None and r[0] != last:
            rows.append("MID")
        label = r[0][0] if r[0] != last else ""
        last = r[0]
        rows.append([label, num(r[1], 1, True) if r[1] else num(0, 1), num(r[2]), num(r[3]), num(r[4]), num(r[5]),
                     num(r[6]), num(r[7])])
    out["books"] = table("crrrrrrr", hdr, rows)

    mat = {r[0][0]: r for r in books if r[1] == "materiality"}
    hdr = (["Portefeuille", "\\makecell{Écart-type de la\\\\ligne dividendes}", "\\makecell{Écart-type\\\\du total}",
            "Part compensée"] if fr else
           ["Book", "\\makecell{Std.\\ dev.\\ of the\\\\dividend line}", "\\makecell{Std.\\ dev.\\\\of the total}",
            "Share cancelled"])
    rows = [[k, num(mat[k][2], 3), num(mat[k][3], 3), pct(mat[k][4], 1)] for k in "ABC"]
    out["materiality"] = table("lrrr", hdr, rows)

    boot = la.example_bootstrap()
    hdr = (["Pilier $T_k$", "Forward $F_k$", "$q_k$ vrai (pb)", "$q_k$ implicite (pb)"] if fr else
           ["Pillar $T_k$", "Forward $F_k$", "True $q_k$ (bp)", "Implied $q_k$ (bp)"])
    rows = [[num(boot["T"][k], 2), num(boot["F"][k], 4), num(boot["true"][k] * 1e4, 2),
             num(boot["implied"][k] * 1e4, 2)] for k in range(3)]
    out["boot"] = table("rrrr", hdr, rows)

    toy = la.example_toy_glrt()
    tnames = {"level": ("Level", "Niveau"), "step@1": ("Step from $T_2$", "Marche dès $T_2$"),
              "step@2": ("Step from $T_3$", "Marche dès $T_3$"), "step@3": ("Step at $T_4$", "Marche en $T_4$"),
              "spike@0": ("Spike at $T_1$", "Pic en $T_1$"), "spike@1": ("Spike at $T_2$", "Pic en $T_2$"),
              "spike@2": ("Spike at $T_3$", "Pic en $T_3$")}
    hdr = (["Forme type $g$", "$z_g$", "$\\hat a_g$ (pb)", "Écart-type (pb)"] if fr else
           ["Typical shape $g$", "$z_g$", "$\\hat a_g$ (bp)", "Std.\\ error (bp)"])
    rows = [[tnames[n][1 if fr else 0], num(z, 2), num(a * 1e4, 1), num(s * 1e4, 1)]
            for n, z, a, s in zip(toy["names"], toy["z"], toy["amp"], toy["se"])]
    out["toy"] = table("lrrr", hdr, rows)

    lev = la.example_leverage()
    hdr = ["Pilier" if fr else "Pillar"] + [tenor(t, lang) for t in PILLARS]
    rows = [["$h_{kk}$"] + [num(x, 2) for x in lev], ["$1-h_{kk}$"] + [num(1 - x, 2) for x in lev]]
    out["leverage"] = table("l" + "r" * len(PILLARS), hdr, rows, size="\\footnotesize\\setlength{\\tabcolsep}{4pt}")

    short = (["\\makecell{Écart\\\\naïf}", "\\makecell{Forme\\\\naïve}", "\\makecell{Spline +\\\\mélange,\\\\courbe}",
              "\\makecell{Spline +\\\\mélange,\\\\écart}", "\\makecell{Détecteur\\\\structurel}"] if fr else
             ["\\makecell{Naive\\\\gap}", "\\makecell{Naive\\\\shape}", "\\makecell{Spline +\\\\mixture,\\\\curve}",
              "\\makecell{Spline +\\\\mixture,\\\\difference}", "\\makecell{Structural\\\\detector}"])
    hdr = ["Scénario" if fr else "Scenario"] + short
    rows = [[names["scenarios"][s]] + [pct(det(s, d)) for d in DETECTORS] for s in SCENARIOS]
    rows.append("MID")
    rows.append(["\\textbf{Moyenne des six anomalies Totem}" if fr else "\\textbf{Mean of the six Totem anomalies}"]
                + [f"\\textbf{{{pct(float(np.mean([det(s, d) for s in EXPECTED_KIND])))}}}" for d in DETECTORS])
    out["detection"] = table("lrrrrr", hdr, rows, size="\\small")

    rows = [[names["scenarios"][s]] + [num(res.auc(s, d), 2) for d in DETECTORS] for s in SCENARIOS if s != "clean"]
    out["auc"] = table("lrrrrr", hdr, rows, size="\\small")

    hdr2 = (["Scénario", "Signalés", "\\makecell{Bon mécanisme\\\\(parmi les signalés)}",
             "\\makecell{Alerte d'erreur\\\\de marquage FO}"] if fr else
            ["Scenario", "Flagged", "\\makecell{Right mechanism\\\\(among flagged)}", "\\makecell{FO mismark\\\\flag}"])
    rows = [[names["scenarios"][r["scenario"]], pct(r["detected"]),
             pct(r["attr_cond"]) if r["scenario"] in EXPECTED_KIND else "--", pct(r["fo_flag"])]
            for r in ne.attribution_table(res)]
    out["attribution"] = table("lrrr", hdr2, rows, size="\\small")

    hdr3 = ["Catégorie" if fr else "Category"] + short + ["$\\kappa$"]
    rows = [[names["categories"][c]] + [pct(res.detection_rate("clean", d, c), 1) for d in DETECTORS]
            + [num(res.variance_scale[c])] for c in CATEGORIES]
    out["fpr"] = table("lrrrrrr", hdr3, rows, size="\\small")

    hdr4 = ["Scénario" if fr else "Scenario"] + [names["categories"][c] for c in CATEGORIES]
    rows = [[names["scenarios"][s]] + [pct(det(s, "structural", c)) for c in CATEGORIES] for s in SCENARIOS]
    out["bycat"] = table("lrrrr", hdr4, rows, size="\\small")

    cal = ({"eu_annual": "annuel, avr.--mai (\\pct{35} avec acompte en nov.)", "us_quarterly": "trimestriel",
            "mixed": "annuel ou trimestriel", "index_eu": "flux mensuel, \\pct{60} en avr.--juin"} if fr else
           {"eu_annual": "annual, Apr--May (\\pct{35} with a Nov.\\ interim)", "us_quarterly": "quarterly",
            "mixed": "annual or quarterly", "index_eu": "monthly stream, \\pct{60} in Apr--Jun"})
    hdr6 = (["Catégorie", "Dividendes", "Rendement", "Repo (pb), court $\\to$ long", "$n^{\\mathrm{opt}}$",
             "$n^{\\mathrm{div}}$"] if fr else
            ["Category", "Dividends", "Yield", "Repo (bp), short $\\to$ long", "$n^{\\mathrm{opt}}$",
             "$n^{\\mathrm{div}}$"])
    rows = []
    for c, spec in CATEGORIES.items():
        rows.append([names["categories"][c], cal[spec.calendar],
                     f"{pct(spec.div_yield[0], 1)}--{pct(spec.div_yield[1], 1)}",
                     f"$[{spec.repo_short_bp[0]}, {spec.repo_short_bp[1]}]\\to[{spec.repo_long_bp[0]}, "
                     f"{spec.repo_long_bp[1]}]$",
                     f"${spec.n_opt[0]}\\to{spec.n_opt[-1]}$", f"${spec.n_div[0]}\\to{spec.n_div[-1]}$"])
    cols = "@{}>{\\raggedright\\arraybackslash}p{2.3cm}>{\\raggedright\\arraybackslash}p{3.6cm}lccc@{}"
    out["categories"] = table(cols, hdr6, rows, size="\\footnotesize\\setlength{\\tabcolsep}{4pt}")

    hdr7 = (["Situation", "Ce qui est injecté", "Ce que dit le détecteur"] if fr else
            ["Situation", "What is injected", "What the detector says"])
    rows = [[a, b, c] for a, b, c in la.running_rows(run, lang)]
    out["running"] = table(">{\\raggedright\\arraybackslash}p{2.6cm}>{\\raggedright\\arraybackslash}p{4.5cm}"
                           ">{\\raggedright\\arraybackslash}X", hdr7, rows, size="\\footnotesize", env="tabularx")
    return out


# ------------------------------------------------------------------ build
def generate(res: BenchmarkResult, figures: bool = True) -> None:
    z_crit = la.typical_z_crit(res)
    run = la.running_example(res)
    shared = shared_values(res, z_crit)
    for lang in ("fr", "en"):
        if figures:
            nf.build_all(res, TEX / "figures" / lang, lang, z_crit)
        gdir = TEX / "generated" / lang
        gdir.mkdir(parents=True, exist_ok=True)
        (gdir / "values.tex").write_text(shared + running_values(run, lang))
        for k, body in tables(res, run, lang).items():
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
    ap.add_argument("--no-figures", action="store_true")
    args = ap.parse_args()
    generate(BenchmarkResult.from_json(args.benchmark), figures=not args.no_figures)
    if not args.no_compile:
        compile_all()
