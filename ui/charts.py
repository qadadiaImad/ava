"""Plotly figure builders — vibrant, presentation-grade visuals."""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go

PALETTE = [
    "#FF5C8A", "#3DDC97", "#FFC857", "#5BC0EB", "#B388EB",
    "#FF8C42", "#00C2D1", "#F45B69", "#7AE582", "#F9F871",
    "#E84A8A", "#43BCCD", "#F86624", "#9BF6FF", "#CDB4DB",
    "#80ED99", "#FFADAD", "#48BFE3", "#FFD6A5", "#A0C4FF",
]

def _layout(top: int = 60) -> dict:
    return {**_LAYOUT, "margin": dict(l=40, r=20, t=top, b=40)}


_LAYOUT = dict(
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(13,17,38,0.55)",
    font=dict(family="Inter, Segoe UI, sans-serif", size=13, color="#EDF0FF"),
    margin=dict(l=40, r=20, t=60, b=40),
)


def _fmt(v: float) -> str:
    a = abs(v)
    if a >= 1e6:
        return f"{v / 1e6:.1f}M"
    if a >= 1e3:
        return f"{v / 1e3:.0f}k"
    return f"{v:.0f}"


def heatmap(
    z: np.ndarray,
    x_labels: list,
    y_labels: list,
    title: str,
    colorscale: str = "Plasma",
    diverging: bool = False,
    colorbar_title: str = "",
    height: int = 420,
) -> go.Figure:
    z = np.asarray(z, dtype=float)
    kwargs = {}
    if diverging:
        lim = np.abs(z).max() or 1.0
        kwargs = dict(zmin=-lim, zmax=lim, colorscale="RdBu_r")
    else:
        kwargs = dict(colorscale=colorscale)
    fig = go.Figure(
        go.Heatmap(
            z=z,
            x=[str(x) for x in x_labels],
            y=[str(y) for y in y_labels],
            colorbar=dict(title=colorbar_title, thickness=14),
            hovertemplate="strike %{x} · tenor %{y}<br>value %{z:,.2f}<extra></extra>",
            **kwargs,
        )
    )
    fig.update_layout(
        title=dict(y=0.98, yanchor="top", text=title, font=dict(size=16)),
        yaxis=dict(autorange="reversed"),
        height=height,
        **_LAYOUT,
    )
    return fig


def partition_figure(
    labels: np.ndarray,
    vega: np.ndarray,
    tenors: list[str],
    strikes: list[float],
    title: str = "Netting sets on the tenor × strike grid",
    set_stats: list | None = None,
    height: int = 520,
) -> go.Figure:
    labels = np.asarray(labels, dtype=int)
    M, K = labels.shape
    n_sets = labels.max() + 1
    colors = [PALETTE[i % len(PALETTE)] for i in range(n_sets)]
    # discrete colorscale
    if n_sets == 1:
        colorscale = [[0.0, colors[0]], [1.0, colors[0]]]
    else:
        colorscale = []
        for i, c in enumerate(colors):
            colorscale += [[i / n_sets, c], [(i + 1) / n_sets, c]]
    hover = np.empty((M, K), dtype=object)
    for m in range(M):
        for k in range(K):
            sid = labels[m, k]
            extra = ""
            if set_stats is not None:
                st_ = set_stats[sid]
                extra = (
                    f"<br>net vega {_fmt(st_.net_vega)} · gross {_fmt(st_.gross_vega)}"
                    f"<br>offset {st_.offset_ratio:.0%} · s̃ {st_.s_tilde:.2f}"
                )
            hover[m, k] = (
                f"set {sid} · tenor {tenors[m]} · strike {strikes[k]}"
                f"<br>vega {_fmt(vega[m, k])}{extra}"
            )
    fig = go.Figure(
        go.Heatmap(
            z=labels,
            zmin=-0.5,
            zmax=n_sets - 0.5,
            x=list(range(K)),
            y=list(range(M)),
            colorscale=colorscale,
            showscale=False,
            opacity=0.92,
            customdata=hover,
            hovertemplate="%{customdata}<extra></extra>",
        )
    )
    # cell annotations: signed vega
    for m in range(M):
        for k in range(K):
            fig.add_annotation(
                x=k, y=m, text=_fmt(vega[m, k]), showarrow=False,
                font=dict(size=10, color="#0B1026"),
            )
    # set boundaries
    lines_x, lines_y = [], []
    for m in range(M):
        for k in range(K):
            if k + 1 < K and labels[m, k] != labels[m, k + 1]:
                lines_x += [k + 0.5, k + 0.5, None]
                lines_y += [m - 0.5, m + 0.5, None]
            if m + 1 < M and labels[m, k] != labels[m + 1, k]:
                lines_x += [k - 0.5, k + 0.5, None]
                lines_y += [m + 0.5, m + 0.5, None]
    fig.add_trace(
        go.Scatter(
            x=lines_x, y=lines_y, mode="lines",
            line=dict(color="#0B1026", width=3.5),
            hoverinfo="skip", showlegend=False,
        )
    )
    fig.update_layout(
        title=dict(y=0.98, yanchor="top", text=title, font=dict(size=16)),
        xaxis=dict(
            tickvals=list(range(K)), ticktext=[str(s) for s in strikes],
            title="moneyness K/F", showgrid=False, zeroline=False,
        ),
        yaxis=dict(
            tickvals=list(range(M)), ticktext=tenors, autorange="reversed",
            title="tenor", showgrid=False, zeroline=False,
        ),
        height=height,
        **_LAYOUT,
    )
    return fig


def ava_waterfall(ava_brut: float, ava_netted: float, ava_full: float, height: int = 420) -> go.Figure:
    fig = go.Figure(
        go.Waterfall(
            orientation="v",
            measure=["absolute", "relative", "total"],
            x=["AVA add-up (2)", "netting benefit", "AVA netted (Def. 4)"],
            y=[ava_brut, -(ava_brut - ava_netted), 0.0],
            text=[_fmt(ava_brut), f"−{_fmt(ava_brut - ava_netted)}", _fmt(ava_netted)],
            textposition="outside",
            connector=dict(line=dict(color="#5BC0EB", dash="dot")),
            increasing=dict(marker=dict(color="#FF5C8A")),
            decreasing=dict(marker=dict(color="#3DDC97")),
            totals=dict(marker=dict(color="#FFC857")),
        )
    )
    fig.add_hline(
        y=ava_full, line=dict(color="#B388EB", dash="dash", width=2),
        annotation_text=f"floor (7): full diversification κ√Var(ΔΠ) = {_fmt(ava_full)}",
        annotation_font_color="#B388EB",
    )
    fig.update_layout(
        title=dict(y=0.98, yanchor="top", text="AVA decomposition", font=dict(size=16)),
        yaxis_title="AVA (EUR)", showlegend=False, height=height, **_LAYOUT,
    )
    return fig


def r2_gauge(r2: float, alpha: float, height: int = 320) -> go.Figure:
    ok = r2 >= alpha
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number+delta",
            value=max(r2, 0.0) * 100,
            number=dict(suffix="%", valueformat=".2f", font=dict(size=42)),
            delta=dict(reference=alpha * 100, valueformat=".2f", suffix=" pts vs α"),
            gauge=dict(
                axis=dict(range=[max(0.0, alpha * 100 - 30), 100], ticksuffix="%"),
                bar=dict(color="#3DDC97" if ok else "#FF5C8A", thickness=0.55),
                bgcolor="rgba(255,255,255,0.05)",
                steps=[
                    dict(range=[0, alpha * 100], color="rgba(255,92,138,0.25)"),
                    dict(range=[alpha * 100, 100], color="rgba(61,220,151,0.20)"),
                ],
                threshold=dict(
                    line=dict(color="#FFC857", width=4), thickness=0.8, value=alpha * 100
                ),
            ),
            title=dict(text="variance score R² (Def. 3)", font=dict(size=15)),
        )
    )
    fig.update_layout(height=height, **_LAYOUT)
    return fig


def merge_history_figure(history: list, budget: float, ava_brut: float, height: int = 430) -> go.Figure:
    steps = [0] + [h.step for h in history]
    te2 = [0.0] + [h.te2 for h in history]
    ava = [ava_brut] + [h.ava for h in history]
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=steps, y=te2, name="TE² consumed", fill="tozeroy",
            line=dict(color="#5BC0EB", width=3),
            fillcolor="rgba(91,192,235,0.25)",
        )
    )
    fig.add_hline(
        y=budget, line=dict(color="#FFC857", dash="dash", width=2),
        annotation_text="budget B = (1−α)·Var(ΔΠ)", annotation_font_color="#FFC857",
    )
    fig.add_trace(
        go.Scatter(
            x=steps, y=ava, name="AVA", yaxis="y2",
            line=dict(color="#FF5C8A", width=3),
        )
    )
    fig.update_layout(
        title=dict(y=0.98, yanchor="top", text="Greedy agglomeration path (sec. 7.3)", font=dict(size=16)),
        xaxis_title="merge step",
        yaxis=dict(title="TE² (residual variance)"),
        yaxis2=dict(title="AVA (EUR)", overlaying="y", side="right", showgrid=False),
        legend=dict(orientation="h", y=1.02, yanchor="bottom"),
        height=height,
        **_layout(110),
    )
    return fig


def spectral_figure(diag, height: int = 430) -> go.Figure:
    n = len(diag.loadings2)
    ks = np.arange(0, n + 1)
    resid_share = diag.residual_curve / diag.var_total if diag.var_total > 0 else diag.residual_curve
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=np.arange(1, n + 1), y=diag.loadings2, name="c_ℓ²λ_ℓ — risk map (Prop. 7)",
            marker=dict(color="#B388EB"),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=ks, y=resid_share, name="TE² floor / Var (Théorème 3)",
            yaxis="y2", line=dict(color="#3DDC97", width=3),
        )
    )
    fig.add_hline(
        y=1 - diag.alpha, yref="y2",
        line=dict(color="#FFC857", dash="dash", width=2),
        annotation_text="budget share (1−α)", annotation_font_color="#FFC857",
    )
    fig.add_vline(
        x=diag.k_star, line=dict(color="#FF5C8A", dash="dot", width=2),
        annotation_text=f"K*(α) = {diag.k_star}", annotation_font_color="#FF5C8A",
    )
    fig.update_layout(
        title=dict(y=0.98, yanchor="top", text="Spectral diagnostic — minimal number of netting sets", font=dict(size=16)),
        xaxis=dict(title="factor rank ℓ / number of sets K", range=[0, min(n, 25) + 0.5]),
        yaxis=dict(title="vega-weighted variance load"),
        yaxis2=dict(title="residual share", overlaying="y", side="right",
                    showgrid=False, range=[0, 1.02]),
        legend=dict(orientation="h", y=1.02, yanchor="bottom"),
        height=height,
        **_layout(110),
    )
    return fig


def smile_decomposition_figure(rows: list[dict], alpha: float, height: int = 430) -> go.Figure:
    tenors = [r["tenor"] for r in rows]
    fig = go.Figure()
    fig.add_trace(go.Bar(x=tenors, y=[r["level"] for r in rows], name="level Σνₖ (nettable)",
                         marker_color="#3DDC97"))
    fig.add_trace(go.Bar(x=tenors, y=[r["risk_reversal"] for r in rows], name="risk-reversal Σνₖ(k−k_ATM)",
                         marker_color="#FFC857"))
    fig.add_trace(go.Bar(x=tenors, y=[r["butterfly"] for r in rows], name="butterfly Σνₖ(k−k_ATM)²",
                         marker_color="#FF5C8A"))
    fig.add_trace(
        go.Scatter(
            x=tenors, y=[r["r2_line"] for r in rows], name="per-tranche R² (stage 1)",
            yaxis="y2", mode="lines+markers",
            line=dict(color="#5BC0EB", width=3), marker=dict(size=9),
        )
    )
    fig.add_hline(
        y=alpha, yref="y2", line=dict(color="#5BC0EB", dash="dash", width=1.5),
        annotation_text="α", annotation_font_color="#5BC0EB",
    )
    fig.update_layout(
        title=dict(y=0.98, yanchor="top", text="Smile decomposition per tranche — net level / RR / FLY (sec. 6.2, Th. 4)",
                   font=dict(size=16)),
        barmode="group",
        yaxis=dict(title="exposure (EUR / vol pt units)"),
        yaxis2=dict(title="R² of level-collapse", overlaying="y", side="right",
                    showgrid=False, range=[min(0.0, min(r["r2_line"] for r in rows)) - 0.05, 1.02]),
        legend=dict(orientation="h", y=1.02, yanchor="bottom"),
        height=height,
        **_layout(110),
    )
    return fig


def two_bucket_figure(s_i: float, s_j: float, nu_i: float, var_total: float,
                      alpha: float, nu_j_now: float, rho_now: float, height: int = 430) -> go.Figure:
    """Admissibility threshold rho_min as a function of |nu_j| (eq. 8)."""
    nu_j = np.linspace(1.0, max(abs(nu_j_now) * 2.5, abs(nu_i) * 1.5), 200)
    rho_min = (s_i ** 2 + s_j ** 2) / (2 * s_i * s_j) - (1 - alpha) * var_total / (2 * nu_j ** 2 * s_i * s_j)
    rho_min = np.clip(rho_min, -1.05, 1.05)
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=nu_j, y=rho_min, name="ρ_min(|ν_j|) — eq. (8)",
            line=dict(color="#FFC857", width=3),
            fill="tozeroy", fillcolor="rgba(255,92,138,0.12)",
        )
    )
    fig.add_hline(y=1.0, line=dict(color="rgba(255,255,255,0.3)", width=1))
    fig.add_trace(
        go.Scatter(
            x=[abs(nu_j_now)], y=[rho_now], mode="markers+text",
            name="your pair",
            marker=dict(size=16, symbol="diamond",
                        color="#3DDC97" if rho_now >= np.interp(abs(nu_j_now), nu_j, rho_min) else "#FF5C8A",
                        line=dict(color="white", width=1.5)),
            text=["pair"], textposition="top center",
        )
    )
    fig.update_layout(
        title=dict(y=0.98, yanchor="top", text="Two-bucket admissibility — required correlation vs netted size",
                   font=dict(size=16)),
        xaxis=dict(title="|ν_j| (vega netted onto the pivot)"),
        yaxis=dict(title="correlation ρ", range=[-1.05, 1.1]),
        legend=dict(orientation="h", y=1.02, yanchor="bottom"),
        height=height,
        **_layout(110),
    )
    return fig


def dendrogram_figure(merges: list, epsilon: float, n_applied: int, height: int = 430) -> go.Figure:
    """Run-1 dendrogram profile: merge heights (base-risk epsilon of the
    merged set) by step, with the retained cut."""
    steps = [m.step for m in merges]
    heights = [m.height for m in merges]
    colors = ["#3DDC97" if m.step <= n_applied else "rgba(160,170,210,0.45)" for m in merges]
    fig = go.Figure(
        go.Bar(
            x=steps, y=heights, name="merge height ε_r",
            marker=dict(color=colors),
            hovertemplate="step %{x}<br>ε = %{y:.3f}<extra></extra>",
        )
    )
    fig.add_hline(
        y=epsilon, line=dict(color="#FFC857", dash="dash", width=2),
        annotation_text=f"cut height ε = {epsilon:.2f}", annotation_font_color="#FFC857",
    )
    if 0 < n_applied < len(merges):
        fig.add_vline(
            x=n_applied + 0.5, line=dict(color="#FF5C8A", dash="dot", width=2),
            annotation_text="retained cut", annotation_font_color="#FF5C8A",
        )
    fig.update_layout(
        title=dict(y=0.98, yanchor="top",
                   text="Dendrogram of the test nodes — base risk d_ij, portfolio-free (sec. 7.5)",
                   font=dict(size=16)),
        xaxis_title="merge step (increasing base risk)",
        yaxis_title="set ε = max d(j, pivot)/s_j",
        showlegend=False,
        height=height,
        **_layout(110),
    )
    return fig


def correlation_curve_figure(x_dense: np.ndarray, rho_dense: np.ndarray,
                             x_points: np.ndarray, rho_points: np.ndarray,
                             height: int = 380) -> go.Figure:
    """Prop. 2 of the two-layer note: the correlation-with-ATM curve
    generated by the deformation model — the curvature 'cliff'."""
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=x_dense, y=rho_dense, mode="lines", name="ρ(x) = s₀/sₓ (Prop. 2)",
            line=dict(color="#5BC0EB", width=3),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=x_points, y=rho_points, mode="markers", name="grid points",
            marker=dict(size=11, color="#FFC857", line=dict(color="white", width=1)),
        )
    )
    fig.update_layout(
        title=dict(y=0.98, yanchor="top",
                   text="Generated correlation with the ATM — the model's trace, no book",
                   font=dict(size=16)),
        xaxis_title="moneyness x = K/F − 1",
        yaxis=dict(title="ρ(x)", range=[0, 1.05]),
        legend=dict(orientation="h", y=1.02, yanchor="bottom"),
        height=height,
        **_layout(110),
    )
    return fig
