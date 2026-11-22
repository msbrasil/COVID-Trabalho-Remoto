"""Padronização visual, helpers editoriais e mapas coropléticos em alta resolução."""

from pathlib import Path

import geobr
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pandas as pd

from src.config import FIGURES_DIR, PALETTE

SOURCE_CAPTION = "Fonte: Microdados da PNAD Covid-19 (IBGE, 2020). Elaboração própria."


def set_portfolio_style() -> None:
    """Configura estilo editorial limpo e consistente para Matplotlib e Seaborn."""
    plt.rcParams.update(
        {
            "figure.facecolor": PALETTE["bg"],
            "axes.facecolor": PALETTE["bg"],
            "axes.edgecolor": PALETTE["grid_gray"],
            "axes.linewidth": 0.9,
            "axes.grid": True,
            "axes.axisbelow": True,
            "grid.color": PALETTE["grid_gray"],
            "grid.linestyle": "--",
            "grid.linewidth": 0.7,
            "grid.alpha": 0.85,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "font.family": "DejaVu Sans",
            "font.size": 10.5,
            "axes.titlesize": 12.5,
            "axes.titleweight": "bold",
            "axes.titlecolor": PALETTE["dark_text"],
            "axes.labelsize": 10.5,
            "axes.labelcolor": PALETTE["dark_text"],
            "xtick.color": PALETTE["muted_text"],
            "ytick.color": PALETTE["muted_text"],
            "xtick.labelsize": 9.5,
            "ytick.labelsize": 9.5,
            "legend.frameon": False,
            "legend.fontsize": 9.5,
            "savefig.dpi": 300,
            "savefig.facecolor": PALETTE["bg"],
            "savefig.bbox": "tight",
        }
    )


def format_ptbr_number(value: float, decimals: int = 1) -> str:
    """Formata número com separador de milhar '.' e decimal ',' (padrão PT-BR)."""
    formatted = f"{value:,.{decimals}f}"
    return formatted.replace(",", "_").replace(".", ",").replace("_", ".")


def add_title_and_source(
    fig: plt.Figure,
    title: str,
    subtitle: str | None = None,
    source: str = SOURCE_CAPTION,
    title_y: float = 0.98,
    subtitle_y: float = 0.935,
    source_y: float = 0.01,
) -> None:
    """Adiciona título principal, subtítulo descritivo e rodapé de fonte à figura."""
    fig.suptitle(
        title,
        x=0.02,
        y=title_y,
        ha="left",
        fontsize=14,
        fontweight="bold",
        color=PALETTE["dark_text"],
    )
    if subtitle:
        fig.text(
            0.02,
            subtitle_y,
            subtitle,
            ha="left",
            fontsize=10.5,
            color=PALETTE["muted_text"],
        )
    fig.text(
        0.02,
        source_y,
        source,
        ha="left",
        fontsize=8.5,
        color=PALETTE["muted_text"],
        style="italic",
    )


def style_percent_axis(ax: plt.Axes, axis: str = "x", decimals: int = 0) -> None:
    """Formata eixo numérico como percentual limpo."""
    fmt = mticker.FuncFormatter(lambda v, _: f"{format_ptbr_number(v, decimals)}%")
    if axis == "x":
        ax.xaxis.set_major_formatter(fmt)
    else:
        ax.yaxis.set_major_formatter(fmt)


def load_br_states_gdf(year: int = 2020):
    """Carrega geometria simplificada das 27 UFs brasileiras via geobr."""
    return geobr.read_state(year=year, simplified=True)


def plot_dual_state_maps(
    uf_df: pd.DataFrame,
    left_col: str,
    right_col: str,
    left_title: str,
    right_title: str,
    left_cbar_label: str,
    right_cbar_label: str,
    suptitle: str,
    subtitle: str,
    left_cmap: str = "YlGnBu",
    right_cmap: str = "OrRd",
    annotate_top_n: int = 5,
) -> plt.Figure:
    """Gera figura lado a lado com dois mapas coropléticos por UF (merge por abbrev_state)."""
    states = load_br_states_gdf(year=2020)
    merged = states.merge(uf_df, left_on="abbrev_state", right_on="uf_sigla", how="left")

    fig, axes = plt.subplots(1, 2, figsize=(15, 7.2))
    fig.subplots_adjust(top=0.83, bottom=0.09, left=0.02, right=0.98, wspace=0.08)

    configs = [
        (axes[0], left_col, left_title, left_cbar_label, left_cmap),
        (axes[1], right_col, right_title, right_cbar_label, right_cmap),
    ]

    for ax, col, panel_title, cbar_label, cmap in configs:
        merged.plot(
            column=col,
            cmap=cmap,
            linewidth=0.55,
            edgecolor="#FFFFFF",
            legend=True,
            legend_kwds={
                "label": cbar_label,
                "orientation": "horizontal",
                "shrink": 0.65,
                "pad": 0.03,
                "aspect": 28,
            },
            ax=ax,
        )
        ax.set_title(panel_title, fontsize=11.5, fontweight="bold", pad=10, color=PALETTE["dark_text"])
        ax.set_axis_off()

        # Destaca as UFs de maior valor com callout externo para estados de menor área (DF, RJ, ES)
        top_states = merged.nlargest(annotate_top_n, col)
        offsets = {
            "DF": (2.3, -1.0),
            "RJ": (2.4, -1.3),
            "ES": (2.3, 0.4),
            "PB": (2.4, 0.6),
            "PE": (2.4, -0.3),
            "SE": (2.2, -0.8),
            "AL": (2.2, -0.2),
            "RN": (2.2, 1.0),
        }
        vmin, vmax = merged[col].min(), merged[col].max()
        for _, row in top_states.iterrows():
            pt = row["geometry"].representative_point()
            sigla = row["abbrev_state"]
            val_txt = f"{sigla}\n{format_ptbr_number(row[col], 1)}%"
            dx, dy = offsets.get(sigla, (0.0, 0.0))
            use_box = (dx != 0.0) or (dy != 0.0)
            rel = (row[col] - vmin) / (vmax - vmin) if vmax > vmin else 0.0
            txt_color = PALETTE["dark_text"] if (use_box or rel <= 0.55) else "#FFFFFF"
            ax.annotate(
                val_txt,
                xy=(pt.x, pt.y),
                xytext=(pt.x + dx, pt.y + dy),
                ha="center",
                va="center",
                fontsize=7.8,
                fontweight="bold",
                color=txt_color,
                arrowprops={"arrowstyle": "-", "color": PALETTE["dark_text"], "lw": 0.9} if use_box else None,
                bbox={
                    "boxstyle": "round,pad=0.2",
                    "facecolor": "#FFFFFF",
                    "alpha": 0.92,
                    "edgecolor": PALETTE["muted_text"],
                    "linewidth": 0.6,
                }
                if use_box
                else None,
            )

    add_title_and_source(fig, suptitle, subtitle, title_y=0.96, subtitle_y=0.905, source_y=0.015)
    return fig


def save_figure(
    fig: plt.Figure,
    filename: str,
    figures_dir: Path | str = FIGURES_DIR,
    dpi: int = 300,
) -> Path:
    """Salva figura em alta resolução no diretório reports/figures/."""
    out_dir = Path(figures_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / filename
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight", facecolor=PALETTE["bg"])
    return out_path
