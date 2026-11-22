"""Funções auxiliares para tabelas e estatísticas descritivas ponderadas."""

from typing import Sequence

import numpy as np
import pandas as pd


def _to_crosstab_arg(df: pd.DataFrame, cols: str | Sequence[str]):
    if isinstance(cols, str):
        return df[cols]
    return [df[c] for c in cols]


def weighted_table(
    df: pd.DataFrame,
    index_col: str | Sequence[str],
    col_col: str | Sequence[str] | None = None,
    weight_col: str = "peso",
    scale: float = 1e3,
    normalize: str | None = None,
) -> pd.DataFrame:
    """Calcula tabela cruzada ou distribuição unidimensional ponderada pelos pesos amostrais."""
    if col_col is None:
        grouped = df.groupby(index_col, observed=False)[weight_col].sum() / scale
        out = grouped.to_frame(name="total")
        if normalize in {"all", "columns"}:
            total = out["total"].sum()
            out["total"] = np.where(total > 0, out["total"] / total * 100.0, 0.0)
        return out

    tab = (
        pd.crosstab(
            _to_crosstab_arg(df, index_col),
            _to_crosstab_arg(df, col_col),
            values=df[weight_col],
            aggfunc="sum",
            dropna=False,
        ).fillna(0.0)
        / scale
    )
    if normalize == "columns":
        col_sums = tab.sum(axis=0).replace(0.0, np.nan)
        tab = tab.div(col_sums, axis=1).fillna(0.0) * 100.0
    elif normalize == "index":
        row_sums = tab.sum(axis=1).replace(0.0, np.nan)
        tab = tab.div(row_sums, axis=0).fillna(0.0) * 100.0
    elif normalize == "all":
        grand_total = tab.values.sum()
        if grand_total > 0:
            tab = tab / grand_total * 100.0
    return tab


def weighted_remote_rate(
    df: pd.DataFrame,
    group_cols: str | Sequence[str],
    target_col: str = "trabalho_remoto",
    weight_col: str = "peso",
    scale: float = 1e3,
) -> pd.DataFrame:
    """Retorna contingente total, contingente remoto, taxa interna (%) e participação (%)."""
    cols = [group_cols] if isinstance(group_cols, str) else list(group_cols)
    work = df[cols + [target_col, weight_col]].copy()
    work["_peso_remoto"] = work[weight_col] * work[target_col].astype(float)

    agg = work.groupby(cols, observed=False)[["_peso_remoto", weight_col]].sum()
    total_remotos = agg["_peso_remoto"].sum()
    total_geral = agg[weight_col].sum()

    res = pd.DataFrame(
        {
            "total_ocupados_mil": agg[weight_col] / scale,
            "remotos_mil": agg["_peso_remoto"] / scale,
            "taxa_remoto_pct": np.where(
                agg[weight_col] > 0,
                agg["_peso_remoto"] / agg[weight_col] * 100.0,
                0.0,
            ),
            "participacao_remoto_pct": np.where(
                total_remotos > 0,
                agg["_peso_remoto"] / total_remotos * 100.0,
                0.0,
            ),
            "participacao_ocupados_pct": np.where(
                total_geral > 0,
                agg[weight_col] / total_geral * 100.0,
                0.0,
            ),
        },
        index=agg.index,
    )
    return res


def panorama_ocupacao_mensal(df: pd.DataFrame) -> pd.DataFrame:
    """Constrói tabela mensal de ocupados, trabalho remoto, presencial e afastamentos (em milhares)."""
    mes_cols = list(df["mes_label"].cat.categories) if hasattr(df["mes_label"], "cat") else sorted(df["mes_label"].unique())

    ocup_total = df[df["ocupado"]].groupby("mes_label", observed=False)["peso"].sum() / 1e3
    nao_afast = df[df["ocupado_nao_afastado"]].groupby("mes_label", observed=False)["peso"].sum() / 1e3
    remoto = (
        df[df["situacao_ocupacao"] == "Ocupado em trabalho remoto"]
        .groupby("mes_label", observed=False)["peso"]
        .sum()
        / 1e3
    )
    presencial = (
        df[df["situacao_ocupacao"] == "Ocupado presencial (não afastado)"]
        .groupby("mes_label", observed=False)["peso"]
        .sum()
        / 1e3
    )
    afastado = df[df["afastado"]].groupby("mes_label", observed=False)["peso"].sum() / 1e3
    afast_dist = (
        df[df["situacao_ocupacao"] == "Afastado por distanciamento social"]
        .groupby("mes_label", observed=False)["peso"]
        .sum()
        / 1e3
    )
    afast_outros = (
        df[df["situacao_ocupacao"] == "Afastado por outros motivos"]
        .groupby("mes_label", observed=False)["peso"]
        .sum()
        / 1e3
    )

    tabela = pd.DataFrame(
        [
            ocup_total,
            nao_afast,
            remoto,
            presencial,
            afastado,
            afast_dist,
            afast_outros,
        ],
        index=[
            "Pessoas ocupadas (total)",
            "  Ocupadas não afastadas",
            "    Em trabalho remoto",
            "    Em trabalho presencial",
            "  Ocupadas e afastadas",
            "    Afastadas por distanciamento social",
            "    Afastadas por outras razões",
        ],
    )[mes_cols]

    nov_col = mes_cols[-1]
    nov_ocup = tabela.loc["Pessoas ocupadas (total)", nov_col]
    nov_afast = tabela.loc["  Ocupadas e afastadas", nov_col]

    tabela["Proporção nov/2020 (%)"] = [
        100.0,
        tabela.loc["  Ocupadas não afastadas", nov_col] / nov_ocup * 100.0,
        tabela.loc["    Em trabalho remoto", nov_col] / nov_ocup * 100.0,
        tabela.loc["    Em trabalho presencial", nov_col] / nov_ocup * 100.0,
        tabela.loc["  Ocupadas e afastadas", nov_col] / nov_ocup * 100.0,
        tabela.loc["    Afastadas por distanciamento social", nov_col] / nov_afast * 100.0,
        tabela.loc["    Afastadas por outras razões", nov_col] / nov_afast * 100.0,
    ]
    return tabela


def resumo_por_uf(df: pd.DataFrame, mes: int = 11) -> pd.DataFrame:
    """Agrega trabalho remoto e ensino superior por Unidade da Federação em um mês de referência."""
    sub = df[(df["mes"] == mes) & df["ocupado"]].copy()
    sub["_peso_remoto"] = sub["peso"] * sub["trabalho_remoto"].astype(float)
    sub["_peso_sup"] = sub["peso"] * sub["ensino_superior"].astype(float)
    sub["_peso_nao_afast"] = sub["peso"] * sub["ocupado_nao_afastado"].astype(float)

    agg = sub.groupby(["uf_sigla", "uf", "regiao"], observed=True)[
        ["peso", "_peso_nao_afast", "_peso_remoto", "_peso_sup"]
    ].sum()
    agg = agg.reset_index()
    total_remotos_br = agg["_peso_remoto"].sum()

    return pd.DataFrame(
        {
            "uf_sigla": agg["uf_sigla"].astype(str),
            "uf": agg["uf"].astype(str),
            "regiao": agg["regiao"].astype(str),
            "ocupados_mil": agg["peso"] / 1e3,
            "remotos_mil": agg["_peso_remoto"] / 1e3,
            "participacao_remoto_pct": np.where(
                total_remotos_br > 0, agg["_peso_remoto"] / total_remotos_br * 100.0, 0.0
            ),
            "taxa_remoto_ocupados_pct": np.where(
                agg["peso"] > 0, agg["_peso_remoto"] / agg["peso"] * 100.0, 0.0
            ),
            "taxa_remoto_nao_afast_pct": np.where(
                agg["_peso_nao_afast"] > 0, agg["_peso_remoto"] / agg["_peso_nao_afast"] * 100.0, 0.0
            ),
            "prop_superior_ocupados_pct": np.where(
                agg["peso"] > 0, agg["_peso_sup"] / agg["peso"] * 100.0, 0.0
            ),
            "prop_sem_superior_ocupados_pct": np.where(
                agg["peso"] > 0, (1.0 - agg["_peso_sup"] / agg["peso"]) * 100.0, 0.0
            ),
        }
    ).sort_values("participacao_remoto_pct", ascending=False, ignore_index=True)


def resumo_por_regiao(df: pd.DataFrame, mes: int = 11) -> pd.DataFrame:
    """Agrega trabalho remoto e escolaridade superior por macrorregião."""
    uf_df = resumo_por_uf(df, mes=mes)
    sub = df[(df["mes"] == mes) & df["ocupado"]].copy()
    sub["_peso_sup"] = sub["peso"] * sub["ensino_superior"].astype(float)
    sup_reg = sub.groupby("regiao", observed=False)[["_peso_sup", "peso"]].sum()

    reg = (
        uf_df.groupby("regiao", observed=False)[["ocupados_mil", "remotos_mil", "participacao_remoto_pct"]]
        .sum()
        .reset_index()
    )
    reg["taxa_remoto_ocupados_pct"] = reg["remotos_mil"] / reg["ocupados_mil"] * 100.0
    reg["prop_superior_ocupados_pct"] = [
        float(sup_reg.loc[r, "_peso_sup"] / sup_reg.loc[r, "peso"] * 100.0) for r in reg["regiao"]
    ]
    return reg.sort_values("participacao_remoto_pct", ascending=False, ignore_index=True)
