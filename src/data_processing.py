"""Carga, limpeza, recodificação vetorizada e validação dos microdados da PNAD Covid-19."""

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.config import (
    ESCOLARIDADE_ORDER,
    FAIXA_ETARIA_DESC_ORDER,
    FAIXA_ETARIA_MODELO_ORDER,
    FORMALIDADE_ORDER,
    GENERO_ORDER,
    INICIATIVA_ORDER,
    MES_LABELS,
    MES_ORDER,
    PROCESSED_DATA_PATH,
    RACA_ORDER,
    RAW_DATA_PATH,
    REGIAO_ORDER,
    SETOR_ORDER,
    SITUACAO_OCUPACAO_ORDER,
    UF_TO_REGIAO,
    UF_TO_SIGLA,
)

_MILITAR_VALUES = [
    "Policial militar ou bombeiro militar",
    "Militar do exercito, marinha ou aeronáutica",
]

_OUTROS_INICIATIVA_VALUES = [
    "Conta própria",
    "Empregador",
    "Trabalhador doméstico (empregado doméstico, cuidados, babá)",
    "Trabalhador familiar não remunerado em ajuda a membro do domicílio ou parente",
]

_INDUSTRIA_ATIVIDADES = [
    "Indústria da transformação (inclusive confecção e fabricação caseira)",
    "Extração de petróleo, carvão mineral, minerais metálicos, pedra, areia, sal etc.",
    "Construção",
    "Fornecimento de eletricidade e gás, água, esgoto e coleta de lixo",
]

_COMERCIO_ATIVIDADES = [
    "Comércio no atacado e varejo;",
    "Reparação de veículos automotores e motocicletas",
]


def load_raw_data(path: Path | str = RAW_DATA_PATH) -> pd.DataFrame:
    """Carrega a base compacta de entrada da PNAD Covid-19 em Parquet."""
    return pd.read_parquet(path)


def classify_situacao_ocupacao(df: pd.DataFrame) -> pd.Series:
    """Classifica a situação detalhada de ocupação e afastamento na semana de referência."""
    c001 = df["C001"].astype("string")
    c002 = df["C002"].astype("string")
    c003 = df["C003"].astype("string")
    c013 = df["C013"].astype("string")

    cond_remoto = c001.eq("Sim").fillna(False) & c013.eq("Sim").fillna(False)
    cond_presencial = c001.eq("Sim").fillna(False) & (~cond_remoto)
    cond_afast_dist = (
        c001.eq("Não").fillna(False)
        & c002.eq("Sim").fillna(False)
        & c003.eq("Estava em quarentena, isolamento, distanciamento social ou férias coletivas").fillna(False)
    )
    cond_afast_outros = c001.eq("Não").fillna(False) & c002.eq("Sim").fillna(False) & (~cond_afast_dist)

    values = np.select(
        [cond_remoto, cond_presencial, cond_afast_dist, cond_afast_outros],
        [
            "Ocupado em trabalho remoto",
            "Ocupado presencial (não afastado)",
            "Afastado por distanciamento social",
            "Afastado por outros motivos",
        ],
        default="Fora da ocupação",
    )
    return pd.Series(
        pd.Categorical(values, categories=SITUACAO_OCUPACAO_ORDER, ordered=False),
        index=df.index,
        name="situacao_ocupacao",
    )


def classify_raca(series: pd.Series) -> pd.Series:
    """Agrupa A004 em 'Branca e Amarela' vs. 'Pretos, Pardos e Indígenas' (Ignorado -> NaN)."""
    mapping = {
        "Branca": "Branca e Amarela",
        "Amarela": "Branca e Amarela",
        "Preta": "Pretos, Pardos e Indígenas",
        "Parda": "Pretos, Pardos e Indígenas",
        "Indígena": "Pretos, Pardos e Indígenas",
    }
    mapped = series.astype("string").map(mapping)
    return pd.Series(
        pd.Categorical(mapped, categories=RACA_ORDER, ordered=False),
        index=series.index,
        name="raca",
    )


def classify_escolaridade(series: pd.Series) -> pd.Series:
    """Padroniza e ordena os 8 níveis de escolaridade da variável A005."""
    cleaned = series.astype("string").replace({"Fundamental completa": "Fundamental completo"})
    return pd.Series(
        pd.Categorical(cleaned, categories=ESCOLARIDADE_ORDER, ordered=True),
        index=series.index,
        name="escolaridade",
    )


def classify_faixa_etaria_desc(series: pd.Series) -> pd.Series:
    """Agrupa idades (14+) em faixas etárias descritivas decenais."""
    numeric_age = pd.to_numeric(series, errors="coerce")
    cut = pd.cut(
        numeric_age,
        bins=[13, 19, 29, 39, 49, 59, 69, 79, 150],
        labels=FAIXA_ETARIA_DESC_ORDER,
        ordered=True,
    )
    return pd.Series(cut, index=series.index, name="faixa_etaria_desc")


def classify_faixa_etaria_modelo(series: pd.Series) -> pd.Series:
    """Agrupa idades (14+) nas faixas etárias de controle para os modelos Logit."""
    numeric_age = pd.to_numeric(series, errors="coerce")
    cut = pd.cut(
        numeric_age,
        bins=[13, 24, 34, 44, 54, 64, 150],
        labels=FAIXA_ETARIA_MODELO_ORDER,
        ordered=True,
    )
    return pd.Series(cut, index=series.index, name="faixa_etaria_modelo")


def classify_iniciativa(series: pd.Series) -> pd.Series:
    """Classifica C007 em Privado, Público ou Outros (excluindo fora do mercado de trabalho)."""
    s = series.astype("string")
    cond_privado = s.eq("Empregado do setor privado").fillna(False)
    cond_publico = (
        s.eq("Empregado do setor público (inclusive empresas de economia mista)").fillna(False)
        | s.isin(_MILITAR_VALUES).fillna(False)
    )
    cond_outros = s.isin(_OUTROS_INICIATIVA_VALUES).fillna(False)

    values = np.select(
        [cond_privado, cond_publico, cond_outros],
        ["Privado", "Público", "Outros"],
        default=None,
    )
    return pd.Series(
        pd.Categorical(values, categories=INICIATIVA_ORDER, ordered=False),
        index=series.index,
        name="iniciativa",
    )


def classify_formalidade(c007: pd.Series, c007b: pd.Series, c014: pd.Series) -> pd.Series:
    """Classifica vínculo formal considerando carteira, estatutário, militares e contribuição ao INSS."""
    s7 = c007.astype("string")
    s7b = c007b.astype("string")
    s14 = c014.astype("string")

    invalid = s7.isna() | s7.str.startswith("Estava fora").fillna(False)
    is_militar = s7.isin(_MILITAR_VALUES).fillna(False)
    is_carteira_estat = s7b.isin(
        ["Sim, tem carteira de trabalho assinada", "Sim, é servidor público estatutário"]
    ).fillna(False)
    is_familiar_nao_rem = s7.eq(
        "Trabalhador familiar não remunerado em ajuda a membro do domicílio ou parente"
    ).fillna(False)
    is_contrib_inss = (~is_familiar_nao_rem) & s14.eq("Sim").fillna(False)

    is_formal = is_militar | is_carteira_estat | is_contrib_inss

    values = np.select(
        [invalid, is_formal],
        [None, "Formal"],
        default="Informal",
    )
    return pd.Series(
        pd.Categorical(values, categories=FORMALIDADE_ORDER, ordered=False),
        index=c007.index,
        name="formalidade",
    )


def classify_setor(c007: pd.Series, c007d: pd.Series) -> pd.Series:
    """Classifica o setor econômico (C007D), alocando militares explicitamente no Setor Público."""
    s7 = c007.astype("string")
    s7d = c007d.astype("string")

    is_militar = s7.isin(_MILITAR_VALUES).fillna(False)
    invalid = s7.isna() | s7.str.startswith("Estava fora").fillna(False) | ((~is_militar) & s7d.isna())

    cond_publico = is_militar | s7d.eq("Administração pública (governo federal, estadual e municipal)").fillna(False)
    cond_agro = s7d.eq("Agricultura, pecuária, produção florestal e pesca").fillna(False)
    cond_comercio = s7d.isin(_COMERCIO_ATIVIDADES).fillna(False)
    cond_industria = s7d.isin(_INDUSTRIA_ATIVIDADES).fillna(False)
    cond_outros = s7d.eq("Outro").fillna(False)

    values = np.select(
        [invalid, cond_publico, cond_agro, cond_comercio, cond_industria, cond_outros],
        [None, "Setor Público", "Agricultura", "Comércio", "Indústria", "Outros"],
        default="Serviços",
    )
    return pd.Series(
        pd.Categorical(values, categories=SETOR_ORDER, ordered=False),
        index=c007.index,
        name="setor",
    )


def build_analytic_dataset(df_raw: pd.DataFrame) -> pd.DataFrame:
    """Constrói a base analítica limpa e padronizada a partir dos microdados brutos."""
    mes = pd.to_numeric(df_raw["V1013"], errors="raise").astype("int8")
    uf_str = df_raw["UF"].astype("string")
    genero_str = df_raw["A003"].astype("string")

    escolaridade = classify_escolaridade(df_raw["A005"])
    raca = classify_raca(df_raw["A004"])
    faixa_etaria_desc = classify_faixa_etaria_desc(df_raw["A002"])
    faixa_etaria_modelo = classify_faixa_etaria_modelo(df_raw["A002"])
    situacao_ocupacao = classify_situacao_ocupacao(df_raw)
    iniciativa = classify_iniciativa(df_raw["C007"])
    formalidade = classify_formalidade(df_raw["C007"], df_raw["C007B"], df_raw["C014"])
    setor = classify_setor(df_raw["C007"], df_raw["C007D"])

    c001_sim = df_raw["C001"].astype("string").eq("Sim").fillna(False)
    c002_sim = df_raw["C002"].astype("string").eq("Sim").fillna(False)
    c012_notna = df_raw["C012"].notna()
    c013_sim = df_raw["C013"].astype("string").eq("Sim").fillna(False)
    peso = pd.to_numeric(df_raw["V1032"], errors="raise").astype("float64")

    genero = pd.Categorical(genero_str, categories=GENERO_ORDER, ordered=False)
    regiao = pd.Categorical(uf_str.map(UF_TO_REGIAO), categories=REGIAO_ORDER, ordered=False)
    uf_sigla = pd.Categorical(uf_str.map(UF_TO_SIGLA), categories=sorted(UF_TO_SIGLA.values()), ordered=False)
    mes_label = pd.Categorical(mes.map(MES_LABELS), categories=MES_ORDER, ordered=True)

    ensino_superior = escolaridade.isin(
        ["Superior completo", "Pós-graduação, mestrado ou doutorado"]
    ).astype("int8")
    trabalho_remoto = (c001_sim & c013_sim).astype("int8")

    amostra_modelo = (
        c001_sim
        & c012_notna
        & raca.notna()
        & faixa_etaria_modelo.notna()
        & pd.Series(genero, index=df_raw.index).notna()
        & pd.Series(regiao, index=df_raw.index).notna()
        & escolaridade.notna()
        & iniciativa.notna()
        & formalidade.notna()
        & setor.notna()
        & (peso > 0)
    ).astype(bool)

    return pd.DataFrame(
        {
            "ano": pd.to_numeric(df_raw["Ano"], errors="raise").astype("int16"),
            "mes": mes,
            "mes_label": mes_label,
            "semana": pd.to_numeric(df_raw["V1012"], errors="raise").astype("int8"),
            "uf": pd.Categorical(uf_str),
            "uf_sigla": uf_sigla,
            "regiao": regiao,
            "peso": peso,
            "idade": pd.to_numeric(df_raw["A002"], errors="raise").astype("int16"),
            "faixa_etaria_desc": faixa_etaria_desc,
            "faixa_etaria_modelo": faixa_etaria_modelo,
            "genero": genero,
            "raca": raca,
            "escolaridade": escolaridade,
            "ensino_superior": ensino_superior,
            "ocupado": (c001_sim | c002_sim).astype(bool),
            "ocupado_nao_afastado": c001_sim.astype(bool),
            "afastado": c002_sim.astype(bool),
            "trabalho_remoto": trabalho_remoto,
            "situacao_ocupacao": situacao_ocupacao,
            "iniciativa": iniciativa,
            "formalidade": formalidade,
            "setor": setor,
            "amostra_modelo": amostra_modelo,
        },
        index=df_raw.index,
    )


def save_analytic_dataset(df: pd.DataFrame, path: Path | str = PROCESSED_DATA_PATH) -> Path:
    """Persiste a base analítica em formato Parquet comprimido com zstd."""
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, compression="zstd", index=False)
    return out_path


def load_analytic_dataset(path: Path | str = PROCESSED_DATA_PATH) -> pd.DataFrame:
    """Carrega a base analítica tratada."""
    return pd.read_parquet(path)


def validate_analytic_dataset(df: pd.DataFrame) -> dict[str, Any]:
    """Executa verificações estruturais e de consistência na base analítica."""
    model_df = df[df["amostra_modelo"]]
    required_model_cols = [
        "trabalho_remoto",
        "ensino_superior",
        "escolaridade",
        "genero",
        "raca",
        "faixa_etaria_modelo",
        "regiao",
        "iniciativa",
        "formalidade",
        "setor",
        "peso",
    ]
    null_counts_model = {c: int(model_df[c].isna().sum()) for c in required_model_cols}

    monthly_summary = (
        df.groupby("mes", observed=True)
        .apply(
            lambda g: pd.Series(
                {
                    "ocupados_milhoes": float(g.loc[g["ocupado"], "peso"].sum() / 1e6),
                    "nao_afastados_milhoes": float(g.loc[g["ocupado_nao_afastado"], "peso"].sum() / 1e6),
                    "remotos_milhoes": float(g.loc[g["trabalho_remoto"] == 1, "peso"].sum() / 1e6),
                    "amostra_modelo_n": int(g["amostra_modelo"].sum()),
                }
            ),
            include_groups=False,
        )
        .round(3)
    )

    remote_outside_model = int(df.loc[(df["trabalho_remoto"] == 1) & (~df["ocupado_nao_afastado"]), :].shape[0])

    return {
        "n_total_linhas": int(len(df)),
        "n_colunas": int(df.shape[1]),
        "meses_presentes": sorted(int(m) for m in df["mes"].unique()),
        "n_ufs": int(df["uf_sigla"].nunique()),
        "n_amostra_modelo": int(len(model_df)),
        "nulos_amostra_modelo": null_counts_model,
        "valores_trabalho_remoto": sorted(int(v) for v in df["trabalho_remoto"].unique()),
        "remoto_fora_nao_afastado": remote_outside_model,
        "resumo_mensal": monthly_summary,
    }
