"""Testes unitários e de integridade para pipeline de dados e modelagem econométrica."""

import numpy as np
import pandas as pd
import pytest

from src.config import RAW_DATA_PATH
from src.data_processing import (
    build_analytic_dataset,
    classify_escolaridade,
    classify_faixa_etaria_desc,
    classify_faixa_etaria_modelo,
    classify_formalidade,
    classify_iniciativa,
    classify_raca,
    classify_setor,
    classify_situacao_ocupacao,
    load_raw_data,
    validate_analytic_dataset,
)
from src.descriptive import panorama_ocupacao_mensal, resumo_por_uf, weighted_table
from src.modeling import (
    build_design_matrix_interaction,
    build_design_matrix_main,
    compute_calibration_bins,
    compute_diagnostics,
    compute_discrete_ame,
    compute_subgroup_predictions,
    compute_vif,
    fit_weighted_logit,
    summarize_logit,
)


def _sample_raw_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Ano": [2020, 2020, 2020, 2020, 2020, 2020],
            "V1013": [5, 5, 5, 5, 11, 11],
            "V1012": [1, 2, 3, 4, 1, 2],
            "UF": ["São Paulo", "Bahia", "Distrito Federal", "Paraná", "Rio de Janeiro", "Amazonas"],
            "V1032": [1000.0, 1200.0, 800.0, 950.0, 1100.0, 900.0],
            "A002": [14, 35, 42, 82, 28, 55],
            "A003": ["Mulher", "Homem", "Mulher", "Homem", "Mulher", "Homem"],
            "A004": ["Branca", "Parda", "Amarela", "Preta", "Indígena", "Ignorado"],
            "A005": [
                "Fundamental completa",
                "Superior completo",
                "Pós-graduação, mestrado ou doutorado",
                "Médio completo",
                "Superior incompleto",
                "Sem instrução",
            ],
            "C001": ["Sim", "Sim", "Sim", "Não", "Sim", "Sim"],
            "C002": ["Não", "Não", "Não", "Sim", "Não", "Não"],
            "C003": [
                None,
                None,
                None,
                "Estava em quarentena, isolamento, distanciamento social ou férias coletivas",
                None,
                None,
            ],
            "C007": [
                "Empregado do setor privado",
                "Policial militar ou bombeiro militar",
                "Conta própria",
                "Empregado do setor privado",
                "Trabalhador familiar não remunerado em ajuda a membro do domicílio ou parente",
                "Estava fora do mercado de trabalho (fazia apenas afazeres domésticos, cuidados de pessoas ou produção para próprio consumo)",
            ],
            "C007B": [
                "Não",
                None,
                None,
                "Sim, tem carteira de trabalho assinada",
                None,
                None,
            ],
            "C007D": [
                "Construção",
                None,
                "Educação",
                "Comércio no atacado e varejo;",
                "Reparação de veículos automotores e motocicletas",
                None,
            ],
            "C012": ["Sim", "Não", "Não", None, "Sim", None],
            "C013": [None, "Não", "Sim", None, None, None],
            "C014": ["Sim", None, "Não", None, "Sim", "Não"],
        }
    )


def test_classify_situacao_ocupacao():
    df = _sample_raw_df()
    res = classify_situacao_ocupacao(df)
    assert res.iloc[0] == "Ocupado presencial (não afastado)"
    assert res.iloc[1] == "Ocupado presencial (não afastado)"
    assert res.iloc[2] == "Ocupado em trabalho remoto"
    assert res.iloc[3] == "Afastado por distanciamento social"
    assert res.iloc[4] == "Ocupado presencial (não afastado)"


def test_classify_raca_and_escolaridade():
    raca = classify_raca(pd.Series(["Branca", "Amarela", "Preta", "Parda", "Indígena", "Ignorado", None]))
    assert raca.iloc[0] == "Branca e Amarela"
    assert raca.iloc[1] == "Branca e Amarela"
    assert list(raca.iloc[2:5]) == ["Pretos, Pardos e Indígenas"] * 3
    assert pd.isna(raca.iloc[5])
    assert pd.isna(raca.iloc[6])

    esc = classify_escolaridade(pd.Series(["Fundamental completa", "Pós-graduação, mestrado ou doutorado"]))
    assert esc.iloc[0] == "Fundamental completo"
    assert esc.cat.ordered is True
    assert len(esc.cat.categories) == 8
    assert esc.iloc[0] < esc.iloc[1]


def test_classify_faixas_etarias():
    idades = pd.Series([13, 14, 19, 25, 65, 80, 105])
    desc = classify_faixa_etaria_desc(idades)
    mod = classify_faixa_etaria_modelo(idades)

    assert pd.isna(desc.iloc[0])
    assert desc.iloc[1] == "14-19"
    assert desc.iloc[2] == "14-19"
    assert desc.iloc[5] == "80+"
    assert desc.iloc[6] == "80+"

    assert pd.isna(mod.iloc[0])
    assert mod.iloc[1] == "14-24"
    assert mod.iloc[3] == "25-34"
    assert mod.iloc[4] == "65+"


def test_classify_iniciativa_formalidade_setor():
    df = _sample_raw_df()
    inic = classify_iniciativa(df["C007"])
    form = classify_formalidade(df["C007"], df["C007B"], df["C014"])
    setor = classify_setor(df["C007"], df["C007D"])

    assert inic.iloc[0] == "Privado"
    assert inic.iloc[1] == "Público"
    assert inic.iloc[2] == "Outros"
    assert pd.isna(inic.iloc[5])

    assert form.iloc[0] == "Formal"  # Sem carteira mas contribui INSS (C014=='Sim')
    assert form.iloc[1] == "Formal"  # Militar
    assert form.iloc[2] == "Informal"  # Conta própria sem INSS
    assert form.iloc[4] == "Informal"  # Trabalhador familiar não remunerado
    assert pd.isna(form.iloc[5])

    assert setor.iloc[0] == "Indústria"  # Construção
    assert setor.iloc[1] == "Setor Público"  # Militar com C007D NaN
    assert setor.iloc[2] == "Serviços"  # Educação
    assert setor.iloc[4] == "Comércio"  # Reparação de veículos
    assert pd.isna(setor.iloc[5])


def test_build_analytic_dataset_amostra_modelo():
    df_raw = _sample_raw_df()
    analytic = build_analytic_dataset(df_raw)

    assert analytic.loc[0, "trabalho_remoto"] == 0
    assert analytic.loc[2, "trabalho_remoto"] == 1
    assert analytic.loc[2, "ensino_superior"] == 1
    assert analytic.loc[0, "ensino_superior"] == 0

    # Elegíveis para amostra_modelo: linhas 0, 1, 2, 4 (linha 3 afastada, linha 5 fora do mercado/Ignorado)
    assert analytic["amostra_modelo"].tolist() == [True, True, True, False, True, False]


def test_raw_dataset_end_to_end_integrity():
    assert RAW_DATA_PATH.exists(), f"Base bruta compacta não encontrada em {RAW_DATA_PATH}"
    df_raw = load_raw_data(RAW_DATA_PATH)
    assert df_raw.shape == (2_650_459, 18)

    df = build_analytic_dataset(df_raw)
    checks = validate_analytic_dataset(df)

    assert checks["n_total_linhas"] == 2_650_459
    assert checks["meses_presentes"] == [5, 6, 7, 8, 9, 10, 11]
    assert checks["n_ufs"] == 27
    assert checks["remoto_fora_nao_afastado"] == 0
    assert all(v == 0 for v in checks["nulos_amostra_modelo"].values())

    resumo = checks["resumo_mensal"]
    assert resumo.loc[5, "nao_afastados_milhoes"] == pytest.approx(66.949, abs=0.05)
    assert resumo.loc[5, "remotos_milhoes"] == pytest.approx(8.709, abs=0.05)
    assert resumo.loc[11, "nao_afastados_milhoes"] == pytest.approx(82.147, abs=0.05)
    assert resumo.loc[11, "remotos_milhoes"] == pytest.approx(7.330, abs=0.05)

    panorama = panorama_ocupacao_mensal(df)
    assert panorama.loc["    Em trabalho remoto", "Proporção nov/2020 (%)"] == pytest.approx(8.24, abs=0.15)
    assert panorama.loc["    Afastadas por distanciamento social", "Proporção nov/2020 (%)"] == pytest.approx(58.4, abs=0.2)

    uf_nov = resumo_por_uf(df, mes=11)
    assert len(uf_nov) == 27
    assert uf_nov["participacao_remoto_pct"].sum() == pytest.approx(100.0, abs=0.01)

    tab_multi = weighted_table(df[df["raca"].notna()], ["genero", "raca"], "mes_label")
    assert len(tab_multi) == 4
    assert list(tab_multi.columns) == ["mai/2020", "jun/2020", "jul/2020", "ago/2020", "set/2020", "out/2020", "nov/2020"]


def test_modeling_pipeline_synthetic():
    rng = np.random.default_rng(123)
    n = 1500
    df_syn = pd.DataFrame(
        {
            "mes": 5,
            "peso": rng.uniform(0.5, 2.5, n),
            "genero": pd.Categorical(rng.choice(["Homem", "Mulher"], n), categories=["Homem", "Mulher"]),
            "raca": pd.Categorical(
                rng.choice(["Branca e Amarela", "Pretos, Pardos e Indígenas"], n),
                categories=["Branca e Amarela", "Pretos, Pardos e Indígenas"],
            ),
            "faixa_etaria_modelo": pd.Categorical(
                rng.choice(["14-24", "25-34", "35-44", "45-54", "55-64", "65+"], n),
                categories=["14-24", "25-34", "35-44", "45-54", "55-64", "65+"],
            ),
            "regiao": pd.Categorical(
                rng.choice(["Sudeste", "Nordeste", "Sul", "Centro-Oeste", "Norte"], n),
                categories=["Sudeste", "Nordeste", "Sul", "Centro-Oeste", "Norte"],
            ),
            "formalidade": pd.Categorical(rng.choice(["Formal", "Informal"], n), categories=["Formal", "Informal"]),
            "iniciativa": pd.Categorical(rng.choice(["Privado", "Público", "Outros"], n), categories=["Privado", "Público", "Outros"]),
            "setor": pd.Categorical(
                rng.choice(["Serviços", "Indústria", "Comércio", "Setor Público", "Agricultura", "Outros"], n),
                categories=["Serviços", "Indústria", "Comércio", "Setor Público", "Agricultura", "Outros"],
            ),
            "escolaridade": pd.Categorical(
                rng.choice(
                    [
                        "Sem instrução",
                        "Fundamental incompleto",
                        "Fundamental completo",
                        "Médio incompleto",
                        "Médio completo",
                        "Superior incompleto",
                        "Superior completo",
                        "Pós-graduação, mestrado ou doutorado",
                    ],
                    n,
                ),
                categories=[
                    "Sem instrução",
                    "Fundamental incompleto",
                    "Fundamental completo",
                    "Médio incompleto",
                    "Médio completo",
                    "Superior incompleto",
                    "Superior completo",
                    "Pós-graduação, mestrado ou doutorado",
                ],
                ordered=True,
            ),
        }
    )
    df_syn["ensino_superior"] = df_syn["escolaridade"].isin(
        ["Superior completo", "Pós-graduação, mestrado ou doutorado"]
    ).astype("int8")

    linpred = (
        -1.8
        + 1.7 * df_syn["ensino_superior"]
        + 0.4 * (df_syn["genero"] == "Mulher").astype(float)
        - 0.5 * (df_syn["raca"] == "Pretos, Pardos e Indígenas").astype(float)
        - 0.8 * (df_syn["setor"] == "Agricultura").astype(float)
        + 0.3 * df_syn["ensino_superior"] * (df_syn["raca"] == "Pretos, Pardos e Indígenas").astype(float)
    )
    prob = 1.0 / (1.0 + np.exp(-linpred))
    df_syn["trabalho_remoto"] = rng.binomial(1, prob, n).astype("int8")

    X1, y1, w1 = build_design_matrix_main(df_syn)
    res1 = fit_weighted_logit(y1, X1, w1)
    summary1 = summarize_logit(res1, X1, w1, include_ame=True)
    diag1 = compute_diagnostics(res1, y1, X1, w1)

    sup_row = summary1.loc[summary1["variavel"] == "escolaridade__Superior completo"].iloc[0]
    assert sup_row["coef"] > 0.8
    assert sup_row["ame_pp"] > 10.0
    assert sup_row["ame_ic_sup"] > sup_row["ame_ic_inf"]
    assert 0.5 < diag1["roc_auc_ponderado"] <= 1.0
    assert 0.0 < diag1["mcfadden_pseudo_r2"] < 1.0

    X2, y2, w2 = build_design_matrix_interaction(df_syn)
    res2 = fit_weighted_logit(y2, X2, w2)
    sub_preds = compute_subgroup_predictions(
        res2, df_syn, build_design_matrix_interaction, w2, dimensions=["genero", "raca"]
    )
    assert len(sub_preds) == 4
    assert (sub_preds["ganho_superior_pp"] > 0).all()
    assert (sub_preds["ganho_ic_sup_pp"] > sub_preds["ganho_ic_inf_pp"]).all()

    cal = compute_calibration_bins(y2, res2.predict(X2), w2, n_bins=5)
    assert len(cal) == 5

    vif = compute_vif(X1, max_rows=None)
    assert not vif.empty
    assert (vif["vif"] >= 1.0).all()
