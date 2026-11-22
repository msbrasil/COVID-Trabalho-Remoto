"""Modelagem Logit ponderada, efeitos marginais médios, previsões por subgrupo e diagnósticos."""

from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import norm
from sklearn.metrics import brier_score_loss, roc_auc_score, roc_curve
from statsmodels.stats.outliers_influence import variance_inflation_factor

REF_CATEGORIES: dict[str, str] = {
    "genero": "Homem",
    "raca": "Branca e Amarela",
    "faixa_etaria_modelo": "35-44",
    "regiao": "Sudeste",
    "formalidade": "Informal",
    "iniciativa": "Privado",
    "setor": "Serviços",
    "escolaridade": "Médio completo",
}

MODEL1_VARIABLES: list[str] = [
    "escolaridade",
    "genero",
    "raca",
    "faixa_etaria_modelo",
    "regiao",
    "formalidade",
    "iniciativa",
    "setor",
]

MODEL2_CONTROLS: list[str] = [
    "genero",
    "raca",
    "faixa_etaria_modelo",
    "regiao",
    "formalidade",
    "iniciativa",
    "setor",
]

INTERACTION_DUMMY_COLS: list[str] = [
    "genero__Mulher",
    "raca__Pretos, Pardos e Indígenas",
    "iniciativa__Público",
    "iniciativa__Outros",
    "formalidade__Formal",
    "setor__Indústria",
    "setor__Comércio",
    "setor__Setor Público",
    "setor__Agricultura",
    "setor__Outros",
]

PRETTY_LABELS: dict[str, str] = {
    "const": "Intercepto",
    "ensino_superior": "Ensino superior completo ou pós",
    "escolaridade__Sem instrução": "Escolaridade: Sem instrução",
    "escolaridade__Fundamental incompleto": "Escolaridade: Fundamental incompleto",
    "escolaridade__Fundamental completo": "Escolaridade: Fundamental completo",
    "escolaridade__Médio incompleto": "Escolaridade: Médio incompleto",
    "escolaridade__Superior incompleto": "Escolaridade: Superior incompleto",
    "escolaridade__Superior completo": "Escolaridade: Superior completo",
    "escolaridade__Pós-graduação, mestrado ou doutorado": "Escolaridade: Pós-graduação/Mestrado/Doutorado",
    "genero__Mulher": "Gênero: Mulher (ref. Homem)",
    "raca__Pretos, Pardos e Indígenas": "Raça: Pretos, Pardos e Indígenas",
    "faixa_etaria_modelo__14-24": "Idade: 14–24 anos (ref. 35–44)",
    "faixa_etaria_modelo__25-34": "Idade: 25–34 anos",
    "faixa_etaria_modelo__45-54": "Idade: 45–54 anos",
    "faixa_etaria_modelo__55-64": "Idade: 55–64 anos",
    "faixa_etaria_modelo__65+": "Idade: 65+ anos",
    "regiao__Nordeste": "Região: Nordeste (ref. Sudeste)",
    "regiao__Sul": "Região: Sul",
    "regiao__Centro-Oeste": "Região: Centro-Oeste",
    "regiao__Norte": "Região: Norte",
    "formalidade__Formal": "Vínculo: Formal (ref. Informal)",
    "iniciativa__Público": "Iniciativa: Público (ref. Privado)",
    "iniciativa__Outros": "Iniciativa: Outros (Conta própria/Empreg./Dom.)",
    "setor__Indústria": "Setor: Indústria (ref. Serviços)",
    "setor__Comércio": "Setor: Comércio",
    "setor__Setor Público": "Setor: Administração Pública",
    "setor__Agricultura": "Setor: Agricultura",
    "setor__Outros": "Setor: Outros",
    "sup_x_genero__Mulher": "Superior × Mulher",
    "sup_x_raca__Pretos, Pardos e Indígenas": "Superior × Pretos, Pardos e Indígenas",
    "sup_x_iniciativa__Público": "Superior × Iniciativa Pública",
    "sup_x_iniciativa__Outros": "Superior × Outras Iniciativas",
    "sup_x_formalidade__Formal": "Superior × Vínculo Formal",
    "sup_x_setor__Indústria": "Superior × Indústria",
    "sup_x_setor__Comércio": "Superior × Comércio",
    "sup_x_setor__Setor Público": "Superior × Administração Pública",
    "sup_x_setor__Agricultura": "Superior × Agricultura",
    "sup_x_setor__Outros": "Superior × Outros Setores",
}


def format_var_label(name: str) -> str:
    """Retorna rótulo legível em português para uma coluna da matriz de desenho."""
    return PRETTY_LABELS.get(name, name)


def prepare_model_sample(df: pd.DataFrame, mes: int) -> pd.DataFrame:
    """Filtra a amostra analítica elegível para o mês especificado."""
    return df.loc[(df["mes"] == mes) & df["amostra_modelo"]].copy()


def _encode_categoricals(df_sub: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    for col in cols:
        ref = REF_CATEGORIES[col]
        dummies = pd.get_dummies(df_sub[col], prefix=col, prefix_sep="__", dtype="float64")
        ref_col = f"{col}__{ref}"
        if ref_col in dummies.columns:
            dummies = dummies.drop(columns=[ref_col])
        parts.append(dummies)
    return pd.concat(parts, axis=1)


def build_design_matrix_main(df_sub: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, np.ndarray]:
    """Constrói matriz X (com constante), vetor y e pesos para o Modelo 1 (efeitos principais)."""
    X_body = _encode_categoricals(df_sub, MODEL1_VARIABLES)
    X = sm.add_constant(X_body, has_constant="add")
    y = df_sub["trabalho_remoto"].astype("float64")
    weights = df_sub["peso"].to_numpy(dtype="float64")
    return X, y, weights


def build_design_matrix_interaction(df_sub: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, np.ndarray]:
    """Constrói matriz X (com constante), vetor y e pesos para o Modelo 2 (com interações do superior)."""
    sup = df_sub["ensino_superior"].astype("float64").rename("ensino_superior")
    X_controls = _encode_categoricals(df_sub, MODEL2_CONTROLS)
    X_body = pd.concat([sup, X_controls], axis=1)

    for col in INTERACTION_DUMMY_COLS:
        if col not in X_body.columns:
            X_body[col] = 0.0
        X_body[f"sup_x_{col}"] = X_body["ensino_superior"] * X_body[col]

    X = sm.add_constant(X_body, has_constant="add")
    y = df_sub["trabalho_remoto"].astype("float64")
    weights = df_sub["peso"].to_numpy(dtype="float64")
    return X, y, weights


def fit_weighted_logit(y: pd.Series | np.ndarray, X: pd.DataFrame, weights: np.ndarray) -> Any:
    """Ajusta Logit via Binomial GLM ponderado com pesos normalizados e erros robustos HC1."""
    w = np.asarray(weights, dtype="float64")
    if np.any(w <= 0):
        raise ValueError("Todos os pesos amostrais devem ser estritamente positivos.")
    w_norm = w * (len(w) / w.sum())
    model = sm.GLM(
        np.asarray(y, dtype="float64"),
        X.astype("float64"),
        family=sm.families.Binomial(),
        var_weights=w_norm,
    )
    return model.fit(cov_type="HC1")


def _logistic_cdf(z: np.ndarray) -> np.ndarray:
    z_clipped = np.clip(z, -500.0, 500.0)
    return 1.0 / (1.0 + np.exp(-z_clipped))


def _infer_categorical_groups(columns: list[str]) -> dict[str, list[str]]:
    groups: dict[str, list[str]] = {}
    for col in columns:
        if col == "const":
            continue
        if "__" in col and not col.startswith("sup_x_"):
            prefix = col.split("__", 1)[0]
            groups.setdefault(prefix, []).append(col)
        else:
            groups[col] = [col]
    return groups


def compute_discrete_ame(
    res: Any,
    X: pd.DataFrame,
    weights: np.ndarray,
    binary_cols: list[str] | None = None,
) -> pd.DataFrame:
    """Calcula Efeitos Marginais Médios (AME em p.p.) discretos com IC 95% via método delta.

    Para dummies pertencentes a uma mesma variável categórica (prefixo ``col__``),
    o cenário contrafactual zera todas as irmãs da mesma variável no estado base (0) e
    ativa exclusivamente a categoria avaliada no estado (1), preservando exclusividade mútua.
    """
    cols = list(X.columns)
    col_idx = {c: i for i, c in enumerate(cols)}
    X_mat = X.to_numpy(dtype="float64")
    beta = np.asarray(res.params, dtype="float64")
    cov = np.asarray(res.cov_params(), dtype="float64")

    w = np.asarray(weights, dtype="float64")
    w_norm = w / w.sum()

    groups = _infer_categorical_groups(cols)
    col_to_group = {c: gcols for gcols in groups.values() for c in gcols}

    target_cols = binary_cols if binary_cols is not None else [c for c in cols if c != "const"]

    rows: list[dict[str, Any]] = []
    for col in target_cols:
        sibling_idxs = [col_idx[c] for c in col_to_group.get(col, [col])]
        target_i = col_idx[col]

        X0 = X_mat.copy()
        X0[:, sibling_idxs] = 0.0
        X1 = X0.copy()
        X1[:, target_i] = 1.0

        p1 = _logistic_cdf(X1 @ beta)
        p0 = _logistic_cdf(X0 @ beta)

        ame = float(np.sum(w_norm * (p1 - p0)))
        dp1 = w_norm * p1 * (1.0 - p1)
        dp0 = w_norm * p0 * (1.0 - p0)
        grad = dp1 @ X1 - dp0 @ X0

        se = float(np.sqrt(max(float(grad @ cov @ grad), 0.0)))
        z_stat = ame / se if se > 0 else np.nan
        p_val = float(2.0 * (1.0 - norm.cdf(abs(z_stat)))) if np.isfinite(z_stat) else np.nan

        rows.append(
            {
                "variavel": col,
                "label": format_var_label(col),
                "ame_pp": ame * 100.0,
                "ame_se_pp": se * 100.0,
                "ame_ic_inf": (ame - 1.96 * se) * 100.0,
                "ame_ic_sup": (ame + 1.96 * se) * 100.0,
                "ame_p_valor": p_val,
            }
        )

    return pd.DataFrame(rows)


def summarize_logit(
    res: Any,
    X: pd.DataFrame | None = None,
    weights: np.ndarray | None = None,
    include_ame: bool = True,
) -> pd.DataFrame:
    """Consolida coeficientes, erros robustos HC1, Odds Ratios e opcionalmente AMEs."""
    params = res.params
    bse = res.bse
    pvals = res.pvalues
    zvals = res.tvalues

    df_out = pd.DataFrame(
        {
            "variavel": params.index,
            "label": [format_var_label(c) for c in params.index],
            "coef": params.to_numpy(dtype="float64"),
            "se_hc1": bse.to_numpy(dtype="float64"),
            "z": zvals.to_numpy(dtype="float64"),
            "p_valor": pvals.to_numpy(dtype="float64"),
            "odds_ratio": np.exp(params.to_numpy(dtype="float64")),
            "or_ic_inf": np.exp(params.to_numpy(dtype="float64") - 1.96 * bse.to_numpy(dtype="float64")),
            "or_ic_sup": np.exp(params.to_numpy(dtype="float64") + 1.96 * bse.to_numpy(dtype="float64")),
        }
    )

    if include_ame and X is not None and weights is not None:
        ame_df = compute_discrete_ame(res, X, weights)
        df_out = df_out.merge(
            ame_df[["variavel", "ame_pp", "ame_se_pp", "ame_ic_inf", "ame_ic_sup"]],
            on="variavel",
            how="left",
        )

    return df_out


def compute_subgroup_predictions(
    res: Any,
    df_sub: pd.DataFrame,
    X_builder_fn: Callable[[pd.DataFrame], tuple[pd.DataFrame, pd.Series, np.ndarray]] = build_design_matrix_interaction,
    weights: np.ndarray | None = None,
    dimensions: list[str] | None = None,
) -> pd.DataFrame:
    """Calcula probabilidades preditas ponderadas (ensino_superior=0 vs. 1) e Δ em p.p. por subgrupo."""
    if dimensions is None:
        dimensions = ["genero", "raca", "iniciativa", "formalidade", "setor"]

    w = df_sub["peso"].to_numpy(dtype="float64") if weights is None else np.asarray(weights, dtype="float64")
    w_norm = w / w.sum()

    model_cols = list(res.model.exog_names)
    beta = np.asarray(res.params, dtype="float64")
    cov = np.asarray(res.cov_params(), dtype="float64")

    rows: list[dict[str, Any]] = []
    for dim in dimensions:
        cats = (
            list(df_sub[dim].cat.categories)
            if hasattr(df_sub[dim], "cat")
            else sorted(df_sub[dim].dropna().unique())
        )
        for cat in cats:
            df0 = df_sub.copy()
            df0[dim] = pd.Categorical([cat] * len(df0), categories=cats) if hasattr(df_sub[dim], "cat") else cat
            df0["ensino_superior"] = 0
            X0, _, _ = X_builder_fn(df0)
            X0_mat = X0.reindex(columns=model_cols, fill_value=0.0).to_numpy(dtype="float64")

            df1 = df0.copy()
            df1["ensino_superior"] = 1
            X1, _, _ = X_builder_fn(df1)
            X1_mat = X1.reindex(columns=model_cols, fill_value=0.0).to_numpy(dtype="float64")

            p0 = _logistic_cdf(X0_mat @ beta)
            p1 = _logistic_cdf(X1_mat @ beta)

            prob0 = float(np.sum(w_norm * p0))
            prob1 = float(np.sum(w_norm * p1))
            diff = prob1 - prob0

            g0 = (w_norm * p0 * (1.0 - p0)) @ X0_mat
            g1 = (w_norm * p1 * (1.0 - p1)) @ X1_mat
            g_diff = g1 - g0

            se0 = float(np.sqrt(max(float(g0 @ cov @ g0), 0.0)))
            se1 = float(np.sqrt(max(float(g1 @ cov @ g1), 0.0)))
            se_diff = float(np.sqrt(max(float(g_diff @ cov @ g_diff), 0.0)))

            rows.append(
                {
                    "dimensao": dim,
                    "categoria": str(cat),
                    "prob_sem_superior_pct": prob0 * 100.0,
                    "prob_sem_sup_se_pp": se0 * 100.0,
                    "prob_com_superior_pct": prob1 * 100.0,
                    "prob_com_sup_se_pp": se1 * 100.0,
                    "ganho_superior_pp": diff * 100.0,
                    "ganho_ic_inf_pp": (diff - 1.96 * se_diff) * 100.0,
                    "ganho_ic_sup_pp": (diff + 1.96 * se_diff) * 100.0,
                    "razao_chances_marginal": prob1 / prob0 if prob0 > 0 else np.nan,
                }
            )

    return pd.DataFrame(rows)


def compute_diagnostics(
    res: Any,
    y: pd.Series | np.ndarray,
    X: pd.DataFrame,
    weights: np.ndarray,
) -> dict[str, float | int]:
    """Calcula métricas de ajuste e discriminação ponderadas para o modelo Logit."""
    y_arr = np.asarray(y, dtype="float64")
    w = np.asarray(weights, dtype="float64")
    probs = np.asarray(res.predict(X.astype("float64")), dtype="float64")

    return {
        "n_obs": int(len(y_arr)),
        "pop_ponderada_milhoes": float(w.sum() / 1e6),
        "taxa_remoto_obs_pct": float(np.average(y_arr, weights=w) * 100.0),
        "taxa_remoto_pred_pct": float(np.average(probs, weights=w) * 100.0),
        "log_lik": float(res.llf),
        "log_lik_null": float(res.llnull),
        "mcfadden_pseudo_r2": float(1.0 - res.llf / res.llnull),
        "roc_auc_ponderado": float(roc_auc_score(y_arr, probs, sample_weight=w)),
        "brier_score_ponderado": float(brier_score_loss(y_arr, probs, sample_weight=w)),
        "aic": float(res.aic),
    }


def compute_weighted_roc(
    y: pd.Series | np.ndarray,
    y_prob: np.ndarray,
    weights: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Retorna FPR, TPR e AUC ponderados pelos pesos amostrais."""
    y_arr = np.asarray(y, dtype="float64")
    w = np.asarray(weights, dtype="float64")
    fpr, tpr, _ = roc_curve(y_arr, y_prob, sample_weight=w)
    auc_val = float(roc_auc_score(y_arr, y_prob, sample_weight=w))
    return fpr, tpr, auc_val


def compute_calibration_bins(
    y_true: pd.Series | np.ndarray,
    y_prob: np.ndarray,
    weights: np.ndarray,
    n_bins: int = 10,
) -> pd.DataFrame:
    """Agrupa predições em quantis ponderados para avaliar calibração observada vs. predita."""
    df_cal = pd.DataFrame(
        {
            "y": np.asarray(y_true, dtype="float64"),
            "prob": np.asarray(y_prob, dtype="float64"),
            "w": np.asarray(weights, dtype="float64"),
        }
    )
    df_cal["decil"] = pd.qcut(df_cal["prob"], q=n_bins, labels=False, duplicates="drop") + 1
    df_cal["wy"] = df_cal["w"] * df_cal["y"]
    df_cal["wp"] = df_cal["w"] * df_cal["prob"]

    grouped = df_cal.groupby("decil", observed=True)[["w", "wy", "wp"]].sum().reset_index()
    grouped["taxa_observada_pct"] = grouped["wy"] / grouped["w"] * 100.0
    grouped["prob_predita_pct"] = grouped["wp"] / grouped["w"] * 100.0
    grouped["peso_total_mil"] = grouped["w"] / 1e3
    return grouped[["decil", "peso_total_mil", "prob_predita_pct", "taxa_observada_pct"]]


def compute_vif(X: pd.DataFrame, max_rows: int | None = 50000, random_state: int = 42) -> pd.DataFrame:
    """Calcula o Fator de Inflação da Variância (VIF) das covariáveis (excluindo a constante)."""
    X_cov = X.drop(columns=["const"], errors="ignore").astype("float64")
    if max_rows is not None and len(X_cov) > max_rows:
        X_cov = X_cov.sample(n=max_rows, random_state=random_state)

    X_with_const = sm.add_constant(X_cov, has_constant="add")
    arr = X_with_const.to_numpy(dtype="float64")
    cols = list(X_with_const.columns)

    rows = []
    for idx, col in enumerate(cols):
        if col == "const":
            continue
        rows.append(
            {
                "variavel": col,
                "label": format_var_label(col),
                "vif": float(variance_inflation_factor(arr, idx)),
            }
        )
    return pd.DataFrame(rows).sort_values("vif", ascending=False, ignore_index=True)
