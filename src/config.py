"""Configurações centrais, caminhos, mapeamentos geográficos e paleta visual."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_PATH = DATA_DIR / "raw" / "pnad_covid_2020.parquet"
PROCESSED_DATA_PATH = DATA_DIR / "processed" / "pnad_covid_analitico.parquet"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

RAW_COLUMNS = [
    "Ano",
    "V1013",
    "V1012",
    "UF",
    "V1032",
    "A002",
    "A003",
    "A004",
    "A005",
    "C001",
    "C002",
    "C003",
    "C007",
    "C007B",
    "C007D",
    "C012",
    "C013",
    "C014",
]

UF_TO_SIGLA: dict[str, str] = {
    "Rondônia": "RO",
    "Acre": "AC",
    "Amazonas": "AM",
    "Roraima": "RR",
    "Pará": "PA",
    "Amapá": "AP",
    "Tocantins": "TO",
    "Maranhão": "MA",
    "Piauí": "PI",
    "Ceará": "CE",
    "Rio Grande do Norte": "RN",
    "Paraíba": "PB",
    "Pernambuco": "PE",
    "Alagoas": "AL",
    "Sergipe": "SE",
    "Bahia": "BA",
    "Minas Gerais": "MG",
    "Espírito Santo": "ES",
    "Rio de Janeiro": "RJ",
    "São Paulo": "SP",
    "Paraná": "PR",
    "Santa Catarina": "SC",
    "Rio Grande do Sul": "RS",
    "Mato Grosso do Sul": "MS",
    "Mato Grosso": "MT",
    "Goiás": "GO",
    "Distrito Federal": "DF",
}

UF_TO_REGIAO: dict[str, str] = {
    "Rondônia": "Norte",
    "Acre": "Norte",
    "Amazonas": "Norte",
    "Roraima": "Norte",
    "Pará": "Norte",
    "Amapá": "Norte",
    "Tocantins": "Norte",
    "Maranhão": "Nordeste",
    "Piauí": "Nordeste",
    "Ceará": "Nordeste",
    "Rio Grande do Norte": "Nordeste",
    "Paraíba": "Nordeste",
    "Pernambuco": "Nordeste",
    "Alagoas": "Nordeste",
    "Sergipe": "Nordeste",
    "Bahia": "Nordeste",
    "Minas Gerais": "Sudeste",
    "Espírito Santo": "Sudeste",
    "Rio de Janeiro": "Sudeste",
    "São Paulo": "Sudeste",
    "Paraná": "Sul",
    "Santa Catarina": "Sul",
    "Rio Grande do Sul": "Sul",
    "Mato Grosso do Sul": "Centro-Oeste",
    "Mato Grosso": "Centro-Oeste",
    "Goiás": "Centro-Oeste",
    "Distrito Federal": "Centro-Oeste",
}

MES_LABELS: dict[int, str] = {
    5: "mai/2020",
    6: "jun/2020",
    7: "jul/2020",
    8: "ago/2020",
    9: "set/2020",
    10: "out/2020",
    11: "nov/2020",
}

MES_ORDER: list[str] = [MES_LABELS[m] for m in sorted(MES_LABELS)]

ESCOLARIDADE_ORDER: list[str] = [
    "Sem instrução",
    "Fundamental incompleto",
    "Fundamental completo",
    "Médio incompleto",
    "Médio completo",
    "Superior incompleto",
    "Superior completo",
    "Pós-graduação, mestrado ou doutorado",
]

FAIXA_ETARIA_DESC_ORDER: list[str] = [
    "14-19",
    "20-29",
    "30-39",
    "40-49",
    "50-59",
    "60-69",
    "70-79",
    "80+",
]

FAIXA_ETARIA_MODELO_ORDER: list[str] = [
    "14-24",
    "25-34",
    "35-44",
    "45-54",
    "55-64",
    "65+",
]

REGIAO_ORDER: list[str] = [
    "Sudeste",
    "Nordeste",
    "Sul",
    "Centro-Oeste",
    "Norte",
]

GENERO_ORDER: list[str] = ["Homem", "Mulher"]

RACA_ORDER: list[str] = ["Branca e Amarela", "Pretos, Pardos e Indígenas"]

SETOR_ORDER: list[str] = [
    "Serviços",
    "Indústria",
    "Comércio",
    "Setor Público",
    "Agricultura",
    "Outros",
]

INICIATIVA_ORDER: list[str] = ["Privado", "Público", "Outros"]

FORMALIDADE_ORDER: list[str] = ["Formal", "Informal"]

SITUACAO_OCUPACAO_ORDER: list[str] = [
    "Ocupado em trabalho remoto",
    "Ocupado presencial (não afastado)",
    "Afastado por distanciamento social",
    "Afastado por outros motivos",
    "Fora da ocupação",
]

PALETTE: dict[str, str] = {
    "navy": "#1B365D",
    "teal": "#2A9D8F",
    "coral": "#E76F51",
    "amber": "#E9C46A",
    "slate": "#5C677D",
    "blue_light": "#6BAED6",
    "green": "#2E7D32",
    "light_gray": "#E5E9F0",
    "grid_gray": "#DCE1E8",
    "dark_text": "#1F2933",
    "muted_text": "#52606D",
    "bg": "#FFFFFF",
}

DATA_DICTIONARY: list[dict[str, str]] = [
    {"variavel": "ano", "origem": "Ano (IBGE)", "tipo": "int16", "descricao": "Ano de referência da pesquisa (2020)."},
    {"variavel": "mes", "origem": "V1013 (IBGE)", "tipo": "int8", "descricao": "Mês da pesquisa (5 = maio a 11 = novembro de 2020)."},
    {"variavel": "mes_label", "origem": "Derivada (V1013)", "tipo": "category", "descricao": "Rótulo cronológico mês/ano (mai/2020 a nov/2020)."},
    {"variavel": "semana", "origem": "V1012 (IBGE)", "tipo": "int8", "descricao": "Semana no mês (1 a 4)."},
    {"variavel": "uf", "origem": "UF (IBGE)", "tipo": "category", "descricao": "Unidade da Federação de residência."},
    {"variavel": "uf_sigla", "origem": "Derivada (UF)", "tipo": "category", "descricao": "Sigla oficial da UF (27 UFs)."},
    {"variavel": "regiao", "origem": "Derivada (UF)", "tipo": "category", "descricao": "Macrorregião geográfica (Sudeste, Nordeste, Sul, Centro-Oeste, Norte)."},
    {"variavel": "peso", "origem": "V1032 (IBGE)", "tipo": "float64", "descricao": "Peso amostral calibrado para expansão populacional."},
    {"variavel": "idade", "origem": "A002 (IBGE)", "tipo": "int16", "descricao": "Idade do morador em anos."},
    {"variavel": "faixa_etaria_desc", "origem": "Derivada (A002)", "tipo": "category", "descricao": "Faixa etária descritiva (14-19 até 80+)."},
    {"variavel": "faixa_etaria_modelo", "origem": "Derivada (A002)", "tipo": "category", "descricao": "Faixa etária de controle econométrico (14-24 até 65+)."},
    {"variavel": "genero", "origem": "A003 (IBGE)", "tipo": "category", "descricao": "Sexo/gênero declarado (Homem, Mulher)."},
    {"variavel": "raca", "origem": "Derivada (A004)", "tipo": "category", "descricao": "Cor ou raça agrupada (Branca e Amarela vs. Pretos, Pardos e Indígenas)."},
    {"variavel": "escolaridade", "origem": "Derivada (A005)", "tipo": "category", "descricao": "Nível de escolaridade em 8 categorias ordenadas."},
    {"variavel": "ensino_superior", "origem": "Derivada (A005)", "tipo": "int8", "descricao": "Indicadora (1 = Superior completo ou Pós-graduação; 0 = demais)."},
    {"variavel": "ocupado", "origem": "Derivada (C001, C002)", "tipo": "bool", "descricao": "Pessoa ocupada (trabalhou ou estava temporariamente afastada)."},
    {"variavel": "ocupado_nao_afastado", "origem": "Derivada (C001)", "tipo": "bool", "descricao": "Pessoa ocupada que trabalhou na semana de referência (C001 == Sim)."},
    {"variavel": "afastado", "origem": "Derivada (C002)", "tipo": "bool", "descricao": "Pessoa ocupada afastada do trabalho na semana de referência."},
    {"variavel": "trabalho_remoto", "origem": "Derivada (C001, C013)", "tipo": "int8", "descricao": "Variável dependente (1 = exerceu trabalho remoto na semana de referência)."},
    {"variavel": "situacao_ocupacao", "origem": "Derivada (C001-C003, C013)", "tipo": "category", "descricao": "Situação detalhada (remoto, presencial, afastado por distanciamento, outros, fora)."},
    {"variavel": "iniciativa", "origem": "Derivada (C007)", "tipo": "category", "descricao": "Natureza da ocupação (Privado, Público, Outros)."},
    {"variavel": "formalidade", "origem": "Derivada (C007, C007B, C014)", "tipo": "category", "descricao": "Vínculo formal (carteira, estatutário, militar ou contribuição ao INSS) vs. Informal."},
    {"variavel": "setor", "origem": "Derivada (C007, C007D)", "tipo": "category", "descricao": "Setor de atividade (Serviços, Indústria, Comércio, Setor Público, Agricultura, Outros)."},
    {"variavel": "amostra_modelo", "origem": "Derivada", "tipo": "bool", "descricao": "Elegibilidade para modelagem Logit (ocupados não afastados no mercado com covariáveis completas)."},
]
