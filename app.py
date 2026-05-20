# =============================================================================
# CALCULADORA DE VaR — MESAS DE TRADING
# Projeto Final: Gestão de Risco e Derivativos
# =============================================================================
# COMO RODAR:
#   1. Instale as dependências: pip install -r requirements.txt
#   2. Execute: streamlit run app.py
# =============================================================================

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from scipy.stats import norm
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import warnings

warnings.filterwarnings("ignore")

# =============================================================================
# CONFIGURAÇÃO DA PÁGINA (deve ser a primeira chamada do Streamlit)
# =============================================================================
st.set_page_config(
    page_title="VaR Calculator | Mesas de Trading",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =============================================================================
# INICIALIZAÇÃO DO SESSION STATE
# O session_state é como a "memória" da aplicação: guarda dados entre páginas.
# =============================================================================
if "mesas" not in st.session_state:
    # Mesas padrão conforme o enunciado
    st.session_state["mesas"] = {
        "Ações Brasil":     {"limite": 600_000},
        "Opções":           {"limite": 300_000},
        "Long & Short":     {"limite": 400_000},
        "Volatilidade":     {"limite": 500_000},
        "Mesa Proprietária":{"limite": 350_000},
    }

if "posicoes" not in st.session_state:
    st.session_state["posicoes"] = None  # DataFrame de posições (carregado pelo usuário)

if "params" not in st.session_state:
    st.session_state["params"] = {
        "confianca":   0.95,
        "horizonte":   1,
        "metodologia": "Histórico",
    }

if "var_results" not in st.session_state:
    st.session_state["var_results"] = None  # Resultados do cálculo de VaR


# =============================================================================
# FUNÇÕES DE SUPORTE
# =============================================================================

def status_limite(utilizacao: float) -> tuple[str, str]:
    """
    Retorna o emoji/texto de status e a cor conforme a tabela do enunciado.
    - Até 70%:       Verde  (risco confortável)
    - 70% a 100%:    Amarelo (atenção)
    - Acima de 100%: Vermelho (excesso de limite)
    """
    if utilizacao <= 0.70:
        return "🟢 Verde", "#2ecc71"
    elif utilizacao <= 1.00:
        return "🟡 Amarelo", "#f39c12"
    else:
        return "🔴 Vermelho", "#e74c3c"


def black_scholes(S: float, K: float, T: float, r: float,
                  sigma: float, tipo: str) -> float:
    """
    Preço de uma opção europeia pelo modelo Black-Scholes.

    Parâmetros:
        S     : Preço atual do ativo subjacente
        K     : Strike (preço de exercício)
        T     : Tempo até o vencimento em anos (ex.: 30/252)
        r     : Taxa livre de risco (ex.: 0.10 para 10% a.a.)
        sigma : Volatilidade anual do ativo
        tipo  : "Call" ou "Put"
    """
    if T <= 0:
        return max(0.0, S - K) if tipo == "Call" else max(0.0, K - S)

    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    if tipo == "Call":
        return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    else:
        return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)


def delta_opcao(S: float, K: float, T: float, r: float,
                sigma: float, tipo: str) -> float:
    """
    Delta de uma opção europeia (derivada do preço em relação ao ativo).
    Usado na aproximação Delta para o VaR de opções.
    """
    if T <= 0:
        return 1.0 if tipo == "Call" else -1.0

    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    return norm.cdf(d1) if tipo == "Call" else norm.cdf(d1) - 1.0


def var_historico(retornos: np.ndarray, valor: float,
                  confianca: float, horizonte: int) -> float:
    """
    VaR Histórico: usa a distribuição empírica dos retornos passados.
    Fórmula: VaR = -Percentil_α(Rp) × Valor da Carteira × √horizonte
    """
    alpha = 1.0 - confianca
    var_1d = -np.percentile(retornos, alpha * 100) * valor
    return max(0.0, var_1d * np.sqrt(horizonte))


def var_parametrico(retornos: np.ndarray, valor: float,
                    confianca: float, horizonte: int) -> float:
    """
    VaR Paramétrico (Normal): assume distribuição normal dos retornos.
    Fórmula: VaR = z_α × σ × Valor × √horizonte
    """
    z = norm.ppf(confianca)
    sigma = np.std(retornos, ddof=1)
    return max(0.0, z * sigma * valor * np.sqrt(horizonte))


def var_montecarlo(retornos: np.ndarray, valor: float,
                   confianca: float, horizonte: int,
                   n_sim: int = 10_000) -> float:
    """
    VaR por Simulação de Monte Carlo.
    Simula n_sim trajetórias de preço usando a dinâmica GBM (Movimento Browniano):
        ST = S0 × exp[(μ - ½σ²)T + σ√T × Z]
    e extrai o percentil das perdas simuladas.
    """
    mu    = np.mean(retornos)
    sigma = np.std(retornos, ddof=1)
    Z = np.random.standard_normal(n_sim)
    # Relação entre preço final e inicial (ST/S0)
    ratio = np.exp((mu - 0.5 * sigma**2) * horizonte + sigma * np.sqrt(horizonte) * Z)
    perdas = (1.0 - ratio) * valor  # perda = queda no valor
    alpha  = 1.0 - confianca
    return max(0.0, np.percentile(perdas, (1 - alpha) * 100))


def buscar_retornos(ticker: str, periodo: str = "1y") -> np.ndarray:
    """
    Baixa retornos diários históricos do Yahoo Finance.
    Caso o ticker não seja encontrado, gera retornos simulados como fallback.
    """
    try:
        df = yf.download(ticker, period=periodo, progress=False, auto_adjust=True)
        if df.empty or len(df) < 30:
            raise ValueError("Dados insuficientes")
        retornos = df["Close"].pct_change().dropna().values
        return retornos.flatten()
    except Exception:
        # Fallback: retornos simulados com parâmetros típicos do mercado brasileiro
        np.random.seed(42)
        return np.random.normal(0.0004, 0.019, 252)


def calcular_var_mesa(posicoes_mesa: pd.DataFrame, metodologia: str,
                      confianca: float, horizonte: int) -> tuple:
    """
    Calcula o VaR total de uma mesa e retorna os detalhes por posição.

    Para AÇÕES: aplica diretamente a metodologia escolhida sobre os retornos.
    Para OPÇÕES: usa a aproximação Delta.
        VaR_opção ≈ |Δ| × z × σ_diária × Valor × √horizonte
    """
    var_total   = 0.0
    valor_total = 0.0
    detalhes    = []

    TAXA_LIVRE_RISCO = 0.105  # Selic aproximada

    for _, row in posicoes_mesa.iterrows():
        ativo          = row["Ativo"]
        tipo           = row["Tipo"]
        valor_posicao  = float(row["Valor da Posição"])
        preco          = float(row["Preço"])
        valor_total   += valor_posicao

        retornos = buscar_retornos(ativo)

        if tipo == "Ação":
            # ── Cálculo direto pela metodologia escolhida ──────────────────
            if metodologia == "Histórico":
                var_pos = var_historico(retornos, valor_posicao, confianca, horizonte)
            elif metodologia == "Paramétrico":
                var_pos = var_parametrico(retornos, valor_posicao, confianca, horizonte)
            else:
                var_pos = var_montecarlo(retornos, valor_posicao, confianca, horizonte)

        elif tipo in ("Opção Call", "Opção Put"):
            # ── Aproximação Delta para opções ──────────────────────────────
            # Strike e vencimento: usa colunas do CSV ou aplica defaults razoáveis
            K    = float(row["Strike"])   if pd.notna(row.get("Strike"))     else preco * 1.05
            dias = float(row["Vencimento"]) if pd.notna(row.get("Vencimento")) else 30.0
            T    = dias / 252.0           # converte dias úteis para anos
            vol_anual = np.std(retornos, ddof=1) * np.sqrt(252)

            tipo_bs = "Call" if tipo == "Opção Call" else "Put"
            delta   = delta_opcao(preco, K, T, TAXA_LIVRE_RISCO, vol_anual, tipo_bs)

            # VaR Delta: sensibilidade do preço da opção ao ativo subjacente
            vol_diaria = np.std(retornos, ddof=1)
            z          = norm.ppf(confianca)
            var_pos    = abs(delta) * z * vol_diaria * valor_posicao * np.sqrt(horizonte)
        else:
            var_pos = 0.0

        var_total += var_pos
        detalhes.append({
            "Ativo":             ativo,
            "Tipo":              tipo,
            "Valor da Posição":  valor_posicao,
            "VaR (R$)":          var_pos,
            "% VaR da Mesa":     0.0,  # calculado abaixo
        })

    # Calcula a contribuição percentual de cada posição no VaR da mesa
    for d in detalhes:
        d["% VaR da Mesa"] = (d["VaR (R$)"] / var_total * 100) if var_total > 0 else 0.0

    return var_total, valor_total, detalhes


# =============================================================================
# SIDEBAR — NAVEGAÇÃO
# =============================================================================
st.sidebar.markdown("## 📊 VaR Calculator")
st.sidebar.markdown("**Gestão de Risco | Mesas de Trading**")
st.sidebar.divider()

pagina = st.sidebar.radio(
    "Navegação",
    options=[
        "🏠 Início",
        "📋 Cadastro de Mesas",
        "📤 Upload de Posições",
        "⚙️ Parâmetros de Risco",
        "📊 Cálculo do VaR",
        "🚦 Monitoramento de Limites",
        "📈 Dashboard Executivo",
    ],
)

st.sidebar.divider()

# Exibe os parâmetros ativos na sidebar como referência rápida
p = st.session_state["params"]
st.sidebar.markdown(
    f"**Configuração atual**\n\n"
    f"- Metodologia: `{p['metodologia']}`\n"
    f"- Confiança: `{p['confianca']*100:.0f}%`\n"
    f"- Horizonte: `{p['horizonte']} dia(s)`\n"
    f"- Posições: `{'✅ carregadas' if st.session_state['posicoes'] is not None else '❌ não carregadas'}`\n"
    f"- VaR: `{'✅ calculado' if st.session_state['var_results'] else '❌ não calculado'}`"
)
st.sidebar.caption("Projeto Final — Gestão de Risco e Derivativos")


# =============================================================================
# PÁGINA 1 — INÍCIO
# =============================================================================
if pagina == "🏠 Início":
    st.title("📊 Calculadora de VaR para Mesas de Trading")

    st.markdown("""
    Bem-vindo! Esta aplicação simula o trabalho de uma **área de risco de mercado**
    que monitora múltiplas mesas de trading, verificando se cada mesa opera
    dentro do seu limite autorizado de **Value at Risk (VaR)**.
    """)

    col1, col2, col3 = st.columns(3)
    col1.info("**📋 Cadastrar Mesas**\nDefina as mesas de trading e seus limites de VaR aprovados pela diretoria.")
    col2.info("**📊 Calcular VaR**\nEscolha entre metodologia Histórica, Paramétrica ou Monte Carlo — para ações e opções.")
    col3.info("**🚦 Monitorar Limites**\nAcompanhe o consumo de limite com alertas automáticos 🟢🟡🔴.")

    st.divider()
    st.subheader("O que é VaR?")
    st.markdown("""
    O **Value at Risk (VaR)** responde à pergunta:

    > *"Qual é a perda máxima que esta carteira pode ter, com X% de probabilidade, em N dias?"*

    Por exemplo: *"Com 95% de confiança, a Mesa de Ações não perderá mais de R$ 450.000 em 1 dia."*
    """)

    st.subheader("Metodologias disponíveis")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("**📜 Histórico**")
        st.caption("Usa a distribuição empírica dos retornos reais do passado. Não assume nenhuma forma paramétrica.")
    with col2:
        st.markdown("**📐 Paramétrico (Normal)**")
        st.caption("Assume que os retornos seguem distribuição normal. Utiliza média e desvio-padrão históricos.")
    with col3:
        st.markdown("**🎲 Monte Carlo**")
        st.caption("Simula 10.000 cenários de preços futuros e calcula a perda no percentil escolhido.")

    st.divider()
    st.subheader("Classificação de risco por mesa")
    col1, col2, col3 = st.columns(3)
    col1.success("🟢 **Até 70%** do limite\n\nRisco confortável")
    col2.warning("🟡 **70% a 100%** do limite\n\nAtenção")
    col3.error("🔴 **Acima de 100%** do limite\n\nExcesso — ação necessária")

    st.divider()
    st.subheader("Fluxo de uso da aplicação")
    st.markdown("""
    1. **📋 Cadastro de Mesas** → Defina as mesas e seus limites
    2. **📤 Upload de Posições** → Carregue um Excel/CSV ou use dados de exemplo
    3. **⚙️ Parâmetros de Risco** → Configure confiança, horizonte e metodologia
    4. **📊 Cálculo do VaR** → Execute o modelo e veja os resultados por posição
    5. **🚦 Monitoramento** → Veja alertas por mesa e tabela consolidada
    6. **📈 Dashboard** → Painel executivo com gráficos e recomendações
    """)


# =============================================================================
# PÁGINA 2 — CADASTRO DE MESAS
# =============================================================================
elif pagina == "📋 Cadastro de Mesas":
    st.title("📋 Cadastro de Mesas de Trading")
    st.markdown("Defina as mesas e seus **limites de VaR** aprovados pela diretoria de risco.")

    # ── Adicionar nova mesa ──────────────────────────────────────────────────
    with st.expander("➕ Adicionar nova mesa", expanded=False):
        col1, col2 = st.columns(2)
        with col1:
            nova_mesa = st.text_input("Nome da mesa", placeholder="Ex: Mesa de Câmbio")
        with col2:
            novo_limite = st.number_input(
                "Limite de VaR (R$)",
                min_value=10_000, max_value=50_000_000,
                value=500_000, step=50_000,
            )
        if st.button("✅ Adicionar", type="primary"):
            if not nova_mesa:
                st.error("Digite um nome para a mesa.")
            elif nova_mesa in st.session_state["mesas"]:
                st.warning("Já existe uma mesa com esse nome.")
            else:
                st.session_state["mesas"][nova_mesa] = {"limite": novo_limite}
                st.success(f"Mesa **{nova_mesa}** criada com limite de **R$ {novo_limite:,.0f}**!")

    # ── Tabela editável ──────────────────────────────────────────────────────
    st.subheader("Mesas cadastradas")

    df_mesas = pd.DataFrame([
        {"Mesa": nome, "Limite de VaR (R$)": info["limite"]}
        for nome, info in st.session_state["mesas"].items()
    ])

    df_editado = st.data_editor(
        df_mesas,
        column_config={
            "Mesa": st.column_config.TextColumn("Mesa", disabled=True),
            "Limite de VaR (R$)": st.column_config.NumberColumn(
                "Limite de VaR (R$)", format="R$ %,.0f", min_value=0
            ),
        },
        use_container_width=True,
        hide_index=True,
    )

    if st.button("💾 Salvar limites editados"):
        for _, row in df_editado.iterrows():
            if row["Mesa"] in st.session_state["mesas"]:
                st.session_state["mesas"][row["Mesa"]]["limite"] = row["Limite de VaR (R$)"]
        st.success("Limites atualizados!")

    # ── Remover mesa ─────────────────────────────────────────────────────────
    st.divider()
    if st.session_state["mesas"]:
        mesa_del = st.selectbox("Remover mesa", list(st.session_state["mesas"].keys()))
        if st.button("🗑️ Remover mesa selecionada", type="secondary"):
            del st.session_state["mesas"][mesa_del]
            st.rerun()


# =============================================================================
# PÁGINA 3 — UPLOAD DE POSIÇÕES
# =============================================================================
elif pagina == "📤 Upload de Posições":
    st.title("📤 Upload de Posições")
    st.markdown(
        "Importe um arquivo **Excel ou CSV** com as posições das mesas, "
        "ou use os **dados de exemplo** do enunciado."
    )

    # ── Dados de exemplo ─────────────────────────────────────────────────────
    EXEMPLO = pd.DataFrame({
        "Mesa":    [
            "Ações Brasil", "Ações Brasil", "Ações Brasil",
            "Opções", "Opções",
            "Long & Short", "Long & Short",
            "Volatilidade", "Volatilidade",
            "Mesa Proprietária",
        ],
        "Ativo":   [
            "PETR4.SA", "VALE3.SA", "ITUB4.SA",
            "PETR4.SA", "VALE3.SA",
            "BBDC4.SA", "ITUB4.SA",
            "MGLU3.SA", "AMER3.SA",
            "WEGE3.SA",
        ],
        "Tipo":    [
            "Ação", "Ação", "Ação",
            "Opção Call", "Opção Put",
            "Ação", "Ação",
            "Ação", "Ação",
            "Ação",
        ],
        "Quantidade": [100_000, 80_000, 120_000,
                       50_000, 30_000,
                       60_000, 90_000,
                       200_000, 150_000,
                       75_000],
        "Preço":   [38.50, 68.20, 32.10,
                    38.50, 68.20,
                    16.80, 32.10,
                    8.50, 5.20,
                    45.30],
        "Valor da Posição": [
            3_850_000, 5_456_000, 3_852_000,
            1_925_000, 2_046_000,
            1_008_000, 2_889_000,
            1_700_000,   780_000,
            3_397_500,
        ],
        "Strike":     [None, None, None, 42.0, 65.0, None, None, None, None, None],
        "Vencimento": [None, None, None, 30.0, 30.0, None, None, None, None, None],
        "Limite de VaR": [
            600_000, 600_000, 600_000,
            300_000, 300_000,
            400_000, 400_000,
            500_000, 500_000,
            350_000,
        ],
    })

    tab_upload, tab_exemplo = st.tabs(["📁 Upload de arquivo", "🗂️ Dados de exemplo"])

    with tab_upload:
        arquivo = st.file_uploader(
            "Selecione um arquivo Excel (.xlsx) ou CSV (.csv)",
            type=["xlsx", "csv"],
        )
        if arquivo:
            try:
                df = pd.read_csv(arquivo) if arquivo.name.endswith(".csv") else pd.read_excel(arquivo)
                st.success(f"✅ {len(df)} posições carregadas.")
                st.dataframe(df, use_container_width=True)
                st.session_state["posicoes"] = df
            except Exception as e:
                st.error(f"Erro ao ler arquivo: {e}")

        st.divider()
        st.markdown("**Colunas obrigatórias no arquivo:**")
        st.dataframe(
            EXEMPLO[["Mesa", "Ativo", "Tipo", "Quantidade", "Preço", "Valor da Posição", "Limite de VaR"]].head(3),
            use_container_width=True,
        )
        st.caption(
            "Colunas opcionais para opções: **Strike** (preço de exercício) e "
            "**Vencimento** (dias úteis até o vencimento)."
        )

    with tab_exemplo:
        st.markdown("Clique abaixo para carregar os dados de exemplo com **5 mesas e 10 posições**.")

        if st.button("📥 Carregar dados de exemplo", type="primary"):
            st.session_state["posicoes"] = EXEMPLO
            st.success("✅ Dados de exemplo carregados!")

        if st.session_state["posicoes"] is not None:
            st.dataframe(st.session_state["posicoes"], use_container_width=True)


# =============================================================================
# PÁGINA 4 — PARÂMETROS DE RISCO
# =============================================================================
elif pagina == "⚙️ Parâmetros de Risco":
    st.title("⚙️ Parâmetros de Risco")
    st.markdown("Configure os parâmetros que serão usados em **todos** os cálculos de VaR.")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Nível de Confiança (α)")
        confianca = st.select_slider(
            "Escolha o nível de confiança:",
            options=[0.90, 0.95, 0.99],
            value=st.session_state["params"]["confianca"],
            format_func=lambda x: f"{x*100:.0f}%",
        )
        interpretacoes = {
            0.90: "📌 **90%**: 1 excesso esperado a cada **10 dias**. Menos conservador.",
            0.95: "📌 **95%**: Padrão de mercado. 1 excesso esperado a cada **20 dias**.",
            0.99: "📌 **99%**: Muito conservador. 1 excesso esperado a cada **100 dias**.",
        }
        st.info(interpretacoes[confianca])

    with col2:
        st.subheader("Horizonte de Tempo")
        horizonte = st.radio(
            "Horizonte para o cálculo:",
            options=[1, 5, 10, 21],
            format_func=lambda x: f"{x} dia{'s' if x > 1 else ''}",
            index=[1, 5, 10, 21].index(st.session_state["params"]["horizonte"]),
        )
        st.caption(
            f"O VaR de 1 dia será escalado pela **raiz do tempo**: √{horizonte} ≈ {np.sqrt(horizonte):.2f}. "
            f"Essa regra assume retornos independentes e identicamente distribuídos."
        )

    st.divider()
    st.subheader("Metodologia de VaR")

    metodologia = st.radio(
        "Escolha a metodologia:",
        options=["Histórico", "Paramétrico", "Monte Carlo"],
        horizontal=True,
        index=["Histórico", "Paramétrico", "Monte Carlo"].index(
            st.session_state["params"]["metodologia"]
        ),
    )

    descricoes = {
        "Histórico":   "Usa os retornos reais observados no passado. Não impõe distribuição. Sensível a choques históricos.",
        "Paramétrico": "Assume normalidade dos retornos (μ e σ históricos). Rápido, mas subestima caudas gordas.",
        "Monte Carlo": "Simula 10.000 cenários futuros de preços via GBM. Mais flexível e computacionalmente intenso.",
    }
    st.info(f"ℹ️ {descricoes[metodologia]}")

    st.divider()
    if st.button("💾 Salvar parâmetros", type="primary"):
        st.session_state["params"] = {
            "confianca":   confianca,
            "horizonte":   horizonte,
            "metodologia": metodologia,
        }
        st.success(
            f"✅ Parâmetros salvos: **{metodologia}** | "
            f"**{confianca*100:.0f}%** de confiança | "
            f"**{horizonte} dia(s)**"
        )


# =============================================================================
# PÁGINA 5 — CÁLCULO DO VaR
# =============================================================================
elif pagina == "📊 Cálculo do VaR":
    st.title("📊 Cálculo do VaR")

    if st.session_state["posicoes"] is None:
        st.warning("⚠️ Nenhuma posição carregada. Vá para **📤 Upload de Posições** primeiro.")
        st.stop()

    p = st.session_state["params"]
    st.markdown(
        f"**Configuração:** {p['metodologia']} | {p['confianca']*100:.0f}% confiança | {p['horizonte']} dia(s)"
    )

    if st.button("🚀 Calcular VaR de todas as mesas", type="primary", use_container_width=True):
        df_pos  = st.session_state["posicoes"]
        mesas   = st.session_state["mesas"]
        results = {}

        barra   = st.progress(0, text="Iniciando cálculo...")
        mesas_encontradas = [m for m in df_pos["Mesa"].unique() if m in mesas]
        total   = len(mesas_encontradas)

        for i, mesa in enumerate(mesas_encontradas):
            barra.progress((i + 1) / total, text=f"Calculando: {mesa}...")
            posicoes_mesa = df_pos[df_pos["Mesa"] == mesa].copy()
            limite        = mesas[mesa]["limite"]

            try:
                var, valor_total, detalhes = calcular_var_mesa(
                    posicoes_mesa, p["metodologia"], p["confianca"], p["horizonte"]
                )
                utilizacao = var / limite if limite > 0 else 0.0
                status, _  = status_limite(utilizacao)

                results[mesa] = {
                    "VaR (R$)":              var,
                    "Valor da Carteira (R$)": valor_total,
                    "Limite (R$)":           limite,
                    "Utilização":            utilizacao,
                    "Status":                status,
                    "Detalhes":              detalhes,
                }
            except Exception as e:
                st.error(f"Erro na mesa {mesa}: {e}")

        barra.empty()
        st.session_state["var_results"] = results
        st.success("✅ Cálculo concluído! Veja os resultados abaixo.")

    # ── Exibição dos resultados ───────────────────────────────────────────────
    if st.session_state["var_results"]:
        st.divider()
        st.subheader("Resultados por Mesa")

        for mesa, res in st.session_state["var_results"].items():
            utilizacao = res["Utilização"]
            status     = res["Status"]

            with st.expander(
                f"{status} | **{mesa}** — VaR: R$ {res['VaR (R$)']:,.0f} | "
                f"Limite: R$ {res['Limite (R$)']:,.0f} | Utilização: {utilizacao*100:.1f}%",
                expanded=True,
            ):
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("💼 Valor da Carteira",  f"R$ {res['Valor da Carteira (R$)']:,.0f}")
                col2.metric("📊 VaR Calculado",       f"R$ {res['VaR (R$)']:,.0f}")
                col3.metric("🎯 Limite de VaR",       f"R$ {res['Limite (R$)']:,.0f}")
                col4.metric("⚡ Utilização",           f"{utilizacao*100:.1f}%")

                # Detalhes por posição (formatado para exibição)
                df_det = pd.DataFrame(res["Detalhes"])
                df_det["Valor da Posição"] = df_det["Valor da Posição"].map("R$ {:,.0f}".format)
                df_det["VaR (R$)"]         = df_det["VaR (R$)"].map("R$ {:,.0f}".format)
                df_det["% VaR da Mesa"]    = df_det["% VaR da Mesa"].map("{:.1f}%".format)
                st.dataframe(df_det, use_container_width=True, hide_index=True)


# =============================================================================
# PÁGINA 6 — MONITORAMENTO DE LIMITES
# =============================================================================
elif pagina == "🚦 Monitoramento de Limites":
    st.title("🚦 Monitoramento de Limites")

    if not st.session_state["var_results"]:
        st.warning("⚠️ Execute o **📊 Cálculo do VaR** primeiro.")
        st.stop()

    results = st.session_state["var_results"]

    # ── Resumo rápido ────────────────────────────────────────────────────────
    n_verde    = sum(1 for r in results.values() if r["Utilização"] <= 0.70)
    n_amarelo  = sum(1 for r in results.values() if 0.70 < r["Utilização"] <= 1.00)
    n_vermelho = sum(1 for r in results.values() if r["Utilização"] > 1.00)

    col1, col2, col3 = st.columns(3)
    col1.metric("🟢 No Verde",    n_verde)
    col2.metric("🟡 Em Atenção",  n_amarelo)
    col3.metric("🔴 Em Excesso",  n_vermelho)

    if n_vermelho > 0:
        mesas_excesso = [m for m, r in results.items() if r["Utilização"] > 1.0]
        st.error(f"🚨 **ALERTA DE LIMITE:** {', '.join(mesas_excesso)} ultrapassaram o VaR máximo autorizado!")

    # ── Tabela consolidada ───────────────────────────────────────────────────
    st.divider()
    st.subheader("Tabela Consolidada — Todas as Mesas")

    linhas = [
        {
            "Mesa":              mesa,
            "Valor da Carteira": res["Valor da Carteira (R$)"],
            "VaR Calculado":     res["VaR (R$)"],
            "Limite de VaR":     res["Limite (R$)"],
            "Utilização (%)":    res["Utilização"] * 100,
            "Status":            res["Status"],
        }
        for mesa, res in results.items()
    ]
    df_tabela = pd.DataFrame(linhas).sort_values("Utilização (%)", ascending=False)

    def colorir_status(val):
        if "Vermelho" in str(val):
            return "background-color: #ffcccc"
        if "Amarelo" in str(val):
            return "background-color: #fff3cc"
        if "Verde" in str(val):
            return "background-color: #ccffcc"
        return ""

    st.dataframe(
        df_tabela.style
            .format({
                "Valor da Carteira": "R$ {:,.0f}",
                "VaR Calculado":     "R$ {:,.0f}",
                "Limite de VaR":     "R$ {:,.0f}",
                "Utilização (%)":    "{:.1f}%",
            })
            .applymap(colorir_status, subset=["Status"]),
        use_container_width=True,
        hide_index=True,
    )

    # ── Gráfico: VaR vs Limite ────────────────────────────────────────────────
    st.divider()
    st.subheader("VaR Calculado vs Limite por Mesa")

    nomes    = df_tabela["Mesa"].tolist()
    var_vals = df_tabela["VaR Calculado"].tolist()
    lim_vals = df_tabela["Limite de VaR"].tolist()
    util_vals= df_tabela["Utilização (%)"].tolist()

    cores = ["#2ecc71" if u <= 70 else "#f39c12" if u <= 100 else "#e74c3c" for u in util_vals]

    fig1 = go.Figure()
    fig1.add_trace(go.Bar(
        name="VaR Calculado",
        x=nomes, y=var_vals,
        marker_color=cores,
        text=[f"R$ {v:,.0f}" for v in var_vals],
        textposition="outside",
    ))
    fig1.add_trace(go.Scatter(
        name="Limite de VaR",
        x=nomes, y=lim_vals,
        mode="markers+lines",
        marker=dict(size=12, color="#2c3e50", symbol="diamond"),
        line=dict(dash="dash", color="#2c3e50", width=2),
    ))
    fig1.update_layout(
        yaxis_title="Valor (R$)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        height=420,
    )
    st.plotly_chart(fig1, use_container_width=True)

    # ── Gráfico: Utilização (%) ───────────────────────────────────────────────
    st.subheader("Utilização do Limite por Mesa (%)")

    fig2 = go.Figure(go.Bar(
        x=nomes, y=util_vals,
        marker_color=cores,
        text=[f"{u:.1f}%" for u in util_vals],
        textposition="outside",
    ))
    fig2.add_hline(y=70,  line_dash="dot", line_color="#f39c12", annotation_text="70% — Atenção")
    fig2.add_hline(y=100, line_dash="dot", line_color="#e74c3c", annotation_text="100% — Limite máximo")
    fig2.update_layout(
        yaxis_title="Utilização (%)",
        yaxis=dict(range=[0, max(max(util_vals) * 1.25, 130)]),
        height=400,
    )
    st.plotly_chart(fig2, use_container_width=True)

    # ── Ranking ───────────────────────────────────────────────────────────────
    st.subheader("🏆 Ranking por Consumo de Risco")

    emojis = ["🥇", "🥈", "🥉", "4️⃣", "5️⃣", "6️⃣", "7️⃣", "8️⃣"]
    for i, row in enumerate(df_tabela.itertuples()):
        emoji = emojis[min(i, len(emojis) - 1)]
        st.markdown(
            f"{emoji} **{row.Mesa}** — {row._5:.1f}% de utilização | {row.Status}"
        )


# =============================================================================
# PÁGINA 7 — DASHBOARD EXECUTIVO
# =============================================================================
elif pagina == "📈 Dashboard Executivo":
    st.title("📈 Dashboard Executivo de Risco")
    st.caption(f"Atualizado em: {datetime.now().strftime('%d/%m/%Y %H:%M')}")

    if not st.session_state["var_results"]:
        st.warning("⚠️ Execute o **📊 Cálculo do VaR** primeiro.")
        st.stop()

    results = st.session_state["var_results"]
    params  = st.session_state["params"]

    # ── KPIs globais ─────────────────────────────────────────────────────────
    st.subheader("📌 Visão Global")

    total_var      = sum(r["VaR (R$)"]              for r in results.values())
    total_carteira = sum(r["Valor da Carteira (R$)"] for r in results.values())
    total_limite   = sum(r["Limite (R$)"]            for r in results.values())
    util_global    = total_var / total_limite if total_limite > 0 else 0.0

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("💼 Valor Total",      f"R$ {total_carteira/1_000_000:.1f}M")
    col2.metric("📊 VaR Consolidado",  f"R$ {total_var/1_000:.0f}K")
    col3.metric("🎯 Limite Total",      f"R$ {total_limite/1_000:.0f}K")
    col4.metric("⚡ Utilização Global", f"{util_global*100:.1f}%")

    st.divider()

    # ── Ranking + Pizza ───────────────────────────────────────────────────────
    col_esq, col_dir = st.columns(2)

    with col_esq:
        st.subheader("🏆 Ranking por Consumo de Risco")
        ranking = sorted(results.items(), key=lambda x: x[1]["Utilização"], reverse=True)
        emojis  = ["🥇", "🥈", "🥉", "4️⃣", "5️⃣", "6️⃣", "7️⃣"]
        for i, (mesa, res) in enumerate(ranking):
            e = emojis[min(i, len(emojis) - 1)]
            st.markdown(
                f"{e} **{mesa}** → {res['Utilização']*100:.1f}% | {res['Status']}"
            )

    with col_dir:
        st.subheader("🥧 Distribuição do VaR")
        fig_pie = go.Figure(go.Pie(
            labels=list(results.keys()),
            values=[r["VaR (R$)"] for r in results.values()],
            hole=0.45,
            textinfo="label+percent",
        ))
        fig_pie.update_layout(height=300, margin=dict(t=10, b=10, l=10, r=10))
        st.plotly_chart(fig_pie, use_container_width=True)

    st.divider()

    # ── Análise de retornos de um ativo específico ────────────────────────────
    st.subheader("📉 Distribuição de Retornos e Evolução de Preço")

    if st.session_state["posicoes"] is not None:
        df_pos   = st.session_state["posicoes"]
        mesa_sel = st.selectbox("Selecione uma mesa para análise:", list(results.keys()))
        pos_sel  = df_pos[df_pos["Mesa"] == mesa_sel]

        if not pos_sel.empty:
            ativo_sel = pos_sel["Ativo"].iloc[0]

            with st.spinner(f"Carregando dados de {ativo_sel}..."):
                retornos_sel = buscar_retornos(ativo_sel)
                try:
                    dados_preco = yf.download(ativo_sel, period="1y", progress=False, auto_adjust=True)
                except Exception:
                    dados_preco = pd.DataFrame()

            col_h, col_p = st.columns(2)

            with col_h:
                # Histograma dos retornos com linha do VaR
                alpha   = 1 - params["confianca"]
                var_pct = np.percentile(retornos_sel, alpha * 100)

                fig_hist = go.Figure()
                fig_hist.add_trace(go.Histogram(
                    x=retornos_sel, nbinsx=60,
                    marker_color="#3498db", opacity=0.75, name="Retornos",
                ))
                fig_hist.add_vline(
                    x=var_pct, line_dash="dash", line_color="#e74c3c",
                    annotation_text=f"VaR {params['confianca']*100:.0f}%",
                    annotation_position="top right",
                )
                fig_hist.update_layout(
                    title=f"Distribuição de Retornos — {ativo_sel}",
                    xaxis_title="Retorno Diário",
                    yaxis_title="Frequência",
                    height=350,
                )
                st.plotly_chart(fig_hist, use_container_width=True)

            with col_p:
                if not dados_preco.empty:
                    close_vals = dados_preco["Close"].values.flatten()
                    fig_preco = go.Figure()
                    fig_preco.add_trace(go.Scatter(
                        x=dados_preco.index, y=close_vals,
                        mode="lines",
                        line=dict(color="#2c3e50", width=2),
                        fill="tozeroy",
                        fillcolor="rgba(52, 152, 219, 0.12)",
                        name="Preço",
                    ))
                    fig_preco.update_layout(
                        title=f"Evolução de Preço — {ativo_sel}",
                        xaxis_title="Data",
                        yaxis_title="Preço (R$)",
                        height=350,
                    )
                    st.plotly_chart(fig_preco, use_container_width=True)
                else:
                    st.info("Dados de preço não disponíveis para visualização.")

    st.divider()

    # ── Comparação entre metodologias ─────────────────────────────────────────
    st.subheader("⚖️ Comparação entre Metodologias (VaR por R$ 1.000.000 de exposição)")
    st.caption("Usando retornos de referência com μ = 0,04% a.d. e σ = 1,8% a.d.")

    np.random.seed(42)
    ret_ref   = np.random.normal(0.0004, 0.018, 252)
    val_ref   = 1_000_000
    comp_data = []

    for met in ["Histórico", "Paramétrico", "Monte Carlo"]:
        if met == "Histórico":
            v = var_historico(ret_ref, val_ref, params["confianca"], params["horizonte"])
        elif met == "Paramétrico":
            v = var_parametrico(ret_ref, val_ref, params["confianca"], params["horizonte"])
        else:
            v = var_montecarlo(ret_ref, val_ref, params["confianca"], params["horizonte"])
        comp_data.append({"Metodologia": met, "VaR (R$)": v})

    df_comp = pd.DataFrame(comp_data)
    fig_comp = px.bar(
        df_comp, x="Metodologia", y="VaR (R$)",
        color="Metodologia",
        color_discrete_sequence=["#3498db", "#2ecc71", "#e74c3c"],
        text_auto=".0f",
    )
    fig_comp.update_layout(showlegend=False, height=350)
    st.plotly_chart(fig_comp, use_container_width=True)

    st.divider()

    # ── Recomendações ─────────────────────────────────────────────────────────
    st.subheader("💡 Recomendações da Área de Risco")

    for mesa, res in sorted(results.items(), key=lambda x: x[1]["Utilização"], reverse=True):
        u = res["Utilização"]
        if u > 1.0:
            st.error(
                f"🔴 **{mesa}** ({u*100:.1f}%): Limite ultrapassado. "
                "Reduzir exposição imediatamente ou submeter para aprovação de limite emergencial."
            )
        elif u > 0.70:
            st.warning(
                f"🟡 **{mesa}** ({u*100:.1f}%): Zona de atenção. "
                "Monitoramento diário recomendado. Evitar novas posições que aumentem o risco."
            )
        else:
            st.success(
                f"🟢 **{mesa}** ({u*100:.1f}%): Risco confortável. "
                "Operando dentro dos parâmetros aprovados."
            )
