import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import locale
from datetime import datetime
from streamlit_option_menu import option_menu

# --- Constantes de Estilo ---
PRIMARY_COLOR = "#6c2bd9"  # Cor roxa principal da marca
FONT_FAMILY = "'system-ui', 'Segoe UI', Roboto, Helvetica, Arial, sans-serif"

# --- Configuração da Página ---
st.set_page_config(
    page_title="ClickPlus Dashboard",
    page_icon="📊",
    layout="wide"
)

# --- Carregamento dos ícones (Bootstrap Icons + Material Symbols) ---
# Observação: se sua rede bloquear CDNs, a fonte não será carregada — veja o checklist abaixo.
st.markdown("""
    <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap-icons@1.13.1/font/bootstrap-icons.css">
    <link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Material+Symbols+Outlined:opsz,wght,FILL,GRAD@48,400,0,0" />
""", unsafe_allow_html=True)

# --- Estilo Customizado (ajustado para NÃO sobrescrever ícones) ---
st.markdown(f"""
    <style>
    /* Aplique a fonte do sistema apenas globalmente (sem forçar em every .st- / .css- element) */
    html, body {{
        font-family: {FONT_FAMILY};
    }}

    /* Títulos */
    h1, h2, h3, h4, h5, h6 {{
        font-family: {FONT_FAMILY};
    }}

    /* Estilo para as métricas (cards) */
    .stMetric {{
        border-radius: 10px;
        padding: 15px;
        background-color: {PRIMARY_COLOR};
        height: 100%;
    }}
    .stMetric [data-testid="stMetricValue"], .stMetric [data-testid="stMetricLabel"] {{
        color: white;
    }}

    /* Cor dos filtros (multiselect) */
    .stMultiSelect div[data-baseweb="tag"] {{
        background-color: {PRIMARY_COLOR} !important;
        border-radius: 5px;
    }}

    /* Slider */
    div[data-baseweb="slider"] > div:nth-child(2) > div {{
        background-color: {PRIMARY_COLOR} !important;
    }}
    div[data-baseweb="slider"] > div:nth-child(3) {{
        background-color: {PRIMARY_COLOR} !important;
    }}

    /* Barra de progresso no DataFrame */
    div[data-testid="stDataFrameBar"] > div {{
        background-color: {PRIMARY_COLOR} !important;
    }}

    /* Métricas dentro do expander */
    [data-testid="stExpander"] .stMetric {{
        background-color: #262730;
        padding: 10px;
    }}
    [data-testid="stExpander"] .stMetric [data-testid="stMetricLabel"] {{
        font-size: 0.9rem !important;
    }}
    [data-testid="stExpander"] .stMetric [data-testid="stMetricValue"] {{
        font-size: 1.2rem !important;
    }}

    /* ===========================
       EXCEÇÕES PARA FONTES DE ÍCONES
       Não deixe que a regra global sobrescreva as families de ícones.
       =========================== */
    /* Material Symbols (Google) */
    .material-symbols-outlined, .material-symbols-sharp, .material-symbols-rounded {{
        font-family: 'Material Symbols Outlined', 'Material Symbols Sharp', 'Material Symbols Rounded' !important;
        font-variation-settings: 'FILL' 0, 'wght' 400, 'GRAD' 0;
        speak: none;
    }}

    /* Bootstrap Icons */
    .bi, [class^="bi-"], [class*=" bi-"] {{
        font-family: 'bootstrap-icons' !important;
        speak: none;
    }}

    </style>
""", unsafe_allow_html=True)

# --- Funções de Apoio ---
@st.cache_data
def load_data():
    try:
        DATA_DIR = './data/redis/'
        df_customers = pd.read_parquet(f'{DATA_DIR}customers.parquet')
        df_sales = pd.read_parquet(f'{DATA_DIR}sales.parquet')
        if 'nome_cliente' in df_sales.columns:
            df_sales.drop(columns=['nome_cliente'], inplace=True)
        df_full = pd.merge(df_sales, df_customers, on='id_cliente', how='left')
        return df_full
    except FileNotFoundError:
        st.error(f"Erro: Arquivos de dados não encontrados no diretório '{DATA_DIR}'. Verifique o caminho.")
        return pd.DataFrame()

def format_currency(value):
    try:
        return f"R$ {value:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")
    except (TypeError, ValueError):
        return "R$ 0,00"

# --- Páginas do Dashboard ---
def show_segmentation_page(df):
    st.title("Dashboard de Segmentação de Clientes")
    st.markdown("Analise os perfis de clientes para criar estratégias de marketing e vendas mais eficazes.")
    today = datetime.now()
    rfv_df = df.groupby('nome_cliente').agg(
        recencia=('data_venda', lambda date: (today - date.max()).days),
        frequencia=('data_venda', 'count'),
        valor_total=('valor_venda', 'sum'),
        segmento=('segmento', 'first')
    ).reset_index()

    st.sidebar.header("Filtros")
    segment_options = sorted(rfv_df['segmento'].unique())
    selected_segments = st.sidebar.multiselect("Filtrar por Grupo do Cliente", options=segment_options, default=segment_options)
    filtered_rfv_df = rfv_df[rfv_df['segmento'].isin(selected_segments)]

    total_clientes = filtered_rfv_df['nome_cliente'].nunique()
    receita_total = filtered_rfv_df['valor_total'].sum()
    ticket_medio = receita_total / total_clientes if total_clientes > 0 else 0

    col1, col2, col3 = st.columns(3)
    col1.metric("Clientes Ativos", f"{total_clientes}")
    col2.metric("Receita Total Gerada", format_currency(receita_total))
    col3.metric("Ticket Médio por Cliente", format_currency(ticket_medio))
    st.markdown("---")

    col1, col2 = st.columns([2, 1])
    with col1:
        st.subheader("Posicionamento de Clientes por RFM")
        fig = px.scatter(
            filtered_rfv_df, x='recencia', y='frequencia', size='valor_total', color='segmento',
            hover_name='nome_cliente', size_max=60,
            labels={'recencia': 'Recência (dias)', 'frequencia': 'Frequência (compras)', 'segmento': 'Grupo do Cliente'},
            color_discrete_sequence=px.colors.qualitative.Plotly
        )
        fig.update_traces(marker=dict(line=dict(width=1, color='DarkSlateGrey')))
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        st.subheader("Resumo por Grupo")
        segment_summary = filtered_rfv_df.groupby('segmento')['valor_total'].sum().sort_values(ascending=False)
        segment_summary_df = segment_summary.reset_index()
        segment_summary_df.columns = ["Segmentação", "Valor Total"]
        segment_summary_df['Valor Total'] = segment_summary_df['Valor Total'].map(format_currency)
        st.dataframe(segment_summary_df, use_container_width=True, hide_index=True)
        st.markdown("#### 💡 Insights Rápidos")
        try:
            top_segment = segment_summary.idxmax()
            st.info(f"O grupo **{top_segment}** é o mais valioso.")
        except ValueError:
            st.warning("Nenhum grupo selecionado.")

    st.subheader("Detalhes dos Clientes")
    st.dataframe(
        filtered_rfv_df, 
        column_config={
            "nome_cliente": "Cliente",
            "recencia": "Recência (dias)",
            "frequencia": "Frequência",
            "valor_total": st.column_config.NumberColumn("Valor Total (R$)", format="R$ %.2f"),
            "segmento": "Grupo do Cliente"
        },
        hide_index=True, use_container_width=True
    )

def show_opportunities_page(df):
    st.title("Radar de Oportunidades de Venda")
    st.markdown("Identifique proativamente quais clientes abordar e o que oferecer.")
    st.sidebar.header("Filtros de Prospecção")
    dias = ['30 dias', '7 dias']
    colunas_predicao = ['prob_compra_30d', 'prob_compra_7d']
    prediction_model = st.sidebar.selectbox("Probabilidade de compra em:", dias)
    index = dias.index(prediction_model)
    prob_column = colunas_predicao[index]
    prob_range = st.sidebar.slider(
        f"Prob. Compra ({dias[index]}) entre:",
        min_value=0, max_value=100, value=(75, 100), step=1
    )
    min_prob, max_prob = prob_range[0] / 100.0, prob_range[1] / 100.0
    df_customers_unique = df.drop_duplicates(subset=['id_cliente']).set_index('id_cliente')
    opportunities_df = df_customers_unique[
        (df_customers_unique[prob_column] >= min_prob) &
        (df_customers_unique[prob_column] <= max_prob)
    ].copy()
    opportunities_df[prob_column] = opportunities_df[prob_column] * 100
    opportunities_df_sorted = opportunities_df.sort_values(by=prob_column, ascending=False)
    st.header(f"{len(opportunities_df_sorted)} clientes encontrados!")
    st.subheader("Lista de Clientes Prioritários")
    st.dataframe(
        opportunities_df_sorted.reset_index(),
        column_config={
            "id_cliente": None,
            "nome_cliente": "Cliente",
            prob_column: st.column_config.ProgressColumn(
                "Prob. Próxima Compra (%)", format="%.0f%%", min_value=0, max_value=100,
                help="Probabilidade prevista pelo modelo XGBoost."
            ),
            "segmento": "Grupo do Cliente",
            "sugestao_prox_produto": "Sugestão de Próxima Viagem"
        }, 
        hide_index=True, use_container_width=True,
        column_order=['nome_cliente', prob_column, 'segmento', 'sugestao_prox_produto']
    )

    st.subheader("🔍 Análise Individual do Cliente")
    if not opportunities_df_sorted.empty:
        selected_customer_name = st.selectbox(
            "Selecione um cliente da lista para ver o perfil completo:",
            options=opportunities_df_sorted['nome_cliente']
        )
        with st.expander(f"Ver Perfil e Histórico de {selected_customer_name}", expanded=True):
            with st.container(): 
                customer_data = opportunities_df_sorted[opportunities_df_sorted['nome_cliente'] == selected_customer_name].iloc[0]
                customer_history = df[df['nome_cliente'] == selected_customer_name]
                ltv = customer_history['valor_venda'].sum()
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric(label="Grupo do Cliente", value=customer_data['segmento'])
                with col2:
                    st.metric(label=f"Prob. Compra ({dias[index]})", value=f"{customer_data[prob_column]:.0f}%")
                with col3:
                    st.metric(label="Sugestão de Viagem", value=customer_data['sugestao_prox_produto'])
                with col4:
                    st.metric(label="Lifetime Value", value=format_currency(ltv))
                st.write("Histórico de Compras:")
                st.dataframe(
                    customer_history, 
                    column_config={
                        "data_venda": "Data da Venda",
                        "produto": "Trecho da Viagem",
                        "valor_venda": st.column_config.NumberColumn("Valor (R$)", format="R$ %.2f")
                    },
                    hide_index=True, use_container_width=True,
                    column_order=['data_venda', 'produto', 'valor_venda']
                )
    else:
        st.warning("Nenhum cliente atende aos critérios de probabilidade selecionados.")

def show_executive_summary_page(df):
    st.title("Resumo Executivo ClickPlus")
    st.markdown(f"Relatório gerado em: **{datetime.now().strftime('%d/%m/%Y %H:%M')}**")
    prob_column = 'prob_compra_30d'
    df_customers_unique = df.drop_duplicates(subset=['id_cliente'])
    receita_preditiva = (df_customers_unique['valor_venda'].mean() * df_customers_unique[prob_column]).sum()
    clientes_em_risco = df_customers_unique[df_customers_unique['segmento'] == 'Em Risco']['nome_cliente'].nunique()
    segment_sales = df.groupby('segmento')['valor_venda'].sum()
    top_segment = segment_sales.idxmax()
    col1, col2, col3 = st.columns(3)
    col1.metric("Receita Preditiva (Próx. 30 dias)", format_currency(receita_preditiva))
    col2.metric("Clientes em Risco de Churn", f"{clientes_em_risco}")
    col3.metric("Grupo Mais Valioso", top_segment, help=f"Receita: {format_currency(segment_sales.max())}")
    st.markdown("---")
    st.subheader("Análise Estratégica")
    st.markdown(f"""
        - **Visão Geral:** Receita estimada de **{format_currency(receita_preditiva)}**.
        - **Foco Principal:** O grupo **'{top_segment}'** é o motor de crescimento.
        - **Ponto de Atenção:** Identificamos **{clientes_em_risco} clientes** em risco de churn.
        """)
    with st.expander("Ver detalhamento da Receita por Grupo de Cliente"):
        st.bar_chart(segment_sales, color=PRIMARY_COLOR)

# --- Função Principal ---
def main():

    
    """
    Função principal que organiza a aplicação Streamlit.
    """
    #locale.setlocale(locale.LC_ALL, 'pt_BR.UTF-8')
    # Gera os dados uma única vez
    #df = generate_fake_data()


    df = load_data()

    if not df.empty:
        with st.sidebar:
            st.image("logo.png", use_container_width=True)
            st.markdown("<br>", unsafe_allow_html=True)
            page_selection = option_menu(
                menu_title=None,
                options=["Segmentação", "Oportunidades", "Resumo Executivo"],
                icons=["people", "bullseye", "bar-chart-line"],
                menu_icon="bus-front", default_index=0,
                orientation="vertical",
                styles={
                    "container": {"padding": "0!important", "background-color": "#1c1c2e"},
                    "icon": {"color": "white", "font-size": "20px"},
                    "nav-link": {
                        "font-size": "16px",
                        "text-align": "left",
                        "margin":"8px",
                        "--hover-color": "#3a3b4a",
                        "font-family": FONT_FAMILY
                    },
                    "nav-link-selected": {"background-color": PRIMARY_COLOR},
                }
            )

        if page_selection == "Segmentação":
            show_segmentation_page(df)
        elif page_selection == "Oportunidades":
            show_opportunities_page(df)
        elif page_selection == "Resumo Executivo":
            show_executive_summary_page(df)

if __name__ == "__main__":
    main()
