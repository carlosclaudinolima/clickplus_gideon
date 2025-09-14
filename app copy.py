import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import locale
from datetime import datetime

# --- Configuração da Página ---
st.set_page_config(
    page_title="ClickPlus Dashboard",
    page_icon="📈",
    layout="wide"
)

# --- Estilo Customizado (Retoque Fino) ---
# Injeta CSS para usar a fonte system-ui e fazer ajustes finos
st.markdown("""
    <style>
    html, body, [class*="st-"], [class*="css-"] {
        font-family: 'system-ui', 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
    }
    h1 {
        font-size: 2.2rem;
        font-weight: 700;
    }
    h2 {
        font-size: 1.8rem;
        font-weight: 600;
    }
    h3 {
        font-size: 1.4rem;
        font-weight: 600;
    }
    .stMetric {
        border-radius: 10px;
        padding: 15px;
        background-color: #262730;
    }
    </style>
    """, unsafe_allow_html=True)

# --- Funções de Apoio ---

@st.cache_data
def load_data():
    """
    Carrega os dados dos clientes e das vendas a partir de arquivos Parquet.
    O cache garante que os dados sejam carregados apenas uma vez.
    """
    try:
        ML_ZONE_DIR = './data/redis/'
        df_customers = pd.read_parquet(f'{ML_ZONE_DIR}customers.parquet')
        df_sales = pd.read_parquet(f'{ML_ZONE_DIR}sales.parquet')
        
        if 'nome_cliente' in df_sales.columns:
            df_sales.drop(columns=['nome_cliente'], inplace=True)
            
        df_full = pd.merge(df_sales, df_customers, on='id_cliente', how='left')
        return df_full
    except FileNotFoundError:
        st.error(f"Erro: Arquivos de dados não encontrados no diretório '{ML_ZONE_DIR}'. Verifique se os notebooks do pipeline de ML foram executados.")
        return pd.DataFrame()

def format_currency(value):
    """Formata um valor numérico como moeda brasileira de forma robusta."""
    try:
        return f"R$ {value:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")
    except (TypeError, ValueError):
        return "R$ 0,00"

# --- Páginas do Dashboard ---

def show_segmentation_page(df):
    """Exibe a página do Dashboard de Segmentação de Clientes."""
    st.title("📈 Dashboard de Segmentação de Clientes")
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
    selected_segments = st.sidebar.multiselect("Selecione os Segmentos", options=segment_options, default=segment_options)
    
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
            labels={'recencia': 'Recência (dias)', 'frequencia': 'Frequência (compras)'}
        )
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        st.subheader("Resumo por Segmento")
        segment_summary = filtered_rfv_df.groupby('segmento')['valor_total'].sum().sort_values(ascending=False)
        st.dataframe(segment_summary.map(format_currency))
        st.markdown("#### 💡 Insights Rápidos")
        try:
            top_segment = segment_summary.idxmax()
            st.info(f"O segmento **{top_segment}** é o mais valioso. Ações de fidelidade para este grupo podem maximizar o retorno.")
        except ValueError:
            st.warning("Nenhum segmento selecionado.")

    st.subheader("Detalhes dos Clientes")
    st.dataframe(filtered_rfv_df, hide_index=True, use_container_width=True)

def show_opportunities_page(df):
    """Exibe a página do Radar de Oportunidades, com o perfil 360 integrado."""
    st.title("🎯 Radar de Oportunidades de Venda")
    st.markdown("Identifique proativamente quais clientes abordar e o que oferecer.")

    st.sidebar.header("Filtros de Prospecção")
    dias = ['30 dias', '7 dias']
    colunas_predicao = ['prob_compra_30d', 'prob_compra_7d']
    prediction_model = st.sidebar.selectbox("Filtrar por probabilidade de compra em:", dias)
    index = dias.index(prediction_model)
    prob_column = colunas_predicao[index]
    
    prob_range = st.sidebar.slider(
        f"Probabilidade de Compra ({dias[index]}) entre:",
        min_value=0, max_value=100, value=(75, 100), step=1
    )
    min_prob, max_prob = prob_range[0] / 100.0, prob_range[1] / 100.0

    df_customers_unique = df.drop_duplicates(subset=['id_cliente']).set_index('id_cliente')
    
    opportunities_df = df_customers_unique[
        (df_customers_unique[prob_column] >= min_prob) &
        (df_customers_unique[prob_column] <= max_prob)
    ].copy()  # Usar .copy() para evitar SettingWithCopyWarning
    
    opportunities_df[prob_column] = opportunities_df[prob_column] * 100

    opportunities_df_sorted = opportunities_df.sort_values(by=prob_column, ascending=False)
    
    st.header(f"⚡ {len(opportunities_df_sorted)} clientes encontrados com os critérios selecionados!")
    
    st.subheader("Lista de Clientes Prioritários")
    st.dataframe(
        opportunities_df_sorted[['nome_cliente', prob_column, 'segmento', 'sugestao_prox_produto']],
        column_config={
            prob_column: st.column_config.ProgressColumn(
                "Prob. Próxima Compra (%)", format="%.0f%%", min_value=0, max_value=100,
                help="Probabilidade prevista pelo modelo XGBoost de o cliente comprar no período."
            )
        }, hide_index=True, use_container_width=True
    )

    st.subheader("🔍 Análise Individual do Cliente")
    if not opportunities_df_sorted.empty:
        selected_customer_name = st.selectbox(
            "Selecione um cliente da lista para ver o perfil completo:",
            options=opportunities_df_sorted['nome_cliente']
        )
        with st.expander(f"Ver Perfil Completo e Histórico de {selected_customer_name}", expanded=True):
            customer_data = opportunities_df_sorted[opportunities_df_sorted['nome_cliente'] == selected_customer_name].iloc[0]
            st.subheader(f"Perfil de Engajamento: {customer_data.name}")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric(label="Segmento (Persona)", value=customer_data['segmento'])
            with col2:
                st.metric(label=f"Prob. Compra ({dias[index]})", value=f"{customer_data[prob_column]:.0f}%")
            with col3:
                st.metric(label="Próximo Trecho Sugerido", value=customer_data['sugestao_prox_produto'])
            
            st.markdown("---")
            customer_history = df[df['nome_cliente'] == selected_customer_name]
            st.metric("Total Gasto (Lifetime Value)", format_currency(customer_history['valor_venda'].sum()))
            st.write("Histórico de Compras:")
            st.dataframe(customer_history[['data_venda', 'produto', 'valor_venda']], hide_index=True, use_container_width=True)
    else:
        st.warning("Nenhum cliente atende aos critérios de probabilidade selecionados.")

def show_executive_summary_page(df):
    """Exibe a página do Resumo Executivo Estratégico."""
    st.title("📊 Resumo Executivo ClickPlus")
    st.markdown(f"Relatório gerado em: **{datetime.now().strftime('%d/%m/%Y %H:%M')}**")
    
    prob_column = 'prob_compra_30d'
    
    df_customers_unique = df.drop_duplicates(subset=['id_cliente'])
    receita_preditiva = (df_customers_unique['valor_venda'].mean() * df_customers_unique[prob_column]).sum()
    clientes_em_risco = df_customers_unique[df_customers_unique['segmento'] == 'Em Risco']['nome_cliente'].nunique()
    segment_sales = df.groupby('segmento')['valor_venda'].sum()
    top_segment = segment_sales.idxmax()

    col1, col2, col3 = st.columns(3)
    col1.metric("Receita Preditiva (Próx. 30 dias)", format_currency(receita_preditiva), delta="Estimativa", delta_color="off")
    col2.metric("Clientes em Risco de Churn", f"{clientes_em_risco}")
    col3.metric("Segmento Mais Valioso", top_segment, help=f"Segmento que mais gerou receita: {format_currency(segment_sales.max())}")

    st.markdown("---")
    st.subheader("Análise Estratégica")
    st.markdown(f"""
        - **Visão Geral:** Nossa análise preditiva estima uma receita de **{format_currency(receita_preditiva)}** no próximo ciclo de vendas.
        - **Foco Principal:** O segmento **'{top_segment}'** continua sendo o motor de crescimento. Estratégias de retenção e up-selling para este grupo são cruciais.
        - **Ponto de Atenção:** Identificamos **{clientes_em_risco} clientes** no segmento 'Em Risco'. Uma campanha de reengajamento é recomendada para mitigar perdas.
        """)
    with st.expander("Ver detalhamento da Receita por Segmento"):
        st.bar_chart(segment_sales)

# --- Função Principal ---

def main():
    """Função principal que organiza a aplicação Streamlit."""
    try:
        locale.setlocale(locale.LC_ALL, 'pt_BR.UTF-8')
    except locale.Error:
        pass

    df = load_data()

    if not df.empty:
        st.sidebar.image("./logo.png", width=100)
        st.sidebar.title("Plataforma ClickPlus")
        
        page_selection = st.sidebar.radio(
            "Navegue pelas Análises",
            ["Dashboard de Segmentação", "Radar de Oportunidades", "Resumo Executivo"]
        )
        
        if page_selection == "Dashboard de Segmentação":
            show_segmentation_page(df)
        elif page_selection == "Radar de Oportunidades":
            show_opportunities_page(df)
        elif page_selection == "Resumo Executivo":
            show_executive_summary_page(df)

if __name__ == "__main__":
    main()