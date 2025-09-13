import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import locale
from datetime import datetime, timedelta


import random
from datetime import date, timedelta

import pyarrow as pa
import pyarrow.parquet as pq

# --- Configuração da Página ---
# Define o layout da página para o modo 'wide', aproveitando toda a largura da tela.
# st.set_page_config(layout="wide")

def df_to_parket(df, nome_do_arquivo):
    
    # Converta e salve o DataFrame em um arquivo Parquet
    # O método 'to_parquet()' do pandas usa pyarrow ou fastparquet automaticamente
    df.to_parquet(nome_do_arquivo, engine='pyarrow', index=False)

    print(f"DataFrame salvo com sucesso em '{nome_do_arquivo}'")

    # Opcional: verifique se o arquivo foi salvo corretamente lendo-o de volta
    df_lido = pd.read_parquet(nome_do_arquivo, engine='pyarrow')
    print("\nConteúdo do DataFrame lido do arquivo Parquet:")
    print(df_lido)


def generate_random_dates(start_date, end_date, num_dates):
    """Gera uma lista de datas aleatórias em formato de string.

    Args:
        start_date (str): Data de início no formato 'dd/mm/yyyy'.
        end_date (str): Data de fim no formato 'dd/mm/yyyy'.
        num_dates (int): Número de datas a serem geradas.

    Returns:
        list: Uma lista de strings com as datas aleatórias.
    """
    
    # Converte as datas de string para objetos date
    start = date.fromisoformat(f"{start_date[6:]}-{start_date[3:5]}-{start_date[:2]}")
    end = date.fromisoformat(f"{end_date[6:]}-{end_date[3:5]}-{end_date[:2]}")
    
    # Calcula a diferença em dias
    delta = end - start
    
    random_dates = []
    for _ in range(num_dates):
        # Gera um número aleatório de dias para adicionar à data de início
        random_days = random.randint(0, delta.days)
        random_date = start + timedelta(days=random_days)
        
        # Formata a data para 'dd/mm/yyyy' e adiciona à lista
        random_dates.append(random_date.strftime("%d/%m/%Y"))
        
    return random_dates



# Configuração da página para o modo wide, para melhor aproveitamento do espaço
st.set_page_config(layout="wide")

@st.cache_data
def generate_fake_data(num_records=1000):
    """
    Gera um DataFrame de vendas fictícias e enriquecidas para a prototipagem.
    """
    # Dados base
    customer_names = [
        "Siderúrgica Atlas", "Construtora Rocha Forte", "Transportes Veloz", "Comércio Varejista Ponto Certo", 
        "Escola Aprender Mais", "Hospital Vida Saudável", "Serviços Tech Inova", "Indústria Têxtil Fina",
        "Logística Global", "Mercado Bom Preço", "Clínica Bem Estar", "Universidade Saber"
    ]
    
    
    # Define as datas de início e fim
    start_date_str = "05/09/2025"
    end_date_str = "31/12/2025" 

    # Gera e imprime a lista de 12 datas aleatórias
    datas_prox_compra = generate_random_dates(start_date_str, end_date_str, len(customer_names))
    
    products = ["Produto A", "Serviço X", "Licença Software", "Consultoria Y", "Material Básico", "Plano Premium"]
    segments = ["Campeões", "Fiéis", "Em Risco", "Novos Clientes", "Hibernando"]
    
    # Criação de um DataFrame de clientes únicos para atribuir características
    customer_ids = range(101, 101 + len(customer_names))
    df_customers = pd.DataFrame({
        'id_cliente': customer_ids,
        'nome_cliente': customer_names,
        'segmento': np.random.choice(segments, len(customer_names), p=[0.1, 0.2, 0.2, 0.3, 0.2]),
        'prob_prox_compra': np.random.uniform(0.05, 0.99, len(customer_names)).round(2),
        'sugestao_prox_produto': np.random.choice(products, len(customer_names)),
        'datas_prox_compra': datas_prox_compra
    })
    
    #df_to_parket(df_customers, "./data/customers.parquet")

    # Geração dos registros de vendas
    sales_data = []
    for _ in range(num_records):
        customer_id = np.random.choice(customer_ids)
        sale_date = datetime.now() - timedelta(days=np.random.randint(1, 730))
        product = np.random.choice(products)
        sale_value = np.random.uniform(500, 15000)
        sales_data.append([customer_id, sale_date, product, sale_value])

    df_sales = pd.DataFrame(sales_data, columns=['id_cliente', 'data_venda', 'produto', 'valor_venda'])
    #df_to_parket(df_sales, "./data/sales.parquet")
    
    # Combina os dados de vendas com os dados dos clientes
    df_full = pd.merge(df_sales, df_customers, on='id_cliente')
    #df_to_parket(df_full, "./data/full.parquet")
    
    return df_full

@st.cache_data
def load_data():
    CURATED_ZONE_DIR = './data/redis/'
    
    df_customers = pd.read_parquet(f'{CURATED_ZONE_DIR}customers.parquet')        
    df_sales = pd.read_parquet(f'{CURATED_ZONE_DIR}sales.parquet')
        
    # Combina os dados de vendas com os dados dos clientes
    df_full = pd.merge(df_sales, df_customers, on='id_cliente')
    
    return df_full

def show_segmentation_page(df):
    """
    Exibe a página do Protótipo 1: Dashboard de Segmentação de Clientes.
    """
    st.title("📈 Dashboard de Segmentação de Clientes")
    st.markdown("Analise os perfis de clientes para criar estratégias de marketing e vendas mais eficazes.")

    # --- Lógica de Análise RFV (Recência, Frequência, Valor) ---
    today = datetime.now()
    
    print(df.info())
    
    rfv_df = df.groupby('nome_cliente').agg(
        recencia=('data_venda', lambda date: (today - date.max()).days),
        frequencia=('data_venda', 'count'),
        valor_total=('valor_venda', 'sum'),
        segmento=('segmento', 'first') # Pega o segmento pré-definido
    ).reset_index()

    # --- Barra Lateral de Filtros ---
    st.sidebar.header("Filtros")
    selected_segments = st.sidebar.multiselect(
        "Selecione os Segmentos",
        options=rfv_df['segmento'].unique(),
        default=rfv_df['segmento'].unique()
    )
    
    filtered_rfv_df = rfv_df[rfv_df['segmento'].isin(selected_segments)]

    # --- Métricas Chave ---
    total_clientes = filtered_rfv_df['nome_cliente'].nunique()
    receita_total = filtered_rfv_df['valor_total'].sum()
    ticket_medio = receita_total / total_clientes if total_clientes > 0 else 0

    col1, col2, col3 = st.columns(3)
    col1.metric("Clientes Ativos", f"{total_clientes}", "no período")
    #col2.metric("Receita Total Gerada", f"R$ {receita_total:,.2f}")
    #col3.metric("Ticket Médio por Cliente", f"R$ {ticket_medio:,.2f}")
    # Streamlit não está obedecendo o locale então uma medida drástica foi tomada para formatar a saída
    col2.metric("Receita Total Gerada", f"R$ {receita_total:,.2f}".replace(",", "X").replace(".", ",").replace("X", "."))
    col3.metric("Ticket Médio por Cliente", f"R$ {ticket_medio:,.2f}".replace(",", "X").replace(".", ",").replace("X", "."))
    st.markdown("---")

    # --- Gráficos e Tabelas ---
    col1, col2 = st.columns([2, 1]) # Coluna do gráfico maior que a do resumo

    with col1:
        st.subheader("Visualização dos Segmentos (Recência vs Frequência)")
        fig = px.scatter(
            filtered_rfv_df,
            x='recencia',
            y='frequencia',
            size='valor_total',
            color='segmento',
            hover_name='nome_cliente',
            size_max=60,
            title="Posicionamento de Clientes por RFV"
        )
        fig.update_layout(xaxis_title="Recência (dias desde a última compra)", yaxis_title="Frequência (nº de compras)")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Resumo por Segmento")
        segment_summary = filtered_rfv_df.groupby('segmento')['valor_total'].sum().sort_values(ascending=False)
        st.dataframe(
            segment_summary,
            column_config={
                "segmento": st.column_config.Column(
                    "Segmento",
                    help="Segmento",
                ),
                "valor_total": st.column_config.NumberColumn(
                    "Valor Total",
                    format="%.2f",
                    help="Valor total de compra",
                ),            
            },
        )
        
        # --- Data Storytelling ---
        st.markdown("#### 💡 Insights Rápidos")
        top_segment = segment_summary.idxmax()
        st.info(f"O segmento **{top_segment}** é o mais valioso, representando a maior parte da receita. Focar em ações de fidelidade para este grupo pode maximizar o retorno.")

    st.subheader("Detalhes dos Clientes no Segmento")
    st.dataframe(
        filtered_rfv_df,
        
        column_config={
            "nome_cliente": st.column_config.Column(
                "Cliente",
                help="Nome do Cliente",
            ),
            "recencia": st.column_config.Column(
                "Recência (dias)",
                help="Recência (dias desde a última compra)",
            ),
            "frequencia": st.column_config.NumberColumn(
                "Frequência",
                help="Frequência (nº de compras)",
            ),
            "valor_total": st.column_config.NumberColumn(                
                "Valor Total",
                format="%.2f",
                help="Valor Total",
            ),
            "segmento": st.column_config.Column(
                "Segmento",
                help="Segmento",
            ),
        
        },
        hide_index=True,
    )


def show_opportunities_page(df):
    """
    Exibe a página do Protótipo 2: Radar de Oportunidades de Venda.
    """
    st.title("🎯 Radar de Oportunidades de Venda")
    st.markdown("Identifique proativamente quais clientes abordar e o que oferecer.")

    # --- Barra Lateral de Filtros ---
    st.sidebar.header("Filtros de Prospecção")
    prob_threshold = st.sidebar.slider(
        "Mostrar clientes com probabilidade de compra acima de:",
        min_value=0, max_value=100, value=75, step=5
    ) / 100.0

    # Filtra clientes únicos com base na probabilidade
    df_customers_unique = df.drop_duplicates(subset=['id_cliente']).set_index('id_cliente')
    
    opportunities_df = df_customers_unique[df_customers_unique['prob_prox_compra'] >= prob_threshold]
    opportunities_df = opportunities_df.sort_values(by='prob_prox_compra', ascending=False)
    
    # --- Data Storytelling Header ---
    st.header(f"⚡ Encontramos {len(opportunities_df)} clientes com alta chance de comprar!")
    
    # --- Tabela de Ação ---
    st.subheader("Lista de Clientes Prioritários")
    st.dataframe(
        opportunities_df[['nome_cliente', 'prob_prox_compra', 'segmento', 'sugestao_prox_produto', 'datas_prox_compra']],
        
        column_config={
            "id_cliente": st.column_config.Column(
                "ID",
                help="ID do Cliente",
            ),
            "nome_cliente": st.column_config.Column(
                "Cliente",
                help="Nome do Cliente",
            ),
            "prob_prox_compra": st.column_config.NumberColumn(
                "Prob. próxima compra (%)",
                format="percent",
                help="Probabilidade da próxima compra",
            ),
            "segmento": st.column_config.Column(
                "Segmento",
                help="Segmento",
            ),
            "sugestao_prox_produto": st.column_config.Column(
                "Próximo trecho",
                help="Próximo trecho",
            ),
            "datas_prox_compra": st.column_config.Column(
                "Data próxima compra",
                help="Data da próxima compra",
            ),
        },
        hide_index=True,
        use_container_width=True
    )

    # --- Detalhes do Cliente (Drill-down) ---
    st.subheader("🔍 Análise Individual do Cliente")
    if not opportunities_df.empty:
        selected_customer = st.selectbox(
            "Selecione um cliente da lista para ver mais detalhes:",
            options=opportunities_df['nome_cliente']
        )
        
        with st.expander(f"Ver histórico de {selected_customer}"):
            customer_history = df[df['nome_cliente'] == selected_customer]
            #st.metric("Total Gasto pelo Cliente", f"R$ {customer_history['valor_venda'].sum():,.2f}")
            # Streamlit não está obedecendo o locale então uma medida drástica foi tomada para formatar a saída
            st.metric("Total Gasto pelo Cliente", f"R$ {customer_history['valor_venda'].sum():,.2f}".replace(",", "X").replace(".", ",").replace("X", "."))
            st.write("Histórico de Compras:")
            st.dataframe(customer_history[['data_venda', 'produto', 'valor_venda']],
                         column_config={
                            "data_venda": st.column_config.DatetimeColumn(
                                "Data da Venda",
                                help="ID do Cliente",
                            ),
                            "produto": st.column_config.Column(
                                "Produto",
                                help="Nome do Produto",
                            ),
                            "valor_venda": st.column_config.NumberColumn(
                                "Valor da Venda",
                                #format="localized",
                                format="%.2f",
                                help="Probabilidade da próxima compra",
                            ),                            
                        },
                        hide_index=True, 
                    )
    else:
        st.warning("Nenhum cliente atende ao critério de probabilidade selecionado.")


def show_executive_summary_page(df):
    """
    Exibe a página do Protótipo 3: Resumo Executivo Estratégico.
    """
    st.title("📊 Resumo Executivo ClickPlus")
    st.markdown(f"Relatório gerado em: **{datetime.now().strftime('%d/%m/%Y %H:%M')}**")
    
    # --- Cálculos para KPIs ---
    df_customers_unique = df.drop_duplicates(subset=['id_cliente'])
    receita_preditiva = (df_customers_unique['valor_venda'].mean() * df_customers_unique['prob_prox_compra']).sum()
    clientes_em_risco = df_customers_unique[df_customers_unique['segmento'] == 'Em Risco']['nome_cliente'].nunique()
    segment_sales = df.groupby('segmento')['valor_venda'].sum()
    top_segment = segment_sales.idxmax()

    # --- Métricas de Alto Impacto ---
    col1, col2, col3 = st.columns(3)
    #col1.metric("Receita Preditiva (Próximo Ciclo)", f"R$ {receita_preditiva:,.0f}", delta="Estimativa", delta_color="off")
    # Streamlit não está obedecendo o locale então uma medida drástica foi tomada para formatar a saída
    col1.metric("Receita Preditiva (Próximo Ciclo)", f"R$ {receita_preditiva:,.0f}".replace(",", "X").replace(".", ",").replace("X", "."), delta="Estimativa", delta_color="off")
    col2.metric("Clientes em Risco de Churn", f"{clientes_em_risco}", help="Clientes no segmento 'Em Risco'.")
    col3.metric("Segmento Mais Valioso", top_segment, help=f"Segmento que mais gerou receita: R$ {segment_sales.max():,.0f}")

    st.markdown("---")

    # --- Análise Narrativa (Data Storytelling) ---
    st.subheader("Análise Estratégica")
    st.markdown(
        f"""
        - **Visão Geral:** Nossa análise preditiva estima uma receita de **R$ {receita_preditiva:,.2f}** no próximo ciclo de vendas, considerando o comportamento atual da base de clientes.
        - **Foco Principal:** O segmento **'{top_segment}'** continua sendo o motor de crescimento. Estratégias de retenção e up-selling para este grupo são cruciais.
        - **Ponto de Atenção:** Identificamos **{clientes_em_risco} clientes** com alto valor em risco de evasão. Uma campanha de reengajamento direcionada é recomendada com urgência para mitigar perdas.
        """
    )

    # --- Gráfico Chave e Detalhamento ---
    with st.expander("Ver detalhamento da Receita por Segmento"):
        st.bar_chart(segment_sales)
        st.markdown("O gráfico acima ilustra a contribuição de cada segmento para a receita total. Use esta informação para alocar recursos de marketing e vendas de forma mais inteligente.")


def show_ingestao_dados():
    #st.title("Ingestão de dados")
    #sst.markdown("Faça upload de um arquivos.")
    up = st.file_uploader("upload file", type={"csv", "txt"})
    if up is not None:
        spectra_df = pd.read_csv(up)
        st.write(spectra_df)
        
        
def show_customer_360_page(df):
    """
    Exibe a página "Visão 360° do Cliente", consolidando todas as informações.
    """
    st.title("👤 Visão 360° do Cliente")
    st.markdown("Uma visão completa dos insights, cluster e previsões para um cliente específico.")

    df_customers_unique = df.drop_duplicates(subset=['id_cliente'])
    
    selected_customer_name = st.selectbox(
        "Selecione um cliente para análise:",
        options=sorted(df_customers_unique['nome_cliente'].unique())
    )

    if selected_customer_name:
        customer_data = df_customers_unique[df_customers_unique['nome_cliente'] == selected_customer_name].iloc[0]
        
        st.header(f"Análise de {customer_data['nome_cliente']}")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.subheader("Perfil do Cliente")
            st.metric(label="Segmento (Persona)", value=customer_data['segmento'])
            st.info(f"Este cliente pertence ao grupo '{customer_data['segmento']}', com base em seu comportamento de compras (RFM).")

        with col2:
            st.subheader("Previsão de Compra")
            st.metric(label="Prob. de Compra em 7 Dias", value=f"{customer_data['prob_compra_7d']:.0%}")
            st.metric(label="Prob. de Compra em 30 Dias", value=f"{customer_data['prob_compra_30d']:.0%}")
            st.warning(f"O modelo XGBoost indica uma probabilidade de {customer_data['prob_compra_30d']:.0%} de este cliente realizar uma nova compra no próximo mês.")

        with col3:
            st.subheader("Recomendação")
            st.metric(label="Próximo Trecho Sugerido", value=customer_data['sugestao_prox_trecho'])
            st.success(f"Com base em seu perfil, a recomendação de próximo trecho é **{customer_data['sugestao_prox_trecho']}**.")

        with st.expander("Ver Histórico de Compras Detalhado"):
            customer_history = df[df['nome_cliente'] == selected_customer_name]
            st.metric("Total Gasto (Lifetime Value)", f"R$ {customer_history['valor_venda'].sum():,.2f}")
            colunas_historico = ['data_venda', 'trecho_alias', 'valor_venda', 'tipo_viagem', 'viaja_sozinho']
            st.dataframe(customer_history[colunas_historico])

def main():
    
    """
    Função principal que organiza a aplicação Streamlit.
    """
    locale.setlocale(locale.LC_ALL, 'pt_BR.UTF-8')
    # Gera os dados uma única vez
    #df = generate_fake_data()
    df = load_data()

    # Menu de navegação na barra lateral
    st.sidebar.image("./logo.png", width=100) # Um logo genérico
    st.sidebar.title("Plataforma ClickPlus")
    page_selection = st.sidebar.radio(
        "Navegue pelos Protótipos",
        [
            "Dashboard de Segmentação", 
            "Radar de Oportunidades", 
            "Resumo Executivo", 
            #"Ingestão de Dados",
            "Visão 360° do Cliente"
        ]
    )

    # Exibe a página selecionada
    if page_selection == "Dashboard de Segmentação":
        show_segmentation_page(df)
    elif page_selection == "Radar de Oportunidades":
        show_opportunities_page(df)
    elif page_selection == "Resumo Executivo":
        show_executive_summary_page(df)
    elif page_selection == "Ingestão de Dados":
        show_ingestao_dados()
    elif page_selection == "Visão 360° do Cliente":
            show_customer_360_page(df)

if __name__ == "__main__":
    main()
    