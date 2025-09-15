import pandas as pd

def sample_dataframe_by_group(df: pd.DataFrame, group_column: str, sample_percentage: float) -> pd.DataFrame:
    """
    Reduz o tamanho de um DataFrame amostrando um percentual de grupos únicos.

    Garante que, se um grupo for selecionado, todos os seus registros
    sejam incluídos no DataFrame final.

    Args:
        df: O DataFrame original.
        group_column: A coluna usada para agrupar os dados (ex: 'fk_contact').
        sample_percentage: O percentual de grupos a serem selecionados (ex: 0.12 para 12%).

    Returns:
        Um novo DataFrame contendo aproximadamente o percentual desejado
        de dados, com a garantia de que os grupos estão completos.
    """
    print(f"DataFrame original com {len(df)} registros e {df[group_column].nunique()} clientes únicos.")

    # 1. Obter a lista de todos os clientes únicos
    unique_customers = df[group_column].unique()

    # 2. Amostrar o percentual desejado de clientes únicos
    # O método sample() do pandas é altamente otimizado para isso.
    # A semente (random_state) garante que a amostragem seja reprodutível.
    sampled_customers = pd.Series(unique_customers).sample(
        frac=sample_percentage,
        random_state=42
    )

    # 3. Filtrar o DataFrame original para manter apenas os registros dos clientes selecionados
    # O método isin() é muito eficiente para este tipo de filtro.
    df_sampled = df[df[group_column].isin(sampled_customers)]

    print(f"DataFrame final com {len(df_sampled)} registros ({len(df_sampled)/len(df):.2%})")
    print(f"e {df_sampled[group_column].nunique()} clientes únicos ({df_sampled[group_column].nunique()/df[group_column].nunique():.2%}).")

    return df_sampled

# --- Exemplo de Uso ---
if __name__ == '__main__':
    
    arquivo_origem = 'data/raw/dados.parquet'
    arquivo_destino = 'data/raw/dados_sample.parquet'
    # **Passo 1: Carregue seu DataFrame original aqui**
    # Substitua "vendas.csv" pelo nome do seu arquivo de 1.7 milhões de registros.
    try:
        #df_original = pd.read_csv("vendas.csv", encoding='latin1')
        df_original = pd.read_parquet(arquivo_origem)
        print("Arquivo de vendas carregado com sucesso.\n")

        # **Passo 2: Aplique a função para obter a amostra de 12%**
        df_reduzido = sample_dataframe_by_group(
            df=df_original,
            group_column='fk_contact',
            sample_percentage=0.12
        )

        # **Passo 3: Verifique o resultado**
        print("\nExemplo do DataFrame reduzido:")
        print(df_reduzido.head())

        # Verificação da integridade:
        # Pega um cliente aleatório da amostra
        cliente_exemplo = df_reduzido['fk_contact'].iloc[0]

        # Compara o número de compras nos dois DataFrames
        compras_originais = len(df_original[df_original['fk_contact'] == cliente_exemplo])
        compras_reduzidas = len(df_reduzido[df_reduzido['fk_contact'] == cliente_exemplo])

        print(f"\nVerificação de integridade para o cliente '{cliente_exemplo}':")
        print(f" - Compras no DataFrame original: {compras_originais}")
        print(f" - Compras no DataFrame reduzido: {compras_reduzidas}")
        assert compras_originais == compras_reduzidas
        print("=> O histórico do cliente foi mantido integralmente. Sucesso!")
        df_reduzido.to_parquet(arquivo_origem)
        print(f"Arquivo salvo com sucesso em {arquivo_origem}")
        
        

    except FileNotFoundError:
        print("Erro: O arquivo 'vendas.csv' não foi encontrado.")
        print("Por favor, substitua pelo nome correto do seu arquivo de dados.")