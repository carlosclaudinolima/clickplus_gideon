# --- PASSO 1: INSTALAÇÃO DAS BIBLIOTECAS ---
# Antes de executar este script, abra seu terminal e instale as dependências necessárias com o comando:
# pip install pandas "scikit-learn~=1.3.0" xgboost

import pandas as pd
import numpy as np
import xgboost as xgb

import warnings
from datetime import datetime
import warnings


from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


warnings.filterwarnings('ignore')


arquivo_feat = './data/raw/xgboost_feat.parquet'

def feature_engineering(df: pd.DataFrame):
    """
    Realiza a engenharia de features a partir do histórico de transações.
    """
    print("Iniciando engenharia de features...")

    # Garante que o dataframe está ordenado por cliente e data
    df = df.sort_values(['fk_contact', 'date_purchase'], ascending=[True, True])

    # Calcula a diferença de dias entre compras consecutivas para o mesmo cliente
    df['days_since_previous_purchase'] = df.groupby('fk_contact')['date_purchase'].diff().dt.days

    # Features de agregação (janela expansiva) para cada transação
    df['cumulative_purchases'] = df.groupby('fk_contact').cumcount() + 1
    df['cumulative_mean_gmv'] = df.groupby('fk_contact')['gmv_success'].transform(
        lambda x: x.expanding().mean()
    )
    df['cumulative_mean_days_between_purchases'] = df.groupby('fk_contact')['days_since_previous_purchase'].transform(
        lambda x: x.expanding().mean()
    )
    
    # Preenche NaNs que aparecem na primeira compra de cada cliente
    df['days_since_previous_purchase'].fillna(0, inplace=True)
    df['cumulative_mean_days_between_purchases'].fillna(0, inplace=True)
    
    df.to_parquet(arquivo_feat)
    
    print("Engenharia de features concluída.")
    return df



def create_targets(df, days_ahead):
    """
    Cria a variável alvo (target) para prever a compra nos próximos 'days_ahead' dias.
    """
    print(f"Criando alvo para {days_ahead} dias...")
    
    df['next_purchase_date'] = df.groupby('fk_contact')['date_purchase'].shift(-1)
    days_to_next_purchase = (df['next_purchase_date'] - df['date_purchase']).dt.days
    
    target_col_name = f'target_{days_ahead}_days'
    df[target_col_name] = np.where(days_to_next_purchase <= days_ahead, 1, 0)
    
    print("Criação do alvo concluída.")
    return df.drop(columns=['next_purchase_date'])

# --- Bloco Principal ---

def main():
    # 1. Carregamento e Preparação dos Dados
    try:
        # **IMPORTANTE**: Substitua "vendas.csv" pelo nome do seu arquivo de dados.
        # O encoding 'latin1' é comum em arquivos CSV gerados no Brasil, ajuste se necessário.
        #df = pd.read_csv("vendas.csv", encoding='latin1')
        df = pd.read_parquet("./data/raw/dados.parquet")
        
        df['date_purchase'] = pd.to_datetime(df['date_purchase'], errors='coerce')
        df.dropna(subset=['date_purchase'], inplace=True)

        print("Dados carregados com sucesso.")
        print(f"Total de registros: {len(df)}")
        print(f"Período dos dados: de {df['date_purchase'].min().date()} a {df['date_purchase'].max().date()}")

    except FileNotFoundError:
        print("Erro: O arquivo 'vendas.csv' não foi encontrado.")
        print("Por favor, certifique-se de que o arquivo de vendas está na mesma pasta que este script e o nome está correto.")
        return

    # 2. Engenharia de Features e Criação de Alvos
    
    #df_features = feature_engineering(df)
    df_features = df = pd.read_parquet(arquivo_feat)
    
    df_features = add_rfm_cluster_feature(df_features)
    
    df_7_days = create_targets(df_features.copy(), 7)
    df_30_days = create_targets(df_features.copy(), 30)

    # 3. Preparação para o Modelo
    features_to_use = [
        'gmv_success',
        'total_tickets_quantity_success',
        'days_since_previous_purchase',
        'cumulative_purchases',
        'cumulative_mean_gmv',
        'cumulative_mean_days_between_purchases',
        'cluster_label'
    ]

    # 4. Treinamento dos dois modelos
    model_7_days = train_and_evaluate(df_7_days, 'target_7_days', features_to_use)
    model_30_days = train_and_evaluate(df_30_days, 'target_30_days', features_to_use)

    print("\nTreinamento concluído!")
    print("Os modelos 'model_7_days' e 'model_30_days' foram gerados e estão prontos para uso.")

def train_and_evaluate(df_target, target_col, features, test_size=0.2):
    
    
    
    print("\n" + "="*50)
    print(f"Treinando modelo para: {target_col}")
    print("="*50)
    
    weight = 1 # Valor padrão
    if target_col == 'target_7_days':
        weight = 3.3
        print(f"\nUsando scale_pos_weight de {weight} para o modelo {target_col}\n")

    X = df_target[features]
    y = df_target[target_col]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42, stratify=y)
    
    print(f"Tamanho do conjunto de treino: {len(X_train)}")
    print(f"Tamanho do conjunto de teste: {len(X_test)}")

    model = xgb.XGBClassifier(
        objective='binary:logistic',
        eval_metric='logloss',
        use_label_encoder=False,
        n_estimators=1000,
        learning_rate=0.05,
        max_depth=4,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
        early_stopping_rounds=50,
        scale_pos_weight=weight,
    )

    model.fit(X_train, y_train,
              #early_stopping_rounds=50,
              eval_set=[(X_test, y_test)],
              verbose=False)

    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    print("\n--- Resultados da Avaliação ---")
    print(f"Acurácia: {accuracy_score(y_test, y_pred):.4f}")
    print(f"AUC-ROC: {roc_auc_score(y_test, y_pred_proba):.4f}")
    print("\nRelatório de Classificação:")
    print(classification_report(y_test, y_pred))
    
    feature_importance = pd.DataFrame({
        'feature': features,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\nImportância das Features:")
    print(feature_importance)

    return model



def add_rfm_cluster_feature(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calcula as métricas RFM para os clientes, executa a clusterização K-Means e
    adiciona o rótulo do cluster de cada cliente ao DataFrame original.

    Args:
        df: DataFrame de vendas contendo as colunas:
            - 'fk_contact' (ID do cliente)
            - 'date_purchase' (Data da compra)
            - 'nk_ota_localizer_id' (ID da compra/pedido)
            - 'gmv_success' (Valor da compra)

    Returns:
        DataFrame original com uma nova coluna 'cluster_label' contendo o
        cluster RFM de cada cliente.
    """
    print("Iniciando a criação da feature de cluster RFM...")

    # --- 1. Preparação dos Dados ---
    # Garante que a data está no formato correto
    df_copy = df.copy()
    df_copy['date_purchase'] = pd.to_datetime(df_copy['date_purchase'])

    # --- 2. Cálculo do RFM ---
    print("Calculando métricas RFM...")
    # Define a data de referência para o cálculo da Recência (última data de compra + 1 dia)
    snapshot_date = df_copy['date_purchase'].max() + pd.Timedelta(days=1)

    # Agrega os dados por cliente
    rfm_df = df_copy.groupby('fk_contact').agg({
        'date_purchase': lambda date: (snapshot_date - date.max()).days,
        'nk_ota_localizer_id': 'count', # ou 'nunique' se o ID de compra for por item
        'gmv_success': 'sum'
    })

    # Renomeia as colunas
    rfm_df.rename(columns={'date_purchase': 'Recency',
                           'nk_ota_localizer_id': 'Frequency',
                           'gmv_success': 'MonetaryValue'}, inplace=True)

    # --- 3. Normalização dos Dados ---
    # É crucial escalar os dados antes de aplicar o K-Means
    print("Normalizando os dados RFM...")
    scaler = StandardScaler()
    rfm_scaled = scaler.fit_transform(rfm_df)
    rfm_scaled = pd.DataFrame(rfm_scaled, index=rfm_df.index, columns=rfm_df.columns)

    # --- 4. Aplicação do K-Means ---
    # Com base no gráfico do Elbow Method, usamos n_clusters=3
    print("Executando o K-Means com k=4...")
    kmeans = KMeans(n_clusters=4, init='k-means++', n_init='auto', random_state=42)
    kmeans.fit(rfm_scaled)

    # Adiciona os rótulos do cluster ao DataFrame RFM
    rfm_df['cluster_label'] = kmeans.labels_

    # --- 5. Unir a feature ao DataFrame Original ---
    print("Adicionando a coluna 'cluster_label' ao DataFrame original...")
    # Une a coluna de cluster ao DataFrame original com base no ID do cliente
    df_final = df.merge(rfm_df[['cluster_label']], on='fk_contact', how='left')
    
    print("Processo concluído com sucesso!")
    return df_final

# --- Exemplo de Uso ---
# if __name__ == '__main__':
#     # Criando um DataFrame de exemplo para demonstrar a função
#     # Em um caso real, você carregaria seu CSV aqui:
#     # df_vendas = pd.read_csv("vendas.csv")
#     data = {
#         'nk_ota_localizer_id': [1, 2, 3, 4, 5, 6, 7],
#         'fk_contact': ['C1', 'C2', 'C1', 'C3', 'C2', 'C1', 'C3'],
#         'date_purchase': [
#             '2024-01-10', '2024-01-15', '2024-03-20', '2024-02-01',
#             '2024-03-25', '2024-04-01', '2024-04-05'
#         ],
#         'gmv_success': [100, 150, 110, 500, 160, 120, 550]
#     }
#     df_exemplo = pd.DataFrame(data)

#     # Aplicando a função
#     df_com_cluster = add_rfm_cluster_feature(df_exemplo)

#     # Visualizando o resultado
#     print("\nDataFrame original com a nova coluna 'cluster_label':")
#     print(df_com_cluster)

#     # Para entender o que cada cluster significa, você pode analisar o RFM médio de cada um
#     print("\nAnálise dos clusters (valores médios de RFM por cluster):")
#     cluster_analysis = df_com_cluster.groupby('cluster_label')[['Recency', 'Frequency', 'MonetaryValue']].mean()
#     print(cluster_analysis)




if __name__ == '__main__':
    main()
    
    
