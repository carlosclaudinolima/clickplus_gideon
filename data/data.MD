DataFrame salvo com sucesso em './data/customers.parquet'
Conteúdo do DataFrame lido do arquivo Parquet:
    id_cliente                    nome_cliente        segmento  prob_prox_compra sugestao_prox_produto datas_prox_compra
0          101               Siderúrgica Atlas        Em Risco              0.60             Produto A        08/12/2025
1          102         Construtora Rocha Forte      Hibernando              0.46      Licença Software        18/09/2025
2          103               Transportes Veloz        Em Risco              0.72             Serviço X        11/12/2025
3          104  Comércio Varejista Ponto Certo  Novos Clientes              0.06             Serviço X        16/12/2025
4          105            Escola Aprender Mais      Hibernando              0.25       Material Básico        23/12/2025
5          106          Hospital Vida Saudável  Novos Clientes              0.23         Plano Premium        07/10/2025
6          107             Serviços Tech Inova      Hibernando              0.07       Material Básico        26/11/2025
7          108           Indústria Têxtil Fina      Hibernando              0.08         Consultoria Y        08/10/2025
8          109                Logística Global  Novos Clientes              0.98             Serviço X        16/10/2025
9          110               Mercado Bom Preço  Novos Clientes              0.60             Serviço X        09/12/2025
10         111               Clínica Bem Estar        Campeões              0.57         Consultoria Y        31/10/2025
11         112              Universidade Saber        Em Risco              0.39             Serviço X        07/11/2025

DataFrame salvo com sucesso em './data/sales.parquet'
Conteúdo do DataFrame lido do arquivo Parquet:
     id_cliente                 data_venda           produto   valor_venda
0           102 2024-08-29 00:33:52.969190         Serviço X   2681.801198
1           108 2025-01-01 00:33:52.969227         Produto A   3757.776678
2           102 2025-05-30 00:33:52.969241  Licença Software   9576.756067
3           101 2024-08-04 00:33:52.969252         Produto A  11865.122744
4           110 2024-12-02 00:33:52.969264   Material Básico   6824.318442
..          ...                        ...               ...           ...
995         108 2023-08-30 00:33:52.980749   Material Básico  11161.321526
996         101 2025-08-19 00:33:52.980760   Material Básico   4928.195920
997         111 2023-12-26 00:33:52.980770     Plano Premium   2792.511815
998         104 2023-09-02 00:33:52.980781     Consultoria Y  12104.347324
999         112 2023-09-15 00:33:52.980791         Produto A  12085.249833

[1000 rows x 4 columns]
DataFrame salvo com sucesso em './data/full.parquet'
Conteúdo do DataFrame lido do arquivo Parquet:
     id_cliente                 data_venda           produto   valor_venda                    nome_cliente        segmento  prob_prox_compra sugestao_prox_produto datas_prox_compra
0           102 2024-08-29 00:33:52.969190         Serviço X   2681.801198         Construtora Rocha Forte      Hibernando              0.46      Licença Software        18/09/2025
1           108 2025-01-01 00:33:52.969227         Produto A   3757.776678           Indústria Têxtil Fina      Hibernando              0.08         Consultoria Y        08/10/2025
2           102 2025-05-30 00:33:52.969241  Licença Software   9576.756067         Construtora Rocha Forte      Hibernando              0.46      Licença Software        18/09/2025
3           101 2024-08-04 00:33:52.969252         Produto A  11865.122744               Siderúrgica Atlas        Em Risco              0.60             Produto A        08/12/2025
4           110 2024-12-02 00:33:52.969264   Material Básico   6824.318442               Mercado Bom Preço  Novos Clientes              0.60             Serviço X        09/12/2025
..          ...                        ...               ...           ...                             ...             ...               ...                   ...               ...
995         108 2023-08-30 00:33:52.980749   Material Básico  11161.321526           Indústria Têxtil Fina      Hibernando              0.08         Consultoria Y        08/10/2025
996         101 2025-08-19 00:33:52.980760   Material Básico   4928.195920               Siderúrgica Atlas        Em Risco              0.60             Produto A        08/12/2025
997         111 2023-12-26 00:33:52.980770     Plano Premium   2792.511815               Clínica Bem Estar        Campeões              0.57         Consultoria Y        31/10/2025
998         104 2023-09-02 00:33:52.980781     Consultoria Y  12104.347324  Comércio Varejista Ponto Certo  Novos Clientes              0.06             Serviço X        16/12/2025
999         112 2023-09-15 00:33:52.980791         Produto A  12085.249833              Universidade Saber        Em Risco              0.39             Serviço X        07/11/2025

[1000 rows x 9 columns]