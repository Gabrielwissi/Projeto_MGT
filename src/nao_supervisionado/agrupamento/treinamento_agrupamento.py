import pandas as pd
import numpy as np
import time
from utils.metricas import metricas_agrupamento

def treinamento_agrupamento(df, pipelines_numericos, pipelines_categoricos):
    
    resultados = []
    pipelines_treinados = {}

    for nome, pipeline in pipelines_numericos.items():

        print(f' Analisando: {nome}')

        try:
            inicio = time.time()
            rotulos = pipeline.fit_predict(df)
            tempo_execucao = time.time() - inicio

            df_trans = pipeline.named_steps['engenharia'].transform(df)

            if len(pipeline.steps) == 3:
                nome_reducao = pipeline.steps[1][0]
                df_trans = pipeline.named_steps[nome_reducao].transform(df_trans)

            metricas_gerais = metricas_agrupamento(df_trans, rotulos)

            if metricas_gerais['Clusters'] < 2:
                continue

            pipelines_treinados[nome] = pipeline

            resultados.append({
                'Modelo': nome,
                'Tipo' : 'Numérico',
                'Clusters': metricas_gerais['Clusters'],
                'Silhouette': metricas_gerais['Silhouette'],
                'Calinski-Harabasz': metricas_gerais['Calinski-Harabasz'],
                'Davies_Bouldin': metricas_gerais['Davies_Bouldin'],
                'Tempo': np.round(tempo_execucao, 4),
            })

        except Exception as e:

            print(f'Erro no Modelo {nome}: {str(e)}')

    for nome, pipeline in pipelines_categoricos.items():

        print(f'Analisando: {nome}')

        try:
            engenharia_mista = pipeline.named_steps['engenharia']
            modelo = pipeline.named_steps['Modelo']
            dados_misto = engenharia_mista.fit_transform(df)

            colunas_finais = engenharia_mista.get_feature_names_out()
            df_misto = pd.DataFrame(dados_misto, columns = colunas_finais, index = df.index)

            for coluna in colunas_finais:
                if 'categorico__' in coluna:
                    df_misto[coluna] = df_misto[coluna].astype(str)
                elif 'numerico__' in coluna:
                    df_misto[coluna] = pd.to_numeric(df_misto[coluna])
        
            if type(modelo).__name__ == 'KPrototypes':
                categorias_index = [df_misto.columns.get_loc(coluna) for coluna in df_misto.columns if 'categorico__' in coluna]
                modelo.categorical = categorias_index

                inicio = time.time()
                rotulos = modelo.fit_predict(df_misto, categorical = modelo.categorical)
                tempo_execucao = time.time() - inicio

            else:
                inicio = time.time()
                rotulos = modelo.fit_predict(df_misto)
                tempo_execucao = time.time() - inicio

            df_resultado = pd.get_dummies(df_misto)

            metricas_gerais = metricas_agrupamento(df_resultado, rotulos)

            if metricas_gerais['Clusters'] < 2:
                continue

            resultados.append({
                'Modelo': nome,
                'Tipo' : 'Misto',
                'Clusters': metricas_gerais['Clusters'],
                'Silhouette': metricas_gerais['Silhouette'],
                'Calinski-Harabasz': metricas_gerais['Calinski-Harabasz'],
                'Davies_Bouldin': metricas_gerais['Davies_Bouldin'],
                'Tempo': np.round(tempo_execucao, 4),
            })

        except Exception as e:

            print(f'Erro no Modelo {nome}: {str(e)}')

    df_resultado = pd.DataFrame(resultados)
    
    if not df_resultado.empty:
        df_resultado = df_resultado.sort_values(by = 'Silhouette', ascending = False).reset_index(drop = True)

    return df_resultado, pipelines_treinados