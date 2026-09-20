import optuna
import numpy as np
import pandas as pd
from sklearn.metrics import silhouette_score
from sklearn.pipeline import Pipeline
from utils.modelos import hiperparams_fixos, hiperparams_otimizacao
from utils.nao_supervisionado.modelos_nsup import modelos_agrupamento
from utils.metricas import metricas_agrupamento

def otimizar_agrupamento(df_resultado, pipelines, df, n_trials = 50, top_n = 3):

    resultados = []
    pipelines_otimizados = {}
    studies = {}
    
    top_modelos = df_resultado.head(top_n)['Modelo'].tolist()

    modelos_num_dict, modelos_cat_dict, reducoes_dict = modelos_agrupamento()

    for nome_completo in top_modelos:

        nome_modelo = None
        origem_dados = None

        for chave in modelos_num_dict.keys():
            if nome_completo.startswith(chave):
                nome_modelo = chave
                origem_dados = modelos_num_dict
                break

        if nome_modelo is None:
            for chave in modelos_cat_dict.keys():
                if nome_completo.startswith(chave):
                    nome_modelo = chave 
                    origem_dados = modelos_cat_dict    
                    break

        if nome_modelo is None:
            print(f'Modelo não Encontrado para: {nome_completo}')
            continue

        modelo_dados = origem_dados[nome_modelo]
        modelo_classe = modelo_dados['modelo']
        modelo_info = modelo_dados['hiperparams']
        
        pipeline_sem_modelo = pipelines[nome_completo].steps[:-1]
        df_trans = Pipeline(steps = pipeline_sem_modelo).fit_transform(df)

        def objetivo(trial):
            
            try:
                hiperparams = hiperparams_otimizacao(trial, modelo_info)
                fixos = hiperparams_fixos(modelo_classe)
                modelo = modelo_classe(**hiperparams, **fixos)

                rotulos = modelo.fit_predict(df_trans)
                clusters = len(set(rotulos)) - (1 if -1 in rotulos else 0)

                if clusters < 2:
                    return - 1
                
                score = silhouette_score(df_trans, rotulos)
                return score

            except Exception:
                return -1
        
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study = optuna.create_study(direction = 'maximize')
        study.optimize(objetivo, n_trials = n_trials)

        fixos = hiperparams_fixos(modelo_classe)
        modelo = modelo_classe(**study.best_params, **fixos)
        modelo_pipeline = Pipeline(steps = pipeline_sem_modelo + [('modelo', modelo)])

        rotulos_treino = modelo_pipeline.fit_predict(df)

        df_treino_trans = Pipeline(steps = pipeline_sem_modelo).transform(df)
        silhouette = study.best_value
        
        metricas_gerais = metricas_agrupamento(df_treino_trans, rotulos_treino)

        pipelines_otimizados[nome_completo] = modelo_pipeline
        studies[nome_completo] = study

        resultados.append({
                'Modelo': nome_completo,    
                'Melhores Hiperparâmetros': study.best_params,
                'Clusters': metricas_gerais['Clusters'],
                'Silhouette best': np.round(silhouette, 4),
                'Silhouette': metricas_gerais['Silhouette'],
                'Calinski-Harabasz': metricas_gerais['Calinski-Harabasz'],
                'Davies_Bouldin': metricas_gerais['Davies_Bouldin'],
            })
        
    df_otimizado = pd.DataFrame(resultados).sort_values(by = 'Silhouette', ascending = False).reset_index(drop = True)

    return df_otimizado, pipelines_otimizados, studies