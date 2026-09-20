import optuna
import time
import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score, KFold, TimeSeriesSplit
from sklearn.metrics import root_mean_squared_error
from sklearn.pipeline import Pipeline
from utils.modelos import hiperparams_fixos, hiperparams_otimizacao
from utils.supervisionado.modelos_sup import modelos_supervisionados
from utils.metricas import metricas_regressao


def otimizar_regressao(df_ranking, pipelines, X_train, X_test, y_train, y_test, tempo = False, n_trials = 50, top_n = 3):

    resultados = []
    pipelines_otimizados = {}
    studies = {}

    if tempo:
        cv = TimeSeriesSplit(n_splits = 5)
    else:
        cv = KFold(n_splits = 5, shuffle = True, random_state = 42)

    top_modelos = df_ranking.head(top_n)['Modelo'].tolist()

    modelos_dict, reducoes_dict = modelos_supervisionados('Regressão')

    for nome_completo in top_modelos:

        nome_modelo = None

        for chave in modelos_dict.keys():
            if nome_completo.startswith(chave):
                nome_modelo = chave
                break

        if nome_modelo is None:
            print(f'Modelo não Encontrado para: {nome_modelo}')
            continue

        modelo_dados = modelos_dict[nome_modelo]
        modelo_classe = modelo_dados['modelo']
        modelo_info = modelo_dados['hiperparams']

        pipeline_sem_modelo = pipelines[nome_completo].steps[:-1]

        print(f'Otimizando: {nome_completo}')

        inicio = time.time()

        def objetivo(trial):

            hiperparams = hiperparams_otimizacao(trial, modelo_info)
            fixos = hiperparams_fixos(modelo_classe)
            modelo = modelo_classe(**hiperparams, **fixos)

            modelo_pipeline = Pipeline(steps = pipeline_sem_modelo + [('modelo', modelo)])

            score = cross_val_score(modelo_pipeline, X_train, y_train, cv = cv, scoring = 'neg_root_mean_squared_error', n_jobs = -1)
        
            return score.mean()
        
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study = optuna.create_study(direction = 'minimize')
        study.optimize(objetivo, n_trials = n_trials)

        tempo_otimizacao = time.time() - inicio

        inicio_treino = time.time()

        fixos = hiperparams_fixos(modelo_classe)
        modelo = modelo_classe(**study.best_params, **fixos)
        modelo_pipeline = Pipeline(steps = pipeline_sem_modelo + [('modelo', modelo)])

        modelo_pipeline.fit(X_train, y_train)
        predicao = modelo_pipeline.predict(X_test)

        metricas_gerais = metricas_regressao(
            y_test = y_test,
            predicao = predicao
        )

        rmse_cv = -study.best_value
        delta = rmse_cv - metricas_gerais['RMSE']

        tempo_treino = time.time() - inicio_treino

        pipelines_otimizados[nome_completo] = modelo_pipeline

        studies[nome_completo] = study

        melhores_hiperparams = ', '.join([f'{k}: {v}' for k, v in study.best_params.items()])

        resultados.append({
            'Modelo': nome_completo,
            'Melhores Hiperparâmetros': melhores_hiperparams,
            'RMSE CV': np.round(rmse_cv, 2),
            'RMSE Teste': metricas_gerais['RMSE'],
            'MAE': metricas_gerais['MAE'],
            'R² Teste': metricas_gerais['R² Teste'],
            'Resíduo Médio': metricas_gerais['Resíduo Médio'],
            'Resíduo Desvio Padrão': metricas_gerais['Resíduo Desvio Padrão'],
            'Delta (%)': np.round(delta, 2),
            'tempo': tempo,
            'Tempo otimização': np.round(tempo_otimizacao, 2),
            'tempo treino': np.round(tempo_treino, 2)
        })
    
    df_otimizado = pd.DataFrame(resultados).sort_values(by = 'R² Teste', ascending = False).reset_index(drop = True)    
    return df_otimizado, pipelines_otimizados, studies