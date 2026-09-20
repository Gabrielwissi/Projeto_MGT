from sklearn.model_selection import cross_val_score, KFold, TimeSeriesSplit
from sklearn.metrics import root_mean_squared_error, r2_score
import pandas as pd
import numpy as np
from utils.metricas import metricas_regressao
import time

def treinamento_regressao(pipelines, X_train, y_train, X_test, y_test, tempo = False):

    resultados = []
    pipelines_treinados = {}

    if tempo:
        cv = TimeSeriesSplit(n_splits = 5)
    else:
        cv = KFold(n_splits = 5, shuffle = True, random_state = 42)

    for nome, pipeline in pipelines.items():

        inicio = time.time()

        print(f'Avaliando: {nome}')

        resultado_cv = cross_val_score(pipeline, X_train, y_train, cv = cv, scoring = 'neg_root_mean_squared_error', n_jobs = 5)

        pipeline.fit(X_train, y_train)
        predicao = pipeline.predict(X_test)

        tempo_execucao = time.time() - inicio

        rmse_cv = -resultado_cv.mean()
        desvpad_cv = resultado_cv.std()

        metricas_gerais = metricas_regressao(
            y_test = y_test,
            predicao = predicao
        )

        pipelines_treinados[nome] = pipeline

        delta = (
            ((metricas_gerais['RMSE'] - rmse_cv ) / rmse_cv) * 100
            if rmse_cv != 0
            else 0
        )

        resultados.append({
            'Modelo': nome,
            'RMSE CV': np.round(rmse_cv, 2),
            'Desvio Padrão CV': np.round(desvpad_cv, 2),
            'RMSE Teste': metricas_gerais['RMSE'],
            'MAE': metricas_gerais['MAE'],
            'R² Teste': metricas_gerais['R² Teste'],
            'Resíduo Médio': metricas_gerais['Resíduo Médio'],
            'Resíduo Desvio Padrão': metricas_gerais['Resíduo Desvio Padrão'],
            'Delta (%)': np.round(delta, 2),
            'tempo': tempo,
            'Tempo de Execução': tempo_execucao
        })

    df_resultados = pd.DataFrame(resultados).sort_values(by = 'R² Teste', ascending = False).reset_index(drop = True)
    return df_resultados, pipelines_treinados