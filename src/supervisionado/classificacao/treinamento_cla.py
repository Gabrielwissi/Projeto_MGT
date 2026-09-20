import pandas as pd
import time
import numpy as np
from sklearn.model_selection import cross_val_score, StratifiedKFold, TimeSeriesSplit
from utils.metricas import metricas_classificacao

def treinamento_classificacao(pipelines, X_train, y_train, X_test, y_test, tempo = False):

    resultados = []
    pipelines_treinados = {}

    if tempo:
        cv = TimeSeriesSplit(n_splits = 5)
    else:
        cv = StratifiedKFold(n_splits = 5, shuffle = True, random_state = 42)

    for nome, pipeline in pipelines.items():

        inicio = time.time()

        print(f'Avaliando: {nome}')

        resultado_cv = cross_val_score(pipeline, X_train, y_train, cv = cv, scoring = 'accuracy', n_jobs = -1)

        pipeline.fit(X_train, y_train)
        predicao = pipeline.predict(X_test)

        tempo_execucao = time.time() - inicio

        acuracia_cv = resultado_cv.mean() * 100
        desvpad_cv = resultado_cv.std() * 100
        
        metricas_gerais, metricas_por_classe = metricas_classificacao(
            pipeline = pipeline,
            X_test = X_test,
            y_test = y_test,
            predicao = predicao
        )

        pipelines_treinados[nome] = pipeline

        delta = acuracia_cv - metricas_gerais['Acurácia (%)']

        resultados.append({
            'Modelo': nome,
            'Acurácia CV (%)': np.round(acuracia_cv, 2),
            'Desvio Padrão CV (%)': np.round(desvpad_cv, 2),
            'Acurácia Teste (%)': metricas_gerais['Acurácia (%)'],
            'F1 - Score': metricas_gerais['F1 Macro'],
            'Precision': metricas_gerais['Precision Macro'],
            'Recall': metricas_gerais['Recall Macro'],
            'AUC': metricas_gerais['AUC'],
            'Delta (%)': np.round(delta, 2),
            'Métricas por classe': metricas_por_classe,
            'tempo': tempo,
            'Tempo de Execução': np.round(tempo_execucao, 2)
        })

    df_resultados = pd.DataFrame(resultados).sort_values(by = 'Acurácia Teste (%)', ascending = False).reset_index(drop = True)
    return df_resultados, pipelines_treinados
