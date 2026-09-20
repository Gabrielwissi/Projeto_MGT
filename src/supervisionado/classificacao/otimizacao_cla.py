import optuna
import time
import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score, StratifiedKFold, TimeSeriesSplit
from sklearn.pipeline import Pipeline
from utils.modelos import hiperparams_fixos, hiperparams_otimizacao
from utils.supervisionado.modelos_sup import modelos_supervisionados
from utils.metricas import metricas_classificacao

def otimizar_classificacao(df_ranking, pipelines, X_train, X_test, y_train, y_test, tempo = False, n_trials = 50, top_n = 3):

    resultados = []
    pipelines_otimizados = {}
    studies = {}

    if tempo:
        cv = TimeSeriesSplit(n_splits = 5)
    else:
        cv = StratifiedKFold(n_splits = 5, shuffle = True, random_state = 42)

    top_modelos = df_ranking.head(top_n)['Modelo'].tolist()

    modelos_dict, reducoes_dict = modelos_supervisionados('Classificação')

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

            score = cross_val_score(modelo_pipeline, X_train, y_train, cv = cv, scoring = 'accuracy', n_jobs = -1)
        
            return score.mean()
        
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study = optuna.create_study(direction = 'maximize')
        study.optimize(objetivo, n_trials = n_trials)

        tempo_otimizacao = time.time() - inicio

        inicio_treino = time.time()

        fixos = hiperparams_fixos(modelo_classe)
        modelo = modelo_classe(**study.best_params, **fixos)
        modelo_pipeline = Pipeline(steps = pipeline_sem_modelo + [('modelo', modelo)])

        modelo_pipeline.fit(X_train, y_train)
        predicao = modelo_pipeline.predict(X_test)

        tempo_treino = time.time() - inicio_treino

        metricas_gerais, metricas_por_classe = metricas_classificacao(
            pipeline = modelo_pipeline,
            X_test = X_test,
            y_test = y_test,
            predicao = predicao
        )

        acuracia_cv = study.best_value * 100
        desvpad_cv = 0.0
        delta = acuracia_cv - metricas_gerais['Acurácia (%)']

        pipelines_otimizados[nome_completo] = modelo_pipeline
        
        studies[nome_completo] = study

        melhores_hiperparams = ', '.join([f'{k}: {v}' for k, v in study.best_params.items()])

        resultados.append({
            'Modelo': nome_completo,
            'Melhores Hiperparâmetros': melhores_hiperparams,
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
            'Tempo otimização': np.round(tempo_otimizacao, 2),
            'tempo treino': np.round(tempo_treino, 2)
        })

    df_otimizado = pd.DataFrame(resultados).sort_values(by = 'Acurácia Teste (%)', ascending = False).reset_index(drop = True)
    return df_otimizado, pipelines_otimizados, studies


