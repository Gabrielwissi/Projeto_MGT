import os
import json
from datetime import datetime
import joblib
import numpy as np
import pandas as pd
from IPython.display import display

def salvar_modelo(pipeline, nome_pipeline, tipo, metrica_nome, metrica_valor, hiperparametros = None, metricas = None, metricas_por_classe = None):

    pasta = 'modelos_salvos'
    os.makedirs(pasta, exist_ok = True)

    data = datetime.now().strftime('%Y_%m_%d_%H-%M')

    nome = nome_pipeline.replace(' ', '_').replace('+', '_')

    valor = np.round(metrica_valor, 4)

    nome_arquivo = f'{data}_{tipo}_{nome}_{metrica_nome}_{valor}'
    
    caminho_pipeline = os.path.join(pasta, f'{nome_arquivo}.joblib')
    caminho_metadado = os.path.join(pasta, f'{nome_arquivo}_metadata.json')

    joblib.dump(pipeline, caminho_pipeline)
    
    print(f'Pipeline salvo em: {caminho_pipeline}')

    metadados = {
        'modelo': nome_pipeline,
        'tipo': tipo,
        'data': data,
        'metrica_principal': {
            'nome': metrica_nome,
            'valor': float(valor)
        },
        'metricas': {
            k: float(v) if isinstance(v, (np.floating, np.integer)) else str(v)
            for k, v in (metricas or {}).items()
            if v is not None and not (isinstance(v, float) and np.isnan(v))
        },
        'metricas_por_classe': metricas_por_classe or (),
        'hiperparametros': {
            k: str(v) if not isinstance(v, (int, float, str, bool)) else v
            for k, v in (hiperparametros or {}).items()
        },
        'caminho_pipeline': caminho_pipeline,
    }

    with open(caminho_metadado, 'w', encoding = 'utf-8') as f:
        json.dump(metadados, f, indent = 4, ensure_ascii = False)
    
    print(f'Metadados salvos em: {caminho_metadado}')

    return caminho_pipeline, caminho_metadado

def selecionar_e_salvar(resultado, tipo, etapa = 'Treinamento'):

    print(f'Salvar Modelo {etapa}')

    tipo = tipo or resultado.get('tipo')

    df_resultado = resultado.get('df_resultado')
    df_otimizado = resultado.get('df_otimizado')
    pipelines_treinados = resultado.get('pipelines_treinados')
    pipelines_otimizados = resultado.get('pipelines_otimizados')
    
    if df_otimizado is not None and not df_otimizado.empty and pipelines_otimizados:
        df_salvo = resultado.get('df_otimizado')
        pipeline_salvo = resultado.get('pipelines_otimizados')
        etapa = etapa or 'Optuna'


    elif df_resultado is not None and not df_resultado.empty and pipelines_treinados:
        df_salvo = resultado.get('df_resultado')
        pipeline_salvo = resultado.get('pipelines_treinados')
        etapa = etapa or 'Treinamento'

    else:
        return None, None, None
    
    resumo = ['Modelo']    

    if 'Acurácia Teste (%)' in df_salvo.columns:
        resumo.append('Acurácia Teste (%)')
    elif 'R² Teste' in df_salvo.columns:
        resumo.append('R² Teste')
    elif 'Silhouette' in df_salvo.columns:
        resumo.append('Silhouette')

    df_resumo = df_salvo[resumo].reset_index(drop = True)
    display(df_resumo)

    print('Deseja Salvar um Modelo? (s/n)')

    resposta = input().strip().lower()

    if resposta != 's':
        print('Nenhum Modelo foi Salvo')
        return None, None, None
    
    print(f'Digite o Número do Modelo que Deseja Salvar (0 a {len(df_salvo) - 1}):')

    try:
        indice = int(input().strip())
        modelo_escolhido = df_salvo.iloc[indice]
    except (ValueError, IndexError):
        print(f'Indice Não Encontrado!, selecione entre (0 a {len(df_salvo) - 1})')    
        return None, None, None

    if tipo == 'Classificação':
        metrica_nome = 'Acuracia'
        metrica_valor = modelo_escolhido['Acurácia Teste (%)']
    elif tipo == 'Regressão':
        metrica_nome = 'R²'
        metrica_valor = modelo_escolhido['R² Teste']
    elif tipo == 'Agrupamento':
        metrica_nome = 'Silhouette'
        metrica_valor = modelo_escolhido.get('Silhouette')

    hiperparametros = modelo_escolhido.get('Melhores Hiperparâmetros', None)
    metricas_por_classe = modelo_escolhido.get('Métricas por Classe', None)
    metricas = {k: v for k, v in modelo_escolhido.items()
                      if k not in ['Modelo', 'Melhores Hiperparâmetros', 'Métricas por Classe']
                      and isinstance(v, (int, float, np.integer, np.floating))
                      and not pd.isna(v)}

    print("Colunas no modelo_escolhido:")
    for k, v in modelo_escolhido.items():
        print(f"  {k}: {type(v).__name__} = {v}")

    caminho_pipeline, caminho_metadado = salvar_modelo(
        pipeline = pipeline_salvo[modelo_escolhido['Modelo']],
        nome_pipeline = modelo_escolhido['Modelo'],
        tipo = tipo,
        metrica_nome = metrica_nome,
        metrica_valor = metrica_valor,
        hiperparametros = hiperparametros,
        metricas = metricas,
        metricas_por_classe = metricas_por_classe
        )
    
    nome_modelo_salvo = modelo_escolhido['Modelo']

    resultado['nome_modelo_salvo'] = nome_modelo_salvo

    return caminho_pipeline, caminho_metadado, nome_modelo_salvo

def carregar_modelo(caminho_pipeline, caminho_metadado = None):

    pipeline = joblib.load(caminho_pipeline)
    print(f'Pipeline Carregado de: {caminho_pipeline}')

    metadado = None

    if caminho_metadado:
        with open(caminho_metadado, 'r', encoding = 'utf-8') as f:
            metadado = json.load(f)
        print(f'Metadado Carregado de: {caminho_metadado}')
        print(f'Modelo: {metadado["Modelo"]}')
        print(f'treinado em: {metadado["data"]}')
        print(f'Métricas: {metadado["metricas"]}')

    return pipeline, metadado
    
def salvar_associacao(df_regras, pipeline, algoritimo, parametros, pasta = 'modelos_salvos'):

    os.makedirs(pasta, exist_ok = True)

    data = datetime.now().strftime('%Y_%m_%d_%H-%M')

    nome_arquivo = f'{data}_associacao_{algoritimo}'
    
    caminho_pipeline = os.path.join(pasta, f'{nome_arquivo}_pipeline.joblib')
    caminho_regras = os.path.join(pasta, f'{nome_arquivo}_regras.csv')
    caminho_metadado = os.path.join(pasta, f'{nome_arquivo}_metadata.json')

    joblib.dump(pipeline, caminho_pipeline)
    df_regras.to_csv(caminho_regras, index = False, encoding = 'utf-8')

    print(f'Pipeline salvo em: {caminho_pipeline}')

    metadados = {
        'algoritimo': algoritimo,
        'data': data,
        'parametros': parametros,
        'resumo': {
            'regras': len(df_regras),
            'suporte_medio': float(np.round(df_regras['support'].mean(), 4)) if not df_regras.empty else None,
            'confianca_media': float(np.round(df_regras['confidence'].mean(), 4)) if not df_regras.empty else None,
            'lift_medio': float(np.round(df_regras['lift'].mean(), 2)) if not df_regras.empty else None,
            'lift_maximo': float(np.round(df_regras['lift'].max(), 2)) if not df_regras.empty else None,
        },
        'caminho_pipeline': caminho_pipeline,
        'caminho_regras': caminho_regras
    }

    with open(caminho_metadado, 'w', encoding = 'utf-8') as f:
        json.dump(metadados, f, indent = 4, ensure_ascii = False)
    
    print(f'Pipeline salvo em: {caminho_pipeline}')
    print(f'Regras salvas em: {caminho_regras}')
    print(f'Metadados salvos em: {caminho_metadado}')

    return caminho_pipeline, caminho_regras, caminho_metadado

def carregar_associacao(caminho_pipeline, caminho_regras, caminho_metadado = None):

    pipeline = joblib.load(caminho_pipeline)
    print(f'Pipeline Carregado de: {caminho_pipeline}')

    df_regras = pd.read_csv(caminho_regras, encoding = 'utf-8')

    metadado = None

    if caminho_metadado:
        with open(caminho_metadado, 'r', encoding = 'utf-8') as f:
            metadado = json.load(f)
        print(f'Metadado Carregado de: {caminho_metadado}')
        print(f'Algoritimo: {metadado["algoritimo"]}')
        print(f'treinado em: {metadado["data"]}')
        print(f'Parametros: {metadado["parametros"]}')

    return pipeline, df_regras, metadado