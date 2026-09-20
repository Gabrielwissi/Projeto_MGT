from utils.supervisionado.visualizacao_sup import EDA_Supervisionado
from utils.nao_supervisionado.visualizacao_nsup import EDA_Nao_Supervisionado
from utils.supervisionado.treinamento_sup import treinamento_sup
from utils.nao_supervisionado.treinamento_nsup import agrupamento, associacao
from utils.otimizacao import selecionar_e_otimizar
import utils.supervisionado.graficos_metricas_sup as msup
import utils.nao_supervisionado.graficos_metricas_nsup as mnsup
from utils.salvar_modelo import selecionar_e_salvar, salvar_associacao
from IPython.display import display


def eda(df, categoria, tipo, target_col = None, coluna_data = None, modo = 'Estático', colunas_id = None):

    if categoria == 'Supervisionado':

        if target_col is None:
            raise ValueError('Para a EDA Supervisionada, precisa passar o target_col')
        
        analisador = EDA_Supervisionado(target_col = target_col, tipo = tipo)

        if modo == 'Série Temporal':
            analisador.eda_series_temporais(df, coluna_data)
        else:
            analisador.eda_estatico(df)
    
    elif categoria == 'Não Supervisionado':

        analisador = EDA_Nao_Supervisionado(colunas_id = colunas_id)

        if modo == 'Série Temporal':
            analisador.eda_series_temporais(df, coluna_data)
        else:
            analisador.eda_estatico(df)
    
    else:
        raise ValueError('Tipo incorreto, escolha entre Supervisionado e Não Supervisionado')

def treinamento(df, itens_ignorar = None, target_col = None, tipo = 'Classificação', modo = 'Estático', colunas_log = None, nome_dataset = 'Dataset'):

    if tipo in ['Classificação', 'Regressão']:

        if target_col is None:
            raise ValueError(f'Para {tipo}, precisa de target_col')
        
        resultado = treinamento_sup(df, target_col, tipo, modo, colunas_log)

    elif tipo in ['Agrupamento', 'Associação']:

        if tipo == 'Agrupamento':
            resultado = agrupamento(df)

        elif tipo == 'Associação':
            
            print('Qual o formato dos dados ?')
            print('1 - Lista (coluna de ID + coluna e itens)')
            print('2 - Tabela (colunas binárias)\n')
            formato = input().strip()

            if formato == '1':

                formato = 'Lista'
                print(f'formato escolhido: {formato}\n')

                print('Digite a coluna de ID\n')
                coluna_id = input().strip()
                print(f'Nome da coluna_id: {coluna_id}\n')

                print('Digite a coluna de itens\n')
                coluna_item = input().strip()
                print(f'Nome da coluna de itens: {coluna_item}\n')
                
                resultado = associacao(
                    df = df,
                    formato = formato,
                    coluna_id = coluna_id,
                    coluna_item = coluna_item,
                    algoritimo = 'apriori',
                    metrica = 'confidence',
                    ordem = 'lift',
                    itens_ignorar = itens_ignorar,
                    min_support = 0.01,
                    min_confianca = 0.2
                )
            
            else:

                formato = 'Tabela'

                print('Nome das colunas de itens (separadas por vírgulas ou enter para usar todas): ')
                colunas = input().strip()

                if colunas == '':
                    colunas_itens = None
                else:
                    colunas_itens = [coluna.strip() for coluna in colunas.split(',')]

                resultado = associacao(
                    df = df,
                    formato = formato,
                    coluna_itens = colunas_itens,
                    algoritimo = 'apriori',
                    metrica = 'confidence',
                    ordem = 'lift',
                    min_suporte = 0.05,
                    min_confianca = 0.5
                )
                
    else:
        
        raise ValueError('Tipo incorreto, escolha entre Classificação, Regressão, Agrupamento, Associação')

    resultado['nome_dataset'] = nome_dataset

    return resultado

def otimizar(resultado):

    tipo = resultado.get('tipo')

    if tipo not in ['Classificação', 'Regressão', 'Agrupamento']:
        print(f'Otimização não disponível para o tipo {tipo}, disponível para Classificação, Regressão e Agrupamento')
        return resultado
    
    df_resultado = resultado.get('df_resultado')
    pipelines_treinados = resultado.get('pipelines_treinados')
    tempo = resultado.get('tempo', False)
    nome_modelo_salvo = resultado.get('nome_modelo_salvo', False)

    if tipo in ['Classificação', 'Regressão']:

        X_train = resultado.get('X_train')
        X_test = resultado.get('X_test')
        y_train = resultado.get('y_train')
        y_test = resultado.get('y_test')

        df_otimizado, pipelines_otimizados, studies = selecionar_e_otimizar(
               df_resultado = df_resultado,
               pipelines = pipelines_treinados,
               nome_modelo_salvo = nome_modelo_salvo,
               tipo = tipo,
               X_train = X_train,
               X_test = X_test,
               y_train = y_train,
               y_test = y_test,
               tempo = tempo    
          )
    
    elif tipo == 'Agrupamento':
        
        df = resultado.get('df')

        df_otimizado, pipelines_otimizados, studies = selecionar_e_otimizar(
           df = df,
           df_resultado = df_resultado,
           pipelines = pipelines_treinados,
           nome_modelo_salvo = nome_modelo_salvo,
           tipo = 'Agrupamento'
        )

    print('Modelos Otimizados')

    df_exibicao_otimizado = df_otimizado.drop(columns  = ['Metricas por classe'], errors = 'ignore')  
    display(df_exibicao_otimizado)

    resultado['df_otimizado'] = df_otimizado
    resultado['pipelines_otimizados'] = pipelines_otimizados
    resultado['studies'] = studies

    return resultado

def relatorio(resultado, tipo = None):

    tipo = tipo or resultado.get('tipo')

    df_resultado = resultado.get('df_resultado')
    df_otimizado = resultado.get('df_otimizado')

    nome_modelo, pipeline_campeao = escolher_modelo(resultado)

    if tipo == 'Classificação':
        X_train = resultado.get('X_train')
        X_test = resultado.get('X_test')
        y_test = resultado.get('y_test')

        if pipeline_campeao is None or X_test is None or y_test is None:
            print('Dados de treino e teste precisam ser passados para gerar o relatório de classificação')
            return

        print(f'Gerando o Relatório para o modelo: {nome_modelo}')

        msup.matriz_confusao(pipeline_campeao, X_test, y_test)
        msup.roc(pipeline_campeao, X_test, y_test)

        if X_train is not None:
            msup.features(pipeline_campeao, X_train, tipo)

        if df_otimizado is not None and df_resultado is not None:
            msup.comparar_classificacao(df_resultado, df_otimizado)

    elif tipo == 'Regressão':
        X_train = resultado.get('X_train')
        X_test = resultado.get('X_test')
        y_test = resultado.get('y_test')

        if pipeline_campeao is None or X_test is None or y_test is None:
            print('Dados de treino e teste precisam ser passados para gerar o relatório de regressão')
            return

        print(f'Gerando o Relatório para o modelo: {nome_modelo}')

        msup.residuos(pipeline_campeao, X_test, y_test)
        msup.predicao(pipeline_campeao, X_test, y_test)

        if X_train is not None:
            msup.features(pipeline_campeao, X_train, tipo)

        if df_otimizado is not None and df_resultado is not None:
            msup.comparar_regressao(df_resultado, df_otimizado)

    elif tipo == 'Agrupamento':

        df = resultado.get('df')

        eda = EDA_Nao_Supervisionado(max_categorias = 15)
        
        colunas_numericas, colunas_categoricas, colunas_data = eda.tipo_colunas(df)

        if pipeline_campeao is None or df is None:
            print('O pipeline e dataframe precisam ser passados para gerar o relatório de agrupamento')
            return

        print(f'Gerando o Relatório para o modelo: {nome_modelo}')

        modelo_cluster = pipeline_campeao.steps[-1][1]
        rotulos = modelo_cluster.labels_ if hasattr(modelo_cluster, 'labels_') else pipeline_campeao.predict(df)

        mnsup.cluster(df, pipeline_campeao)
        mnsup.cluster_distribuicao(pipeline_campeao, rotulos)
        mnsup.cluster_perfil(df, pipeline_campeao, rotulos, colunas_numericas)

        if df_otimizado is not None and df_resultado is not None:
            mnsup.comparar_agrupamento(df_resultado, df_otimizado)

    elif tipo == 'Associação':

        df_regras = resultado.get('df_regras')
        
        if df_regras is None or df_regras.empty:
            print('O pipeline e dataframe precisam ser passados para gerar o relatório de agrupamento')
            return

        print('Gerando o Relatório para as regras de associação')   

        mnsup.regras_associacao(df_regras)
        mnsup.regras_detalhes(df_regras)

    else:
        print(f'{tipo} não reconhecido pelo gerador de relatórios, escolha entre Classificação, Regressão, Agrupamento, Associação')

def escolher_modelo(resultado):
    
    df_resultado = resultado.get('df_resultado')
    df_otimizado = resultado.get('df_otimizado')
    pipelines_treinados = resultado.get('pipelines_treinados')
    pipelines_otimizados = resultado.get('pipelines_otimizados')
    
    if df_otimizado is not None and not df_otimizado.empty and pipelines_otimizados:
        df_escolhido = resultado.get('df_otimizado')
        pipeline_escolhido = resultado.get('pipelines_otimizados')
    
    elif df_resultado is not None and not df_resultado.empty and pipelines_treinados:
        df_escolhido = resultado.get('df_resultado')
        pipeline_escolhido = resultado.get('pipelines_treinados')

    else:
        return None, None
    
    resumo = ['Modelo']    

    if 'Acurácia Teste (%)' in df_escolhido.columns:
        resumo.append('Acurácia Teste (%)')
    elif 'R² Teste' in df_escolhido.columns:
        resumo.append('R² Teste')
    elif 'Silhouette' in df_escolhido.columns:
        resumo.append('Silhouette')

    df_resumo = df_escolhido[resumo].reset_index(drop = True)
    display(df_resumo)

    print('Deseja Gerar um Relatório para um Modelo? (s/n)')

    resposta = input().strip().lower()

    if resposta != 's':
        print('Nenhum relatório foi gerado')
        return None, None
    
    print(f'Digite o Número do Modelo que Deseja Analisar (0 a {len(df_escolhido) - 1}):')

    try:
        indice = int(input().strip())
        modelo_escolhido = df_escolhido.iloc[indice]
    except (ValueError, IndexError):
        print(f'Indice Não Encontrado!, selecione entre (0 a {len(df_escolhido) - 1})')    
        return None, None
    
    nome_modelo = modelo_escolhido['Modelo']
    pipeline = pipeline_escolhido.get(nome_modelo)

    return nome_modelo, pipeline

def salvar(resultado, etapa = None):

    tipo = resultado.get('tipo')

    if tipo is None:
        raise ValueError('O dicionário de resultados precisa conter a chave tipo')
    
    print('Iniciando o Processo de Salvamento ')

    if tipo == 'Associação':

        df_regras = resultado.get('df_regras')
        pipeline = resultado.get('pipeline')
        algoritimo = resultado.get('algoritimo')
        parametros = resultado.get('parametros', {})

        if df_regras is None or df_regras.empty:
            print('Sem regras de Assosciação para Salvar')
            return None, None, None
        
        return salvar_associacao(df_regras, pipeline, algoritimo, parametros)

    caminho_pipeline, caminho_metadado, nome_modelo_salvo = selecionar_e_salvar(
        resultado = resultado,
        tipo = tipo,
        etapa = etapa
    )

    return caminho_pipeline, caminho_metadado, nome_modelo_salvo