from src.supervisionado.classificacao.otimizacao_cla import otimizar_classificacao
from src.supervisionado.regressao.otimizacao_reg import otimizar_regressao
from src.nao_supervisionado.agrupamento.otimizacao_agrupamento import otimizar_agrupamento

def selecionar_e_otimizar(df_resultado, pipelines, nome_modelo_salvo, tipo, df = None, X_train = None, X_test = None, y_train = None, y_test = None, tempo = False):

    if nome_modelo_salvo is None:
        print('Nenhum modelo salvo anteriormente, Selecionando o top 3 melhores modelos para otimizar')
        modelos = df_resultado.head(3)['Modelo'].tolist()
    else:
        print('Escolha os Modelos para Otimizar')
        print('1 - Top 3 Melhores Modelos')
        print('2 - Modelo que Escolhi Salvar')

        opcao = input().strip()

        if opcao == '2' and nome_modelo_salvo is not None:
            modelos = [nome_modelo_salvo]
        else: 
            if opcao == '2' and nome_modelo_salvo is None:
                print('Nenhum Modelo foi Salvo, usando Top 3')
            modelos = df_resultado.head(3)['Modelo'].tolist()

    print('Modelos Selecionados para Otimizção:')

    for nome in modelos:
        print(f'{nome}')

    print('Quantos Trials Deseja Rodar ? (Enter para usar 50):')
    
    try:
        n_trials = int(input().strip())
    except ValueError:
        n_trials = 50

    pipelines_modelo = {nome: pipelines[nome] for nome in modelos if nome in pipelines}

    if tipo in ['Classificação', 'Regressão']:
            return otimizar_supervisionado(
                df_resultado, pipelines_modelo, tipo, 
                X_train, X_test, y_train, y_test,
                tempo, n_trials
            )

    elif tipo == 'Agrupamento':
            return otimizar_nao_supervisionado(
                df, df_resultado,
                pipelines_modelo, n_trials
            )
        
    else:
        raise ValueError(f'Tipo incorreto: {tipo}, deve ser Classificação, Regressão ou Agrupamento')

def otimizar_supervisionado(df_resultado, pipelines, tipo, X_train, X_test, y_train, y_test, tempo, n_trials):
    
    if tipo == 'Classificação':
        return otimizar_classificacao(
            df_ranking = df_resultado,
            pipelines = pipelines,
            X_train = X_train,
            X_test = X_test,
            y_train = y_train,
            y_test = y_test,
            tempo = tempo,
            n_trials = n_trials
        )

    if tipo == 'Regressão':
        return otimizar_regressao(
            df_ranking = df_resultado,
            pipelines = pipelines,
            X_train = X_train,
            X_test = X_test,
            y_train = y_train,
            y_test = y_test,
            tempo = tempo,
            n_trials = n_trials
        )

def otimizar_nao_supervisionado(df, df_resultado, pipelines, n_trials):

        return otimizar_agrupamento(
            df = df,
            df_resultado = df_resultado,
            pipelines = pipelines,
            n_trials = n_trials
        )