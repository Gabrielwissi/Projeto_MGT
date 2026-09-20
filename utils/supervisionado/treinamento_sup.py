from sklearn.model_selection import train_test_split
from utils.supervisionado.visualizacao_sup import EDA_Supervisionado
from utils.supervisionado.modelagem_sup import Modelagem_Supervisionada
from src.supervisionado.classificacao.treinamento_cla import treinamento_classificacao
from src.supervisionado.regressao.treinamento_reg import treinamento_regressao
from IPython.display import display

def treinamento_sup(df, target_col, tipo, modo, colunas_log):
        
        eda = EDA_Supervisionado(target_col = target_col, tipo = tipo)

        colunas_numericas, colunas_categoricas, colunas_data = eda.tipo_colunas(df)

        previsores = df.drop(columns = [target_col])
        alvo = df[target_col]

        if modo == 'Série Temporal':
             print('Aplicando o Split Cronológico para a Série Temporal')
             linha_corte = int(len(previsores) * 0.7)
             X_train, X_test = previsores.iloc[:linha_corte], previsores.iloc[linha_corte:]
             y_train, y_test = alvo.iloc[:linha_corte], alvo.iloc[linha_corte:]
             tempo = True

        else:
             print(f'Aplicando o Split Estatístico Aleatório para {tipo}')
             tempo = False

             if tipo == 'Classificação':
                X_train, X_test, y_train, y_test = train_test_split(
                     previsores, alvo, test_size = 0.3, random_state = 42, shuffle = True, stratify = alvo)

             else:
                X_train, X_test, y_train, y_test = train_test_split(
                     previsores, alvo, test_size = 0.3, random_state = 42, shuffle = True, stratify = None) 

        print('Construindo os Pipelines')

        fabrica = Modelagem_Supervisionada(target_col, colunas_numericas, colunas_categoricas, colunas_data, tipo, colunas_log)
        pipelines = fabrica.pipeline()

        print('Treinando os modelos')

        if tipo == 'Classificação':     
          df_resultado, pipelines_treinados = treinamento_classificacao(
          pipelines = pipelines,
          X_train = X_train, y_train = y_train,
          X_test = X_test, y_test = y_test,
          tempo = tempo)
        
        elif tipo == 'Regressão':
          df_resultado, pipelines_treinados = treinamento_regressao(
          pipelines = pipelines,
          X_train = X_train, y_train = y_train,
          X_test = X_test, y_test = y_test,
          tempo = tempo)

        df_exibicao_treino = df_resultado.drop(columns  = ['Metricas por classe', 'tempo'], errors = 'ignore')  
        display(df_exibicao_treino)

        resultado = {
          'tipo': tipo,
          'target_col': target_col,
          'df': df,
          'df_resultado': df_resultado,
          'df_otimizado': None,
          'pipelines_treinados': pipelines_treinados,
          'pipelines_otimizados': None,
          'X_train': X_train,
          'X_test': X_test,
          'y_train': y_train,
          'y_test': y_test,
          'tempo': tempo,
          'nome_modelo_salvo': None
        }

        return resultado