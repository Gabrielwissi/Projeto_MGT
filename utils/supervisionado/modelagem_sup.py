import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, PowerTransformer, OrdinalEncoder, OneHotEncoder, FunctionTransformer
from sklearn.pipeline import Pipeline
from utils.supervisionado.modelos_sup import modelos_supervisionados
from utils.modelos import montar_modelo

class Modelagem_Supervisionada:
    
    def __init__(self, target_col, colunas_numericas, colunas_categoricas, colunas_data, tipo, colunas_log = None):

        if tipo not in ['Classificação', 'Regressão']:
            raise ValueError("Tipo incorreto, Escolha entre 'Classificação' ou 'Regressão'.")
        
        self.target_col = target_col
        self.colunas_numericas = colunas_numericas
        self.colunas_categoricas = colunas_categoricas
        self.colunas_data = colunas_data
        self.tipo = tipo

        self.colunas_log = colunas_log if colunas_log is not None else []
        self.transformar_target = self.target_col in self.colunas_log
        self.colunas_log_x = [coluna for coluna in self.colunas_log if coluna != self.target_col]

        self.pipelines = {}

    def extrair_datas(self, previsores):

        df_previsores = pd.DataFrame(index = previsores.index)

        for coluna in self.colunas_data:

            datas = pd.to_datetime(previsores[coluna])

            df_previsores[f'{coluna}_ano'] = datas.dt.year
            df_previsores[f'{coluna}_mes'] = datas.dt.month
            df_previsores[f'{coluna}_dia'] = datas.dt.day
            df_previsores[f'{coluna}_dia_semana'] = datas.dt.dayofweek       
            df_previsores[f'{coluna}_trimestre'] = datas.dt.quarter

        return df_previsores
    
    def _pipeline_numerico(self):
        
        colunas_normais = [coluna for coluna in self.colunas_numericas if coluna not in self.colunas_log_x]

        transformers = []

        if colunas_normais:
            transformers.append(('numero_padrao', StandardScaler(), colunas_normais))

        if self.colunas_log_x:
            pipeline_log =  Pipeline(
                [
                    ('power_transform', PowerTransformer(method = 'yeo-johnson')),
                    ('scaler', StandardScaler())
                ]
            )
            transformers.append(('numero_log', pipeline_log, self.colunas_log_x))

        return transformers                       

    def transformadores(self):

        tempo = Pipeline(steps = [
            ('extrator', FunctionTransformer(self.extrair_datas)),
            ('normalizador', StandardScaler())
        ])

        data = len(self.colunas_data) > 0
        categorica = len(self.colunas_categoricas) > 0
        passos_numericos = self._pipeline_numerico()

        if not categorica:
            passos_padrao = passos_numericos.copy()

            if data:
                passos_padrao.append(('tempo', tempo, self.colunas_data))

            padrao = ColumnTransformer(transformers = passos_padrao, remainder = 'drop')

            engenharia_dados = {'padrao': padrao}

            return engenharia_dados

        passos_ordinal = passos_numericos + [
            ('categorico', OrdinalEncoder(handle_unknown = 'use_encoded_value', unknown_value = -1), self.colunas_categoricas)
        ]

        passos_onehot = passos_numericos + [
                ('categorico', OneHotEncoder(handle_unknown = 'ignore', drop = 'first'), self.colunas_categoricas)
        ]

        if data:
            passos_ordinal.append(('tempo', tempo, self.colunas_data))
            passos_onehot.append(('tempo', tempo, self.colunas_data))
        
        ordinal = ColumnTransformer(transformers = passos_ordinal, remainder = 'drop')
        onehot = ColumnTransformer(transformers = passos_onehot, remainder = 'drop')   

        engenharia_dados = {
            'ordinal': ordinal,
            'onehot': onehot
        }

        return engenharia_dados
    
    def pipeline(self):

        modelos, reducoes = modelos_supervisionados(self.tipo)
        engenharia_dados = self.transformadores()

        for nome_modelo, info in modelos.items():
            for nome_engenharia, engenharia in engenharia_dados.items():
                for nome_reducao, reducao in reducoes.items():

                    modelo = montar_modelo(
                        info,
                        transformar_target = (self.tipo == 'Regressão' and self.transformar_target)
                        )

                    nomes = [nome_modelo, nome_engenharia]

                    logs = []

                    if len(self.colunas_log_x) > 0:
                        logs.append('log_X')
                    if self.transformar_target:
                        logs.append('log_y')

                    if logs:
                        nomes.append('_'.join(logs))

                    passos = [('Engenharia', clone(engenharia))]       

                    if reducao is not None:
                        nomes.append(nome_reducao)
                        nome_passo_reducao = nome_reducao.replace(' + ', '').lower()
                        passos.append((nome_passo_reducao, clone(reducao)))

                    passos.append(('Modelo', modelo))

                    nome = ' + '.join(nomes)

                    self.pipelines[nome] = Pipeline(steps = passos)

        return self.pipelines

