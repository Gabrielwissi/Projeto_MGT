import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer
from sklearn.pipeline import Pipeline
from utils.nao_supervisionado.modelos_nsup import modelos_agrupamento
from utils.modelos import montar_modelo
from sklearn.base import clone

class Modelagem_Agrupamento:
    
    def __init__(self, colunas_numericas, colunas_categoricas, colunas_data):
        
        self.colunas_numericas = colunas_numericas
        self.colunas_categoricas = colunas_categoricas
        self.colunas_data = colunas_data

        self.pipelines_numericas = {}
        self.pipelines_categorias = {}

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

    def transformadores(self):

        tempo = Pipeline(steps = [
            ('extrator', FunctionTransformer(self.extrair_datas)),
            ('normalizador', StandardScaler())
        ])

        data = len(self.colunas_data) > 0

        passos_numericos = [
                ('numerico', StandardScaler(), self.colunas_numericas),
                ('categorico', OneHotEncoder(handle_unknown = 'ignore', drop = 'first', sparse_output = False), self.colunas_categoricas)
        ]

        passos_categoricos = [
                ('numerico', StandardScaler(), self.colunas_numericas),
                ('categorico', FunctionTransformer(lambda x: x, feature_names_out = 'one-to-one'), self.colunas_categoricas)
        ]

        if data:
            passos_numericos.append(('tempo', tempo, self.colunas_data))
            passos_categoricos.append(('tempo', tempo, self.colunas_data))
        
        numerico = ColumnTransformer(transformers = passos_numericos, remainder = 'drop')
        categorica = ColumnTransformer(transformers = passos_categoricos, remainder = 'drop')   

        engenharia_dados = {
            'numerico': numerico,
            'misto': categorica
        }

        return engenharia_dados
    
    def pipeline(self):

        modelos_numericos, modelos_categoricos, reducoes = modelos_agrupamento()
        engenharia_dados = self.transformadores()

        engenharia_numerica = engenharia_dados['numerico']
        engenharia_mista = engenharia_dados['misto']

        for nome_modelo, info in modelos_numericos.items():

            modelo = montar_modelo(info)
            
            for nome_reducao, reducao in reducoes.items():

                nome = f'{nome_modelo} + {nome_reducao}'

                passos = [('engenharia', clone(engenharia_numerica))]

                if reducao is not None:
                    nome_passo_reducao = nome_reducao.replace(' + ', '').lower()
                    passos.append((nome_passo_reducao, reducao))

                passos.append(('Modelo', modelo))

                self.pipelines_numericas[nome] = Pipeline(steps = passos)
        
        for nome_modelo, info in modelos_categoricos.items():

            modelo = montar_modelo(info)

            nome = f'{nome_modelo} + Dados mistos'

            passos = [('engenharia', clone(engenharia_mista))]

            passos.append(('Modelo', modelo))

            self.pipelines_categorias[nome] = Pipeline(steps = passos)

        return self.pipelines_numericas, self.pipelines_categorias