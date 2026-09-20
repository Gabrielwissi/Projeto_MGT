import pandas as pd
from mlxtend.preprocessing import TransactionEncoder

class Modelagem_Associacao:
    
    @staticmethod
    def transformar_lista(df, coluna_id, coluna_item):

        print(f'Convertendo para o Formato Transacional')

        transacoes = df.groupby(coluna_id)[coluna_item].apply(list).values.tolist()

        pipeline = TransactionEncoder()
        dados_ft = pipeline.fit(transacoes).transform(transacoes)

        df_lista = pd.DataFrame(dados_ft, columns = pipeline.columns_)
        df_lista.astype(int)

        return df_lista, pipeline

    @staticmethod
    def transformar_tabela(df, colunas_itens = None):

        print('Transformando as Colunas para o Formato Binário')

        df_alvo = df[colunas_itens].copy() if colunas_itens is not None else df.copy()

        df_binario = df_alvo.map(lambda x: 1 if pd.notnull(x) and isinstance(x, (int, float)) and x > 0 else 0)

        pipeline = list(df_alvo.columns)

        return df_binario, pipeline