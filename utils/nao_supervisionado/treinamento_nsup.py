from src.nao_supervisionado.agrupamento.treinamento_agrupamento import treinamento_agrupamento
from src.nao_supervisionado.associacao.treinamento_associacao import treinamento_associacao
from utils.nao_supervisionado.modelagem_agrupamento import Modelagem_Agrupamento
from utils.nao_supervisionado.modelagem_associacao import Modelagem_Associacao
from utils.nao_supervisionado.visualizacao_nsup import EDA_Nao_Supervisionado
from IPython.display import display

def agrupamento(df):

    print('Iniciando o Treinamento de Agrupamento')

    eda = EDA_Nao_Supervisionado(max_categorias = 15)
        
    colunas_numericas, colunas_categoricas, colunas_data = eda.tipo_colunas(df)

    fabrica = Modelagem_Agrupamento(colunas_numericas, colunas_categoricas, colunas_data)

    pipeline_numerico, pipeline_categorico = fabrica.pipeline()

    df_resultado, pipelines_treinados = treinamento_agrupamento(df, pipeline_numerico, pipeline_categorico)

    df_exibicao_treino = df_resultado.drop(columns  = ['Metricas por classe'], errors = 'ignore')  
    display(df_exibicao_treino)
        
    resultado = {
        'tipo': 'Agrupamento',
        'df': df,
        'df_resultado': df_resultado,
        'df_otimizado': None,
        'pipelines_treinados': pipelines_treinados,
        'pipelines_otimizados': None,
        'nome_modelo_salvo': None
    }

    return resultado
    
def associacao(df, formato, itens_ignorar = None, coluna_id = None, coluna_item = None, coluna_itens = None, 
                   algoritimo = 'apriori', metrica = 'confidence', ordem = 'lift', 
                   min_support = 0.05, min_confianca = 0.5):
        
    print(f'Iniciando regras com o {algoritimo}')

    if formato == 'Lista':
        if not coluna_id or not coluna_item:
            raise ValueError("Para o Formato Lista, Precisa passar 'coluna_id' e 'coluna_item'.")
        df_binario, pipeline = Modelagem_Associacao.transformar_lista(df, coluna_id, coluna_item)
            
    elif formato == 'Tabela':
        df_binario, pipeline = Modelagem_Associacao.transformar_tabela(df, coluna_itens)

    else:
        raise ValueError('O Tipo do Formato deve ser Lista ou Tabela')
        
    df_regras = treinamento_associacao(df_binario, itens_ignorar, algoritimo, metrica, ordem, min_support, min_confianca)

    display(df_regras)

    resultado = {
        'tipo': 'Associação',
        'df_regras': df_regras,
        'pipeline': pipeline,
        'algoritimo': algoritimo,
        'parametros': {
            'formato': formato,
            'metrica': metrica,
            'min_support': min_support,
            'min_confianca': min_confianca
        }
    }
        
    return resultado