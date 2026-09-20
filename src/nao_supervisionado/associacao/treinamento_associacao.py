import numpy as np
import time
from mlxtend.frequent_patterns import apriori, association_rules, fpgrowth

def treinamento_associacao(df, itens_ignorar = None, algoritimo = 'apriori', metrica = 'confidence', ordem = 'lift', min_support = 0.01, min_confianca = 0.2):

    df_filtrado = df.copy()

    if itens_ignorar:
        if isinstance(itens_ignorar, str):
            itens_ignorar = [itens_ignorar]
        colunas_ignorar = [coluna for coluna in itens_ignorar if coluna in df_filtrado.columns]
        if colunas_ignorar:
            df_filtrado = df_filtrado.drop(columns = colunas_ignorar)
            print(f'Ignorando as colunas: {colunas_ignorar}')

    inicio = time.time()

    if algoritimo == 'apriori':
        itens_frequentes = apriori(df_filtrado, min_support = min_support, use_colnames = True)

    elif algoritimo in ['eclat', 'fpgrowth']:
        itens_frequentes = fpgrowth(df_filtrado, min_support = min_support, use_colnames = True)
    else:
        raise ValueError("Algoritimo incorreto. Escolha entre 'apriori' ou 'eclat'.")
    
    tempo_execucao = time.time() - inicio
    
    print(f'{len(itens_frequentes)} itens frequentes encontrados.')

    regras = association_rules(itens_frequentes, metric = metrica, min_threshold = min_confianca)

    if regras.empty:
        print('Nenhuma Regra de Associação Passou pelo Filtro de Confiança mínimo.')

    regras = regras.sort_values(by = ordem, ascending = False).reset_index(drop = True)

    regras = regras.round({
        'support': 4,
        'confidence': 4,
        'lift': 2,
        'leverage': 4,
        'conviction': 2
    })

    regras['Tempo'] = np.round(tempo_execucao, 4)
    regras['Algoritimo'] = algoritimo

    print(f'Sucesso: {len(regras)} Regras Geradas em {tempo_execucao:.4f}s')

    return regras
