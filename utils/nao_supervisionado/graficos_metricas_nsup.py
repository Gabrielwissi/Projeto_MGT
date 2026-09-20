import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA

def comparar_agrupamento(df_resultado, df_otimizado):

    modelos = df_otimizado['Modelo'].tolist()
    nomes = [nome.split(' + ')[0] for nome in modelos]

    df_resultado_index = df_resultado.drop_duplicates(subset = ['Modelo']).set_index('Modelo')
    df_otimizado_index = df_otimizado.set_index('Modelo')

    silhouette_antes = np.asarray(df_resultado_index.loc[modelos, 'Silhouette'].values, dtype = float)
    silhouette_depois = np.asarray(df_otimizado_index.loc[modelos, 'Silhouette'].values, dtype = float)

    calinski_antes = np.asarray(df_resultado_index.loc[modelos, 'Calinski-Harabasz'].values, dtype = float)
    calinski_depois = np.asarray(df_otimizado_index.loc[modelos, 'Calinski-Harabasz'].values, dtype = float)

    x = np.arange(len(modelos))
    largura = 0.2

    fig, axes = plt.subplots(1, 2, figsize = (16, 6))
    fig.suptitle('Agrupamento - Comparação antes e depois da otimização', fontsize = 14, fontweight = 'bold', y = 1.02)

    for ax, antes, depois, titulo, ylabel in zip(
        axes,
        [silhouette_antes, calinski_antes],
        [silhouette_depois, calinski_depois],
        ['Silhouette', 'Calinski-Harabasz'],
        ['Silhouette', 'Calinski-Harabasz']):

        ax.bar(x - largura/2, antes, largura, label = 'Antes - Modelos padrão', color = 'blue', alpha = 0.85) 
        ax.bar(x + largura/2, depois, largura, label = 'Depois - Modelos otimizados', color = 'green', alpha = 0.85)

        for i, (valor_antes, valor_depois) in enumerate(zip(antes, depois)):
            
            ganho = valor_depois - valor_antes
            pct_reducao = (ganho / abs(valor_antes)) * 100 if valor_antes != 0 else 0
            cor = 'green' if pct_reducao >= 0 else 'red'
            sinal = '+' if pct_reducao >= 0 else ''

            ax.annotate(f'{sinal}{pct_reducao:.1f}%',
                        xy = (float(x[i] + largura/2), float(valor_depois)),
                        xytext = (0, 5), textcoords = 'offset points',
                        ha = 'center', fontsize = 8, color = cor, fontweight = 'bold')
        
        ax.set_title(titulo, fontweight = 'bold', fontsize = 10)
        ax.set_xticks(x)
        ax.set_xticklabels(nomes, rotation = 15, ha = 'right')
        ax.set_ylabel(ylabel)
        ax.legend(loc = 'lower right')

        min_valor = min(float(np.min(antes)), float(np.min(depois)))
        max_valor = max(float(np.max(antes)), float(np.max(depois)))
        
        y_min = min_valor * 1.25 if min_valor < 0 else 0
        y_max = max_valor * 1.25 if max_valor > 0 else 1        
                
        ax.set_ylim(y_min, y_max)
        
        ax.grid(axis = 'y', alpha = 0.2)

    plt.tight_layout()
    plt.savefig('comparacao_agrupamento.png', dpi = 150, bbox_inches = 'tight')
    plt.show()

def cluster(df, pipeline):

    nome_reducao = None

    for passo in ['pca', 'kpca']:
        if passo in pipeline.named_steps:
            nome_reducao = passo
            break

    df_trans = df.copy()
    df_trans = pipeline[:-1].transform(df)

    if hasattr(df_trans, 'toarray'): df_trans = df_trans.toarray()
    elif hasattr(df_trans, 'values'): df_trans = df_trans.values()
    df_trans = np.asarray(df_trans)

    if nome_reducao:
        reducao = pipeline.named_steps[nome_reducao]

        if hasattr(reducao, 'explained_variance_ratio_'):
            eixo_x = f'PC1 ({reducao.explained_variance_ratio_[0] * 100:.1f}%)'
            eixo_y = f'PC2 ({reducao.explained_variance_ratio_[1] * 100:.1f}%)'
        else:
            eixo_x = f'PC1 ({nome_reducao.upper()})'
            eixo_y = f'PC2 ({nome_reducao.upper()})'
    
    else:
        if df_trans.shape[1] > 2:
            pca = PCA(n_components = 2, random_state = 42)
            df_trans = pca.fit_transform(df_trans)
            eixo_x = f'PC1 ({pca.explained_variance_ratio_[0] * 100:.1f}%)'
            eixo_y = f'PC2 ({pca.explained_variance_ratio_[1] * 100:.1f}%)'
        else:
            eixo_x, eixo_y = 'Dimensão 1', 'Dimensão 2'

    modelo_cluster = pipeline.steps[-1][1]
    rotulos = modelo_cluster.labels_ if hasattr(modelo_cluster, 'labels_') else pipeline.predict(df)

    nome_algoritimo = modelo_cluster.__class__.__name__

    rotulos_unicos = sorted(list(set(rotulos)))
    n_cluster = len(rotulos_unicos) - (1 if -1 in rotulos else 0)

    base_cores = plt.cm.tab10(np.linspace(0, 1, max(10, n_cluster)))

    fig, ax = plt.subplots(figsize = (10, 7))

    for i, cluster in enumerate(rotulos_unicos):
        mascara = (rotulos == cluster)
        label = f'Cluster {cluster}' if cluster != -1 else 'Ruído'
        cor = 'red' if cluster == -1 else base_cores[i % 10]

        ax.scatter(df_trans[mascara, 0], df_trans[mascara, 1], c =[cor], s = 35, alpha = 0.6, 
                   edgecolors = 'white', linewidth = 0.2, label = label)

    ax.set_title(f'Visualização dos Clusters - {nome_algoritimo}')
    ax.set_xlabel = (eixo_x)
    ax.set_ylabel = (eixo_y)
    ax.legend(bbox_to_anchor = (1.05, 1), loc = 'upper left')
    ax.grid(alpha = 0.2)
    
    plt.tight_layout()
    plt.savefig('cluster.png', dpi = 150, bbox_inches = 'tight')
    plt.show()    
    
def cluster_distribuicao(pipeline, rotulos):

    cluster, contagem = np.unique(rotulos, return_counts = True)
    labels = [f'Cluster {c}' if c != -1 else 'Ruído' for c in cluster]

    base_cores = plt.cm.tab10(np.linspace(0, 1, max(10, len(cluster))))
    cores = ['red' if c == -1 else base_cores[i % 10] for i, c in enumerate(cluster)]

    nome_algoritimo = pipeline.steps[-1][1].__class__.__name__

    fig, axes = plt.subplots(1, 2, figsize = (14, 5))

    fig.suptitle(f'Distribuição dos Clusters - {nome_algoritimo}', fontsize = 14, fontweight = 'bold', y = 1.05)
    bars = axes[0].bar(labels, contagem, color = cores, alpha = 0.85, edgecolor = 'white')
    axes[0].bar_label(bars, fmt = '%d', color = 'white', fontweight = 'bold', padding = 2, label_type = 'center')
    axes[0].set_title('Contagem Absoluta por Cluster', fontweight = 'bold')
    axes[0].set_xlabel('Grupos Identificados')
    axes[0].set_ylabel('Quantidade de Amostras')
    axes[0].tick_params(axis = 'x', rotation = 45)
    axes[0].grid(False)
    
    axes[1].pie(contagem, labels = labels, colors = cores,
                textprops = {'color': 'white'},
                autopct = '%1.1f%%', startangle = 90,
                wedgeprops = {'edgecolor': 'white', 'linewidth': 2})
    axes[1].set_title('Proporção Relativa (%)', fontweight = 'bold')

    plt.tight_layout()
    plt.savefig('cluster_distribuicao.png', dpi = 150, bbox_inches = 'tight')
    plt.show()

def cluster_perfil(df, pipeline, rotulos, colunas_numericas):

    colunas_validas = []

    for coluna in colunas_numericas:

        qtd_unicos = df[coluna].nunique()

        nome_id = any(palavra in coluna.lower() for palavra in ['id', 'codigo', 'code'])
            
        if (qtd_unicos > (len(df) * 0.9) and nome_id) or (qtd_unicos == len(df) and nome_id):
            print(f'Ignorando {coluna}, pois parece ser uma coluna de ID')
            print('-' * 50)
        else:
            colunas_validas.append(coluna)

    if not colunas_validas:
        print('Nenhuma coluna numérica válida para análise de perfil')
        return

    df_perfil = df[colunas_validas].copy()
    df_perfil['Cluster'] = rotulos
    df_perfil = df_perfil[df_perfil['Cluster'] != -1]

    if df_perfil.empty or len(df_perfil['Cluster'].unique()) <= 1:
        print('Dados Insuficientes ou apenas Ruidos detectados')
        return

    medias_absolutas = df_perfil.groupby('Cluster')[colunas_validas].mean()

    medias_populacional = df[colunas_validas].mean()
    std_devs = df[colunas_validas].std().replace(0, 1.0).fillna(1.0)
    
    medias_norm = (medias_absolutas - medias_populacional) / std_devs

    nome_algoritimo = pipeline.steps[-1][1].__class__.__name__

    fig, axes = plt.subplots(1, 2, figsize = (18, 6))

    fig.suptitle(f'Análise de Perfil dos Clusters - {nome_algoritimo}')

    sns.heatmap(medias_norm.T, annot = True, fmt = '.2f', cmap = 'Greens',
                ax = axes[0], linewidth = 0.5)
    axes[0].set_title('Comportamento Relativo (Z-Score por Feature)', fontweight = 'bold')
    axes[0].set_xlabel('Cluster')
    axes[0].set_ylabel('Features Analisadas')

    sns.heatmap(medias_absolutas.T, annot = True, fmt = '.2f', cmap = 'Blues', center = 0,
                ax = axes[1], linewidth = 0.5)
    axes[1].set_title('Valores absolutos Médios', fontweight = 'bold')
    axes[1].set_xlabel('Cluster')
    axes[1].set_ylabel('Features Analisadas')

    plt.tight_layout()
    plt.savefig('cluster_perfil.png', dpi = 150, bbox_inches = 'tight')
    plt.show()

def optuna_agrupamento(pipeline, study):

    df = study.trials_dataframe()

    trials_df = df[df['state'] == 'COMPLETE'].copy()

    if trials_df.empty:
        print('Nenhum Trial com status COMPLETE foi encontrado')
        return
    
    direcao = study.direction.name

    if direcao == 'MAXIMIZE':
        curva_melhor = trials_df['value'].cummax()
        label_melhor = 'Melhor Acumulado (Máx)'
    else:
        curva_melhor = trials_df['value'].cummin()
        label_melhor = 'Melhor Acumulado (Mín)'

    nome_modelo = pipeline.steps[-1][1].__class__.__name__

    nome_limpo = nome_modelo.split(' + ')[0]
    
    fig, axes = plt.subplots(1, 2, figsize = (16, 5))
    fig.suptitle(f'Histórico de Otimização Agrupamento - {nome_limpo}', fontsize = 14, fontweight = 'bold', y = 1.05)

    axes[0].plot(trials_df['number'], trials_df['value'], color = 'green', linewidth = 1, alpha = 0.5, label = 'Trial Individual')
    axes[0].plot(trials_df['number'], curva_melhor, color = 'blue', linewidth = 2.5, label = label_melhor)
    axes[0].axhline(y = study.best_value, color = 'red', linestyle = '--', linewidth = 2, label = f'Melhor Geral - {study.best_value:.4f}')
    axes[0].set_title('Conergência dos Trials', fontweight = 'bold', pad = 10)
    axes[0].set_xlabel('Número do Trial')
    axes[0].set_ylabel('Score da Métrica (CV)')
    axes[0].legend(loc = 'best')
    axes[0].grid(alpha = 0.2)

    axes[1].hist(trials_df['value'], bins = 20, color = 'green', alpha = 0.85, edgecolor = 'white')
    axes[1].axvline(x = study.best_value, color = 'red', linestyle = '--', linewidth = 2, label = f'Melhor Geral - {study.best_value:.4f}')
    axes[1].set_title('Distribuição dos Scores', fontweight = 'bold', pad = 10)
    axes[1].set_xlabel('Score da Métrica (CV)')
    axes[1].set_ylabel('Frequência de Ocorrências')
    axes[1].legend(loc = 'best')
    axes[1].grid(alpha = 0.2)

    plt.tight_layout()
    plt.savefig(f'optuna_{nome_limpo}.png', dpi = 150, bbox_inches = 'tight')
    plt.show()

def regras_associacao(df_regras, top_n = 20):

    if df_regras.empty:
        print('Nenhuma regra')
        return
    
    fig, axes = plt.subplots(1, 2, figsize = (16, 6))
    fig.suptitle('Análise das Regras de Associação', fontsize = 14, fontweight ='bold', y = 1.05)

    scatter = axes[0].scatter(
        df_regras['support'],
        df_regras['confidence'],
        s = df_regras['lift'] * 35,
        c = df_regras['lift'],
        cmap = 'YlOrRd',
        alpha = 0.7,
        edgecolors = 'white',
        linewidth = 0.5
    )

    fig.colorbar(scatter, ax = axes[0], label = 'Força da Regra (Lift)')
    axes[0].set_title('Dispersão: Suporte vc Confiança', fontweight = 'bold')
    axes[0].set_xlabel('Suporte')
    axes[0].set_ylabel('Confiança')
    axes[0].grid(alpha = 0.2)

    top_regras = df_regras.head(top_n).copy()

    antecedentes = top_regras['antecedents'].apply(lambda x: ', '.join(list(x))).astype(str)
    consequentes = top_regras['consequents'].apply(lambda x: ', '.join(list(x))).astype(str)
    top_regras['regra'] = antecedentes + ' -> ' + consequentes

    bars = axes[1].barh(range(len(top_regras)), top_regras['lift'],
                        color = plt.cm.YlOrRd(top_regras['lift'] / top_regras['lift'].max()),
                        alpha = 0.85, edgecolor = 'white')
    
    axes[1].set_yticks(range(len(top_regras)))
    axes[1].set_yticklabels(top_regras['regra'], fontsize = 10)
    axes[1].invert_yaxis()
    axes[1].set_title(f'Top {top_n} Associações por Força de Acoplamento (Lift)', fontweight = 'bold')
    axes[1].set_xlabel('Metrica Lift')
    axes[1].grid(axis ='x', alpha = 0.2)

    plt.tight_layout()
    plt.savefig(f'regras_associacao.png', dpi = 150, bbox_inches = 'tight')
    plt.show()

def regras_detalhes(df_regras, top_n = 20):

    if df_regras.empty:
        print('Nenhuma regra')
        return
    
    top_regras = df_regras.head(top_n).copy()

    antecedentes = top_regras['antecedents'].apply(lambda x: ', '.join(list(x))).astype(str)
    consequentes = top_regras['consequents'].apply(lambda x: ', '.join(list(x))).astype(str)
    top_regras['regra'] = antecedentes + ' -> ' + consequentes

    metricas  = ['support', 'confidence', 'lift']
    titulos   = ['Suporte', 'Confiança', 'Lift']
    cores      = ['#5B9BD5', '#70AD47', '#ED7D31']

    fig, axes = plt.subplots(1, 3, figsize = (26, 9))
    fig.suptitle(f'Top {top_n} Regras de Negócio', fontweight = 'bold', y = 1.05)

    for i, (ax, metrica, titulo, cor) in enumerate(zip(axes, metricas, titulos, cores)):
        bars = ax.barh(range(len(top_regras)), top_regras[metrica],
                        color = cor, alpha = 0.85, edgecolor = 'white')
        ax.set_yticks(range(len(top_regras)))

        if i == 0:
            ax.set_yticklabels(top_regras['regra'], fontsize = 10)
        else:
            ax.set_yticklabels([])
        
        ax.invert_yaxis()
        ax.set_title(titulo, fontweight = 'bold')
        ax.set_xlabel(titulo)

        max_valor = top_regras[metrica].max()
        ax.set_xlim(0, max_valor * 1.2)

        ax.bar_label(bars, fmt = '%.4f', padding = 2, fontsize = 6)
        ax.grid(axis ='x', alpha = 0.2)

    plt.tight_layout()
    plt.savefig(f'regras_detalhes.png', dpi = 150, bbox_inches = 'tight')
    plt.show()