import pandas as pd
import re
import matplotlib.pyplot as plt
import seaborn as sns

class EDA_Nao_Supervisionado:
        
    def __init__(self, max_categorias = 15, colunas_id = None):

        self.max_categorias = max_categorias

        if colunas_id is None:
            self.colunas_id = []
        elif isinstance(colunas_id, str):
            self.colunas_id = [colunas_id]
        else:
            self.colunas_id = colunas_id

        sns.set_theme(style = 'whitegrid')

    def tipo_colunas(self, df):
        
        features = df.copy()
        
        colunas_numericas = []
        colunas_categoricas = []
        colunas_data = []

        for coluna in features.columns:

            if hasattr(self, 'colunas_id') and coluna in self.colunas_id:
                continue

            if pd.api.types.is_datetime64_any_dtype(features[coluna]) or \
                re.search(r'data|date|time', coluna.lower()):
                colunas_data.append(coluna)
                continue
            
            if pd.api.types.is_object_dtype(features[coluna]) or \
            pd.api.types.is_string_dtype(features[coluna]) or \
            isinstance(features[coluna].dtype, pd.CategoricalDtype):

                amostra = features[coluna].dropna().head(5)

                data = False

                if not amostra.empty:
                    data_padrao = r'^\d{2,4}[-/\.]\d{1,2}[-/\.]\d{2,4}'
                    data_valida = amostra.astype(str).str.match(data_padrao).all()

                    if data_valida:
                        try:
                            pd.to_datetime(amostra, errors = 'coerce', format = 'mixed')
                            data = True
                            
                        except (ValueError, TypeError):
                            data = False

                if data:
                    colunas_data.append(coluna)
                else:
                    colunas_categoricas.append(coluna)
                        
            elif pd.api.types.is_numeric_dtype(features[coluna]):

                if features[coluna].nunique() > 1:
                    colunas_numericas.append(coluna)
                else:
                    colunas_categoricas.append(coluna)

            else:
                colunas_categoricas.append(coluna)

        return colunas_numericas, colunas_categoricas, colunas_data

    def validar_e_preparar(self, df, coluna, tipo_coluna = 'categorica'):

        if coluna in self.colunas_id:
            print(f'Ignorando {coluna}, Informada pelo usuário como uma coluna de ID')
            print('-' * 50)
            return False, None, None

        qtd_unicos = df[coluna].nunique()
        nome_id = any(palavra in coluna.lower() for palavra in ['id', 'codigo', 'code'])
                
        if (qtd_unicos > (len(df) * 0.9) and nome_id) or (qtd_unicos == len(df) and nome_id):
            print(f'Ignorando {coluna}, pois parece ser uma coluna de ID')
            print('-' * 50)
            return False, None, None

        print(f'Variável {coluna}\n')

        if tipo_coluna == 'numerica':
            print(f'{df[coluna].describe().to_string()}\n')
            print(f'Valores nulos: {df[coluna].isna().sum()} ({df[coluna].isna().mean() * 100:.2f}%)\n')
            print('-' * 50)
            return True, df, None

        elif tipo_coluna == 'categorica':
            print(f'Total de categorias {qtd_unicos}\n')
            print(f'Valores nulos: {df[coluna].isna().sum()} ({df[coluna].isna().mean():.2f}%)\n')
            print(f'Top 5 categorias mais frequentes\n')
            print(f'{df[coluna].value_counts().head().to_string()}\n')
            print('-' * 50)
            
            if qtd_unicos > self.max_categorias:
                print(f'{coluna} tem muitas categorias ({qtd_unicos}). Plotando apenas as {self.max_categorias} mais frequentes')
            
                top_categorias = df[coluna].value_counts().index[:self.max_categorias]   
                df_filtrado = df[df[coluna].isin(top_categorias)]
            
            else:
                top_categorias = df[coluna].value_counts().index
                df_filtrado = df.copy()
            
            return True, df_filtrado, top_categorias

    def eda_num_estatico(self, df, coluna):

        valido, _, _ = self.validar_e_preparar(df, coluna, tipo_coluna = 'numerica')
        if not valido:
            return None

        fig, axes = plt.subplots(1, 2, figsize = (14, 5))

        sns.histplot(data = df, x = coluna, kde = True, ax = axes[0], color = 'green')
        axes[0].set_title(f'Distribuição de {coluna}')

        sns.boxplot(data = df, x = coluna, ax = axes[1], color = 'green')
        axes[1].set_title(f'Boxplot de {coluna}')

        caminho = f'eda_num{coluna}.png'
        plt.tight_layout()
        plt.savefig(caminho, dpi = 150)
        plt.show()
        plt.close()
        return caminho
        
    def eda_cat_estatico(self, df, coluna):

        valido, df_filtrado, top_categorias = self.validar_e_preparar(df, coluna, tipo_coluna = 'categorica')
        if not valido:
            return None

        fig, ax = plt.subplots(1, 1, figsize = (14, 5))
        
        sns.countplot(data = df_filtrado, x = coluna, ax = ax, order = top_categorias)
        ax.set_title(f'Frequência geral de {coluna}')

        if hasattr(ax, 'containers') and len(ax.containers) > 0:
            for container in ax.containers:
                ax.bar_label(container, fmt = '%d', label_type = 'edge', padding = 3)
            ax.tick_params(axis = 'x', rotation = 45)
        
        caminho = f'eda_cat{coluna}.png'
        plt.tight_layout()
        plt.savefig(f'analise_{coluna}.png', dpi = 150)
        plt.show()
        plt.close()
        return caminho

    def eda_num_tempo(self, df, coluna, coluna_data, agregacao = 'mean'):
            
        valido, _, _ = self.validar_e_preparar(df, coluna, tipo_coluna = 'numerica')
        if not valido:
            return None

        df_temp = df.copy()
        df_temp['Periodo'] = df_temp[coluna_data].dt.to_period('M').dt.to_timestamp()
        df_agrupado = df_temp.groupby('Periodo')[coluna].agg(agregacao).reset_index()

        plt.figure(figsize = (14, 5))
        sns.lineplot(data = df_agrupado, x = 'Periodo', y = coluna, marker = 'o', color = 'green')
        label_agg = 'Média' if agregacao == 'mean' else 'Soma'
        plt.title(f'Evolução Temporal ({label_agg}) de {coluna}', weight = 'bold')
        plt.xlabel('Período', weight = 'bold')
        plt.ylabel(f'{label_agg} de {coluna}', weight = 'bold')
        plt.grid(False)

        caminho = f'eda_num_temp{coluna}.png'
        plt.tight_layout()
        plt.savefig(caminho, dpi = 150)
        plt.show()
        plt.close()
        return caminho        
                                     

    def eda_cat_tempo(self, df, coluna, coluna_data):
        
        valido, df_filtrado, top_categorias = self.validar_e_preparar(df, coluna, tipo_coluna = 'categorica')
        if not valido:
            return None

        col_ano = f'{coluna_data}_ano'
        col_ano_mes = f'{coluna_data}_ano_mes'

        if col_ano not in df_filtrado.columns:
            df_filtrado[f'{coluna_data}_ano'] = df_filtrado[coluna_data].dt.year
        if col_ano_mes not in df_filtrado.columns:
            df_filtrado[f'{coluna_data}_ano_mes'] = df_filtrado[coluna_data].dt.to_period('M').astype(str)

        matriz_mensal = df_filtrado.groupby([col_ano_mes, coluna]).size().unstack(fill_value = 0)
        matriz_mensal = matriz_mensal.reindex(columns = [coluna for coluna in top_categorias if coluna in matriz_mensal.columns])

        fig, axes = plt.subplots(2, 1, figsize = (14, 10))

        linhas_categorias = top_categorias[:5]
        cores = sns.color_palette('tab10', n_colors = len(linhas_categorias))

        for idx, coluna_cat in enumerate(linhas_categorias):
            if coluna_cat in matriz_mensal.columns:
                axes[0].plot(matriz_mensal.index, matriz_mensal[coluna_cat], marker = 'o', linewidth = 2, label = coluna_cat, color = cores[idx])

        axes[0].set_title(f'Tendência Mensal: Top 5 categorias em {coluna}', weight = 'bold')
        axes[0].set_ylabel('Frequência', weight = 'bold')
        axes[0].grid(True, linestyle = '--', alpha = 0.2)
        axes[0].tick_params(axis = 'x', rotation = 45)
        axes[0].legend(title = coluna, bbox_to_anchor = (1.02, 1), loc = 'upper left', frameon = False)

        sns.heatmap(matriz_mensal.T, cmap = 'YlGnBu', annot = True, fmt = 'd', cbar = True, ax = axes[1], linewidths = 0.5)
        axes[1].set_title(f'Volume Mensal: {coluna}', weight = 'bold')
        axes[1].set_xlabel('Mês/Ano', weight = 'bold')
        axes[1].set_ylabel(coluna, weight = 'bold')
        axes[1].tick_params(axis = 'x', rotation = 45)

        caminho = f'eda_cat_tempo{coluna_data}.png'
        plt.tight_layout()
        plt.savefig(caminho, dpi = 150)
        plt.show()
        plt.close()
        return caminho

    def eda_estatico(self, df):

        sns.set_theme(style = 'whitegrid')

        colunas_numericas, colunas_categoricas, colunas_data = self.tipo_colunas(df)

        print(f'Análise exploratória\n')
        print(f'{len(colunas_numericas)} colunas numéricas')
        print(f'{colunas_numericas}\n')
        print(f'{len(colunas_categoricas)} colunas categóricas')
        print(f'{colunas_categoricas}\n')
        if colunas_data:
            print(f'{len(colunas_data)} colunas de tempo')
            print(f'{colunas_data}\n')

        if colunas_numericas:
            print('Resumo e Análise das Variáveis Numéricas\n') 
            for coluna in colunas_numericas:
                self.eda_num_estatico(df, coluna)

        if colunas_categoricas:
            print('Resumo e Análise das Variáveis Categóricas\n')
            for coluna in colunas_categoricas:
                self.eda_cat_estatico(df, coluna)

    def eda_series_temporais(self, df, coluna_data):

        sns.set_theme(style = 'whitegrid')

        print(f'EDA da Série Temporal na coluna {coluna_data}')

        colunas_numericas, colunas_categoricas, colunas_data = self.tipo_colunas(df)

        print(f'Análise exploratória\n')
        print(f'{len(colunas_numericas)} colunas numéricas')
        print(f'{colunas_numericas}\n')
        print(f'{len(colunas_categoricas)} colunas categóricas')
        print(f'{colunas_categoricas}\n')
        if colunas_data:
            print(f'{len(colunas_data)} colunas de tempo')
            print(f'{colunas_data}\n')

        df_data = df.copy()
        df_data[coluna_data] = pd.to_datetime(df_data[coluna_data], dayfirst = True, errors = 'coerce')
        df_data[f'{coluna_data}_ano'] = df_data[coluna_data].dt.year
        df_data[f'{coluna_data}_ano_mes'] = df_data[coluna_data].dt.to_period('M').astype(str)

        df_volume = df_data.groupby(df_data[coluna_data].dt.to_period('M').dt.to_timestamp()).size().reset_index(name = 'Volume')

        plt.figure(figsize = (14, 5))
        sns.lineplot(data = df_volume, x = coluna_data, y = 'Volume', marker = 'o', color = 'green', linewidth = 2)
        plt.title('Volume de Registros Coletados por Ano/Mês', weight = 'bold')
        plt.xlabel('Período')
        plt.ylabel('Registros')
        plt.xticks(rotation = 45)
        plt.grid(False)

        caminho = f'eda_volume_geral_{coluna_data}.png'
        plt.tight_layout()
        plt.savefig(caminho, dpi = 150)
        plt.show()
        plt.close()

        if colunas_numericas:
                    
            print('Analisando as Variáveis Numéricas ao longo tempo')
        
            for coluna in colunas_numericas:
                self.eda_num_tempo(df_data, coluna = coluna, coluna_data = coluna_data, agregacao = 'mean')

        if colunas_categoricas:
            
            print('Analisando as Variáveis Categóricas ao longo tempo')

            for coluna in colunas_categoricas:
                self.eda_cat_tempo(df_data, coluna = coluna, coluna_data = coluna_data)

