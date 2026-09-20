import pandas as pd
import re
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

class EDA_Supervisionado:

    def __init__(self, target_col, tipo, max_categorias = 15):
        self.target_col = target_col
        self.tipo = tipo
        self.max_categorias = max_categorias

        sns.set_theme(style = 'whitegrid')

    def tipo_colunas(self, df):
        
        features = df.drop(columns = [self.target_col])
        
        colunas_numericas = []
        colunas_categoricas = []
        colunas_data = []

        for coluna in features.columns:

            if pd.api.types.is_datetime64_any_dtype(features[coluna]) or \
                re.search(r'data|date|time', coluna.lower()):
                colunas_data.append(coluna)
                continue

            if pd.api.types.is_object_dtype(features[coluna]) or \
               pd.api.types.is_string_dtype(features[coluna]) or \
               isinstance(features[coluna].dtype, pd.CategoricalDtype):

                try:
                    amostra = features[coluna].dropna().head(5)
                    if not amostra.empty:
                        pd.to_datetime(amostra, errors = 'raise')
                        colunas_data.append(coluna)
                    else:
                        colunas_categoricas.append(coluna)
                except:
                    colunas_categoricas.append(coluna)

            elif pd.api.types.is_numeric_dtype(features[coluna]):
                if features[coluna].nunique() > 2:
                    colunas_numericas.append(coluna)
                else:
                    colunas_categoricas.append(coluna)

        return colunas_numericas, colunas_categoricas, colunas_data

    def eda_alvo(self, df):

        print(f'Valores nulos: {df[self.target_col].isna().sum()} ({df[self.target_col].isna().mean():.2f}%)\n')

        if self.tipo == 'Classificação':

            print(f'Total de categorias {df[self.target_col].nunique()}\n')
            print('-' * 50)                

            fig = plt.figure(figsize = (6, 4))

            ax = sns.countplot(data = df, x = self.target_col, palette = 'viridis')

            ax.set_title(f'Frequencia de {self.target_col}')

            for container in ax.containers:
                ax.bar_label(container, fmt = '%d', label_type = 'edge', padding = 3)

        elif self.tipo == 'Regressão':

            print(f'Estatísticas Descritivas de {self.target_col}\n')
            print(f'{df[self.target_col].describe().round(2)}\n')
            print('-' * 50)

            negativos = (df[self.target_col] < 0).any()

            if not negativos:

                fig, axes = plt.subplots(1, 2, figsize = (14, 5))

                sns.histplot(data = df, x = self.target_col, kde = True, ax = axes[0], color = 'blue', bins = 30)
                axes[0].set_title(f'Distribuição de {self.target_col}')

                sns.histplot(data = df, x = np.log1p(df[self.target_col]), kde = True, ax = axes[1], color = 'green', bins = 30)
                axes[1].set_title(f'Distribuição de {self.target_col} com transformação logarítmica')

            else:

                fig, ax = plt.subplots(figsize = (7, 4))

                sns.histplot(data = df, x = self.target_col, kde = True, ax = ax, color = 'blue', bins = 30)
                ax.set_title(f'Distribuição de {self.target_col}')

        caminho = f'analise_{self.target_col}.png'
        plt.tight_layout()
        plt.savefig(caminho, dpi = 150)
        plt.show()
        return caminho

    def eda_num(self, df, coluna):

        print(f'Variável {coluna}\n')
        print(f'{df[coluna].describe().to_string()}\n')
        print(f'Valores nulos: {df[coluna].isna().sum()} ({df[coluna].isna().mean() * 100:.2f}%)\n')
        print('-' * 50)

        fig, axes = plt.subplots(1, 3, figsize = (14, 5))

        sns.histplot(data = df, x = coluna, kde = True, ax = axes[0])
        axes[0].set_title(f'Distribuição geral {coluna}')

        if self.tipo == 'Classificação':

            sns.histplot(data = df, x = coluna, hue = self.target_col, kde = True, ax = axes[1], multiple = 'stack', palette = 'crest')
            axes[1].set_title(f'Distribuição {coluna} por {self.target_col}')

            sns.boxplot(data = df, x = self.target_col, y = coluna, ax = axes[2], palette = 'crest')
            axes[2].set_title(f'Boxplot de {coluna}')

            if pd.api.types.is_numeric_dtype(df[coluna]):
                try:
                    if df[coluna].max() > (df[coluna].quantile(0.75) * 10) and df[coluna].min() >= 0:
                        axes[2].set_yscale('symlog')
                        axes[2].set_title(f'Boxplot de {coluna} (Escala Log)')
                except Exception:
                    pass
        
        elif self.tipo == 'Regressão':

            sns.regplot(data = df, x = coluna, y = self.target_col, ax = axes[1], scatter_kws = {'alpha': 0.2, 'color': 'blue'}, line_kws = {'color': 'red'})
            axes[1].set_title(f'Dispersão {coluna} vs {self.target_col}')

            sns.boxplot(data = df, x = coluna, ax = axes[2], color = 'blue')
            axes[2].set_title(f'Boxplot de {coluna}')

        caminho = f'analise_{self.tipo}_{coluna}.png'
        plt.tight_layout()  
        plt.savefig(caminho, dpi = 150)
        plt.show()
        return caminho

    def colunas_categoricas(self, df, coluna):
        qtd_unicos = df[coluna].nunique()

        if qtd_unicos > (len(df) * 0.9):
            print(f'Ignorando {coluna}, pois parece ser uma coluna de ID')
            print('-' * 50)
            return False, None, None

        print(f'Variável {coluna}\n')
        print(f'Total de categorias {qtd_unicos}\n')
        print(f'Valores nulos: {df[coluna].isna().sum()} ({df[coluna].isna().mean():.2f}%)\n')
        print(f'Top 5 categorias mais frequentes\n')
        print(f'{df[coluna].value_counts().head().to_string()}')
        print('-' * 50)

        if qtd_unicos > self.max_categorias:
            print(f'{coluna} tem muitas categorias ({qtd_unicos}). Plotando apenas as {self.max_categorias} mais frequentes')

            top_categorias = df[coluna].value_counts().index[:self.max_categorias]   
            df_filtrado = df[df[coluna].isin(top_categorias)]

        else:
            top_categorias = df[coluna].value_counts().index
            df_filtrado = df.copy()

        return True, top_categorias, df_filtrado

    def eda_cat(self, df, coluna):

        valido, top_categorias, df_filtrado = self.colunas_categoricas(df, coluna)

        if not valido:
            return

        fig, axes = plt.subplots(1, 2, figsize = (14, 5))
        
        ax0 = sns.countplot(data = df_filtrado, x = coluna, ax = axes[0], order = top_categorias)
        axes[0].set_title(f'Frequência geral de {coluna}')

        if self.tipo == 'Classificação':
            ax1 = sns.countplot(data = df_filtrado, x = coluna, hue = self.target_col, ax = axes[1], palette = 'viridis', order = top_categorias)
            axes[1].set_title(f'Frequência de {coluna} por {self.target_col}')

        elif self.tipo == 'Regressão':
            ordem_target = df_filtrado.groupby(coluna)[self.target_col].mean().sort_values(ascending = False).index if top_categorias is None else top_categorias
            ax1 = sns.boxplot(data = df_filtrado, x = coluna, y = self.target_col, ax = axes[1], palette = 'viridis', order = ordem_target)
            axes[1].set_title(f'Distribuição de {self.target_col} por {coluna}')

        for ax in [ax0, ax1]:
            if hasattr(ax, 'containers') and len(ax.containers) > 0:
                for container in ax.containers:
                    ax.bar_label(container, fmt = '%d', label_type = 'edge', padding = 3)
            ax.tick_params(axis = 'x', rotation = 45)

        caminho = f'analise_{self.tipo}_{coluna}.png'
        plt.tight_layout()
        plt.savefig(caminho, dpi = 150)
        plt.show()
        return caminho

    def eda_cat_tempo(self, df, coluna_data, coluna):
        
        valido, top_categorias, df_filtrado = self.colunas_categoricas(df, coluna)

        if not valido:
            return
        
        ordem_dias = ['Seg', 'Ter', 'Qua', 'Qui', 'Sex', 'Sab', 'Dom']
        ordem_meses = sorted(df_filtrado[f'{coluna_data}_ano_mes'].unique())

        fig, axes = plt.subplots(2, 1, figsize = (14, 10))
        
        sns.countplot(data = df_filtrado, x = f'{coluna_data}_ano_mes', hue = coluna, ax = axes[0], order = ordem_meses, hue_order = top_categorias)
        axes[0].set_title(f'Frequência geral de {coluna} por ao longo do Tempo')

        sns.countplot(data = df_filtrado, x = f'{coluna_data}_dia_semana', hue = coluna, ax = axes[1], order = ordem_dias, hue_order = top_categorias)
        axes[1].set_title(f'Frequência geral de {coluna} por Ano ao longo do Tempo')

        for ax in axes:
            if hasattr(ax, 'containers') and len(ax.containers) > 0:
                for container in ax.containers:
                    ax.bar_label(container, fmt = '%d', label_type = 'edge', padding = 3)
                ax.tick_params(axis = 'x', rotation = 45)

        caminho = f'analise_{coluna}.png'
        plt.tight_layout()
        plt.savefig(caminho, dpi = 150)
        plt.show()
        return caminho

    def eda_estatico(self, df):

        sns.set_theme(style = 'whitegrid')

        colunas_numericas, colunas_categoricas, colunas_data = self.tipo_colunas(df)

        print(f'Análise exploratória\n')
        print(f'Alvo: {self.target_col}\n')
        print(f'{len(colunas_numericas)} colunas numéricas')
        print(f'{colunas_numericas}\n')
        print(f'{len(colunas_categoricas)} colunas categóricas')
        print(f'{colunas_categoricas}\n')
        if colunas_data:
            print(f'{len(colunas_data)} colunas de tempo')
            print(f'{colunas_data}\n')

        self.eda_alvo(df)

        if colunas_numericas:
            print('Resumo e Análise das Variáveis Numéricas\n') 
            for coluna in colunas_numericas:
                self.eda_num(df, coluna)

        if colunas_categoricas:
            print('Resumo e Análise das Variáveis Categóricas\n')
            for coluna in colunas_categoricas:
                self.eda_cat(df, coluna)

    def eda_series_temporais(self, df, coluna_data):

        sns.set_theme(style = 'whitegrid')

        print(f'EDA da Série Temporal na coluna {coluna_data}')

        colunas_numericas, colunas_categoricas, colunas_data = self.tipo_colunas(df)

        print(f'Análise exploratória')
        print(f'{len(colunas_numericas)} colunas numéricas')
        print(colunas_numericas)
        print(f'{len(colunas_categoricas)} colunas categóricas')
        print(colunas_categoricas)
        print(f'{len(colunas_data)} colunas de tempo')
        print(colunas_data)
        
        df_data = df.copy()
        df_data[coluna_data] = pd.to_datetime(df_data[coluna_data])

        df_data[f'{coluna_data}_ano'] = df[coluna_data].dt.year
        df_data[f'{coluna_data}_ano_mes'] = df_data[coluna_data].dt.to_period('M').astype(str)
        df_data[f'{coluna_data}_dia_semana'] = df[coluna_data].dt.dayofweek

        dias = {0:'Seg', 1:'Ter', 2:'Qua', 3:'Qui', 4:'Sex', 5:'Sab', 6:'Dom'}
        df_data[f'{coluna_data}_dia_semana'] = df_data[f'{coluna_data}_dia_semana'].map(dias)
        ordem_dias = ['Seg', 'Ter', 'Qua', 'Qui', 'Sex', 'Sab', 'Dom']
        ordem_meses = sorted(df_data[f'{coluna_data}_ano_mes'].unique())

        plt.figure(figsize = (14, 5))
        df_data.groupby(df_data[coluna_data].dt.to_period('M')).size().plot(
        kind = 'line', marker = 'o', color = 'green', linewidth = 2)
        plt.title('Volume de Registros Coletados por Ano/Mês', weight = 'bold')
        plt.xlabel('Período')
        plt.ylabel('Registros')
        plt.xticks(rotation = 45)
        plt.tight_layout()
        plt.show()

        fig, axes = plt.subplots(2, 1, figsize = (14, 10))

        if self.tipo == 'Classificação':

            ordem_meses = sorted(df_data[f'{coluna_data}_ano_mes'].unique())
            
            sns.countplot(data = df_data, x = f'{coluna_data}_ano_mes', hue = self.target_col, ax = axes[0], order = ordem_meses, color = 'blue')
            axes[0].set_title(f'Distribuição de {self.target_col} po Mês', weight = 'bold')

            sns.countplot(data = df_data, x = f'{coluna_data}_dia_semana', hue = self.target_col, ax = axes[1], order = ordem_dias, color = 'green')
            axes[1].set_title(f'Distribuição de {self.target_col} por Dia da Semana', weight = 'bold')

            for ax in axes:
                if hasattr(ax, 'containers') and len(ax.containers) > 0:
                    for container in ax.containers:
                        ax.bar_label(container, fmt = '%d', label_type = 'edge', padding = 3)
                    ax.tick_params(axis = 'x', rotation = 45)

        if self.tipo == 'Regressão':

            df_mes = df_data.groupby(f'{coluna_data}_ano_mes')[self.target_col].mean().reset_index()
            sns.lineplot(data = df_mes, x = f'{coluna_data}_ano_mes', y = self.target_col, marker = 'o', ax = axes[0], color = 'blue', linewidth = 2)
            axes[0].set_title(f'Evolução da Média de {coluna_data} ao Longo dos Meses', weight = 'bold')
            axes[0].tick_params(axis = 'x', rotation = 45)

            df_dia = df_data.groupby(f'{coluna_data}_dia_semana')[self.target_col].mean().reindex(ordem_dias).reset_index()
            sns.barplot(data = df_dia, x = f'{coluna_data}_dia_semana', y = self.target_col, marker = 'o', ax = axes[1], order = ordem_dias, color = 'blue', linewidth = 2)
            axes[1].set_title(f'Evolução da Média de {self.target_col} ao Longo da Semana', weight = 'bold')
            axes[1].tick_params(axis = 'x', rotation = 45)

        plt.tight_layout()
        plt.show()

        if colunas_categoricas:
            
            print('Analisando as Variáveis Categóricas ao longo tempo')

            for coluna in colunas_categoricas:
                self.eda_cat_tempo(df_data, coluna_data, coluna)