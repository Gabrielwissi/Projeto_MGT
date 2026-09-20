import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, ConfusionMatrixDisplay,
    roc_curve, auc
)
from sklearn.preprocessing import label_binarize
from scipy import stats


def comparar_classificacao(df_resultado, df_otimizado):

    modelos = np.atleast_1d(df_otimizado['Modelo'].values)
    nomes = [nome.split(' + ')[0] for nome in modelos]

    df_resultado_index = df_resultado.drop_duplicates(subset = ['Modelo']).set_index('Modelo')
    df_otimizado_index = df_otimizado.set_index('Modelo')

    acuracia_antes = np.atleast_1d(np.asarray(df_resultado_index.loc[modelos, 'Acurácia Teste (%)'].to_numpy(), dtype = np.float64)).flatten()
    acuracia_depois = np.atleast_1d(np.asarray(df_otimizado_index.loc[modelos, 'Acurácia Teste (%)'].to_numpy(), dtype = np.float64)).flatten()

    f1_antes_raw = np.atleast_1d(np.asarray(df_resultado_index.loc[modelos, 'F1 - Score'].to_numpy(), dtype = np.float64)).flatten()
    f1_depois_raw = np.atleast_1d(np.asarray(df_otimizado_index.loc[modelos, 'F1 - Score'].to_numpy(), dtype = np.float64)).flatten()

    multiplicador = 100 if float(np.max(f1_antes_raw)) <= 1.0 else 1

    f1_antes = f1_antes_raw * multiplicador
    f1_depois = f1_depois_raw * multiplicador

    x = np.arange(len(modelos))
    largura = 0.3

    fig, axes = plt.subplots(1, 2, figsize = (16, 6))
    fig.suptitle('Classificação - Comparação antes e depois da otimização', fontsize = 14, fontweight = 'bold', y = 1.02)

    for ax, antes, depois, titulo, ylabel in zip(
        axes,
        [acuracia_antes, f1_antes],
        [acuracia_depois, f1_depois],
        ['Acurácia Teste (%)', 'F1 - Score (%)'],
        ['Acurácia (%)', 'F1 - Score (%)']):

        ax.bar(x - largura/2, antes, largura, label = 'Antes - Modelos padrão', color = 'blue', alpha = 0.85) 
        ax.bar(x + largura/2, depois, largura, label = 'Depois - Modelos otimizados', color = 'green', alpha = 0.85)

        for i, (antes, depois) in enumerate(zip(antes, depois)):
            
            ganho = depois - antes  
            cor = 'green' if ganho >= 0 else 'red'
            sinal = '+' if ganho >= 0 else '-'

            ax.annotate(f'{sinal}{ganho:.1f}%',
                        xy = (x[i] + largura/2, depois),
                        xytext = (0, 5), textcoords = 'offset points',
                        ha = 'center', fontsize = 8, color = cor, fontweight = 'bold')
        
        ax.set_title(titulo, fontweight = 'bold', fontsize = 10)
        ax.set_xticks(x)
        ax.set_xticklabels(nomes, rotation = 15, ha = 'right')
        ax.set_ylabel(ylabel)
        ax.legend(loc = 'lower right')

        teto_y = min(100, float(np.max(depois)) * 1.15)
        ax.set_ylim(0, teto_y)

        ax.grid(axis = 'y', alpha = 0.2)

    plt.tight_layout()
    plt.savefig('comparacao_classificacao.png', dpi = 150, bbox_inches = 'tight')
    plt.show()

def matriz_confusao(pipeline, X_test, y_test):

    predicao = pipeline.predict(X_test)

    try:
        classes = pipeline.steps[-1][1].classes_
    except AttributeError:
        classes = np.unique(y_test)

    cm = confusion_matrix(y_test, predicao, labels = classes)

    fig, axes = plt.subplots(1, 2, figsize = (16, 6))
    fig.suptitle('Matriz de Confusão', fontsize = 14, fontweight = 'bold', y = 1.05)

    disp = ConfusionMatrixDisplay(confusion_matrix = cm, display_labels = classes)
    disp.plot(ax = axes[0], colorbar = False, cmap = 'Blues')
    axes[0].set_title('Contagem Absoluta', fontweight = 'bold', pad = 10)
    axes[0].grid(False)

    soma_linhas = cm.sum(axis = 1, keepdims = True)
    cm_norm = np.nan_to_num(cm.astype(float) / np.where(soma_linhas == 0, 1, soma_linhas))

    disp_norm = ConfusionMatrixDisplay(confusion_matrix = np.round(cm_norm, 2), display_labels = classes)
    disp_norm.plot(ax = axes[1], colorbar = False, cmap = 'Greens')
    axes[1].set_title('Normalizada - proporção por classe', fontweight = 'bold', pad = 10)
    axes[1].grid(False)

    plt.tight_layout()
    plt.savefig('matriz_de_confusao.png', dpi = 150, bbox_inches = 'tight')
    plt.show()

def roc(pipeline, X_test, y_test):

    modelo = pipeline.steps[-1][1]

    if not hasattr(modelo, 'predict_proba'):
        print('O modelo do pipeline não possui o método predict_proba - curva ROC não disponível')
        return
    
    probabilidades = pipeline.predict_proba(X_test)

    try:
        classes = modelo.classes_
    except AttributeError:
        classes = np.unique(y_test)

    n_classes = len(classes)
    fig, ax = plt.subplots(figsize = (8, 6))

    if n_classes == 2:

        fpr, tpr, _ = roc_curve(y_test, probabilidades[:, 1])
        auc_score = auc(fpr, tpr)
        ax.plot(fpr, tpr, color = 'blue', linewidth = 2,
                label = f'AUC = {auc_score:.4f}')
    
    else:
        y_bin = label_binarize(y_test, classes = classes)

        if y_bin.shape[1] == 1:
            y_bin = np.hstack((1 - y_bin, y_bin))

        cores = plt.cm.tab10(np.linspace(0, 1, n_classes))

        for i, (classe, cor) in enumerate(zip(classes, cores)):
            fpr, tpr, _ = roc_curve(y_bin[:, i], probabilidades[:, i])
            auc_score = auc(fpr, tpr)
            ax.plot(fpr, tpr, color = cor, linewidth = 2,
                    label = f'Classe{classe} - AUC = {auc_score:.4f}')

    ax.plot([0, 1], [0, 1], 'k--', linewidth = 1, label = 'Aleatório - AUC = 0.5')

    ax.set_title('Curva ROC - Modelo Final', fontsize = 14, fontweight = 'bold', pad = 15)
    ax.set_xlabel('Taxa de Falsos Positivos (FPR)', fontsize = 10)
    ax.set_ylabel('Taxa de Verdadeiros (TPR)', fontsize = 10)
    ax.legend(loc = 'lower right', fontsize = 10)
    ax.grid(alpha = 0.2)

    plt.tight_layout()
    plt.savefig('curva_roc.png', dpi = 150, bbox_inches = 'tight')
    plt.show()

def features(pipeline, X_train, tipo):

    try:
        import shap
        tem_shap = True
    except ImportError:
        tem_shap = False

    modelo = pipeline.named_steps['modelo']
    nome_modelo = type(modelo).__name__
    tem_importance = hasattr(modelo, 'features_importances_')
    tem_coef = hasattr(modelo, 'coef_')

    try:
        engenharia = pipeline.named_steps['Engenharia']
        nomes_features = engenharia.get_feature_names_out()
    except Exception:
        nomes_features = [f'feature_{i}' for i in range(X_train.shape[1])]

    try:
        import scipy
        X_trans = pipeline.named_steps['Engenharia'].transform(X_train)
        if scipy.sparse.issparse(X_trans):
            X_trans = X_trans.toarray()

        X_trans_df = pd.DataFrame(X_trans, columns = nomes_features)

    except Exception:
        X_trans_df = X_train

    modelos_arvores = ['RandomForestClassifier', 'RandomForestRegressor',
                        'XGBClassifier', 'XGBRegressor',
                        'LGBMClassifier', 'LGBMRegressor',
                        'CatBoostClassifier', 'CatBoostRegressor',
                        'DecisionTreeClassifier', 'DecisionTreeRegressor']
        
    if tem_shap and nome_modelo in modelos_arvores:
        try:
            explainer = shap.TreeExplainer(modelo)
            shap_values = explainer.shap_values(X_trans_df)

            if isinstance(shap_values, list):
                if len(shap_values) == 2:
                    shap_values = shap_values[1]
                else:
                    shap_values = np.abs(np.array(shap_values)).mean(axis = 0)

            elif hasattr(shap_values, 'values'):
                shap_values_original = shap_values.values
                if len(shap_values.shape) == 3:
                    shap_values = shap_values_original[:, :, 1]
                else:
                    shap_values = shap_values_original
            
            elif isinstance(shap_values, np.ndarray) and len(shap_values.shape) == 3:
                if shap_values.shape[2] == 2:
                    shap_values = shap_values[:, :, 1]
                else:
                    shap_values = np.abs(shap_values).mean(axis = 2)
            
            fig, ax = plt.subplots(figsize = (10, 8))
            shap.summary_plot(shap_values, X_trans_df, plot_type = 'bar', show = False)
            plt.title(f'Importância de Features (SHAP) - {nome_modelo}', fontsize = 15, fontweight = 'bold', pad = 15)
            plt.tight_layout()
            plt.savefig('features_importance.png', dpi = 150, bbox_inches = 'tight')
            plt.show()
            return
        
        except Exception as e:
            print('SHAP falhou devido a um erro de compatibilidade {e}, usando a imporância nativa')

    if tem_importance:
        importancias = modelo.feature_importances_
        _plotar_barras_importancia(importancias, nomes_features, nome_modelo, 'features_importances.png')

    elif tem_coef:
        if len(modelo.coef_.shape)  > 1:
            importancias = np.abs(modelo.coef_[0])
        else:
            importancias = np.abs(modelo.coef_)
        _plotar_barras_importancia(importancias, nomes_features, nome_modelo, 'features_coeficientes.png')

    else:
        print(f'Modelo {nome_modelo} não suporta imortância de features diretamente')
        print('Instale shap para usar com este modelo: pip install shap')

def _plotar_barras_importancia(importancias, nomes_features, nome_modelo, nome_arquivo):

    indices = np.argsort(importancias)[::-1][:20]

    fig, ax = plt.subplots(figsize = (10, 8))
    ax.barh(range(len(indices)), importancias[indices], color = 'blue', alpha = 0.85)
    ax.set_yticks(range(len(indices)))
    ax.set_yticklabels([nomes_features[i] for i in indices])
    ax.invert_yaxis()
    ax.set_title(f'Importância de Features - {nome_modelo}', fontsize = 14, fontweight = 'bold', pad = 15)
    ax.set_xlabel('Importância')
    ax.grid(axis = 'x', alpha = 0.2)
    plt.tight_layout()
    plt.savefig(nome_arquivo, dpi = 150, bbox_inches = 'tight')
    plt.show()

def optuna_calssificacao(study, nome_modelo):

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

    nome_limpo = nome_modelo.split(' + ')[0]
    
    fig, axes = plt.subplots(1, 2, figsize = (16, 5))
    fig.suptitle(f'Histórico de Otimização Classificação - {nome_limpo}', fontsize = 14, fontweight = 'bold', y = 1.05)

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

def comparar_regressao(df_resultado, df_otimizado):

    modelos = df_otimizado['Modelo'].tolist()
    nomes = [nome.split(' + ')[0] for nome in modelos]

    df_resultado_index = df_resultado.drop_duplicates(subset = ['Modelo']).set_index('Modelo')
    df_otimizado_index = df_otimizado.set_index('Modelo')

    rmse_antes = np.asarray(df_resultado_index.loc[modelos, 'RMSE Teste'].values, dtype = float)
    rmse_depois = np.asarray(df_otimizado_index.loc[modelos, 'RMSE Teste'].values, dtype = float)

    r2_antes_raw = np.asarray(df_resultado_index.loc[modelos, 'R² Teste'].values, dtype = float)
    r2_depois_raw = np.asarray(df_otimizado_index.loc[modelos, 'R² Teste'].values, dtype = float)

    multiplicador = 100 if np.max(r2_antes_raw) <= 1.0 else 1

    r2_antes = r2_antes_raw * multiplicador
    r2_depois = r2_depois_raw * multiplicador

    x = np.arange(len(modelos))
    largura = 0.3

    fig, axes = plt.subplots(1, 2, figsize = (16, 6))
    fig.suptitle('Regressão - Comparação antes e depois da otimização', fontsize = 14, fontweight = 'bold', y = 1.02)

    for i, (ax, antes, depois, titulo, metrica) in enumerate(zip(
        axes,
        [rmse_antes, r2_antes],
        [rmse_depois, r2_depois],
        ['RMSE Teste', 'R² Teste'],
        [False, True])):

        ax.bar(x - largura/2, antes, largura, label = 'Antes - Modelos padrão', color = 'blue', alpha = 0.85) 
        ax.bar(x + largura/2, depois, largura, label = 'Depois - Modelos otimizados', color = 'green', alpha = 0.85)

        for i, (valor_antes, valor_depois) in enumerate(zip(antes, depois)):
            
            ganho = valor_depois - valor_antes

            if not metrica:
                pct_reducao = (ganho / abs(valor_antes)) * 100 if valor_antes != 0 else 0
                cor = 'green' if pct_reducao < 0 else 'red'
                sinal = '+' if pct_reducao > 0 else ''
                texto_anotacao = f'{sinal}{pct_reducao:.1f}%'
            else: 
                cor = 'green' if ganho >= 0 else 'red'
                sinal = '+' if ganho > 0 else ''
                texto_anotacao = f'{sinal}{ganho:.1f}%'
            
            ax.annotate(texto_anotacao,
                        xy = (x[i] + largura/2, valor_depois),
                        xytext = (0, 5), textcoords = 'offset points',
                        ha = 'center', fontsize = 8, color = cor, fontweight = 'bold')
        
        ax.set_title(titulo, fontweight = 'bold', fontsize = 10)
        ax.set_xticks(x)
        ax.set_xticklabels(nomes, rotation = 15, ha = 'right')
        ax.set_ylabel(titulo)
        ax.legend(loc = 'upper right')

        min_valor = min(float(np.min(antes)), float(np.min(depois)))
        max_valor = max(float(np.max(antes)), float(np.max(depois)))

        y_min = min_valor * 1.25 if min_valor < 0 else 0
        y_max = max_valor * 1.25 if max_valor > 0 else 1        
        
        ax.set_ylim(y_min, y_max)

        ax.grid(axis = 'y', alpha = 0.2)

    plt.tight_layout()
    plt.savefig('comparacao_regressao.png', dpi = 150, bbox_inches = 'tight')
    plt.show()

def residuos(pipeline, X_test, y_test):

    y_array = y_test.values if (y_test, 'values') else np.array(y_test)

    predicao = pipeline.predict(X_test)
    residuos = y_array - predicao

    fig, axes = plt.subplots(1, 3, figsize = (18, 5))
    fig.suptitle('Análise de Residuos', fontsize = 14, fontweight = 'bold', y = 1.05)

    sns.histplot(residuos, kde = True, ax = axes[0], color = 'blue', alpha = 0.7)
    axes[0].axvline(x = 0, color = 'red', linestyle = '--', linewidth = 1.5)
    axes[0].set_title('Distribuição dos Resíduos', fontweight = 'bold', pad = 10)
    axes[0].set_xlabel('Resíduo')
    axes[0].set_ylabel('Frequência')
    axes[0].grid(alpha = 0.2)
    axes[0].legend()

    axes[1].scatter(predicao, residuos, color = 'green', linestyle = '--', linewidth = 0.5, alpha = 0.2, s = 25, edgecolor = 'white')
    axes[1].axhline(y = 0, color = 'red', linestyle = '--', linewidth = 1.5)
    axes[1].set_title('Resíduos vs Valor Prdeito', fontweight = 'bold', pad = 11)
    axes[1].set_xlabel('Valor Predito pelo Modelo')
    axes[1].set_ylabel('Resíduo')
    axes[1].grid(alpha = 0.2)

    (osm, osr), (slope, intercept, r) = stats.probplot(residuos, dist = 'norm')

    axes[2].scatter(osm, osr, color = 'blue', alpha = 0.5, s = 20)
    linha_x = np.array([osm.min(), osm.max()])
    linha_y = slope * linha_x + intercept
    axes[2].plot(linha_x, linha_y, color = 'red', linestyle = '-', linewidth = 2.0)
    axes[2].set_title('Gráfico Quantil-Quantil (Q-Q)', fontweight = 'bold', pad = 10)
    axes[2].set_xlabel('Quantis Teóricos')
    axes[2].set_ylabel('Resíduos')
    axes[2].grid(alpha = 0.2)

    plt.tight_layout()
    plt.savefig('residuos.png', dpi = 150, bbox_inches = 'tight')
    plt.show()

def predicao(pipeline, X_test, y_test):

    y_array = y_test.values if hasattr(y_test, 'values') else np.array(y_test)

    predicao = pipeline.predict(X_test)

    valor_min = min(y_array.min(), predicao.min())
    valor_max = max(y_array.max(), predicao.max())

    margem = (valor_max - valor_min)
    limite_inferior = valor_min - margem
    limite_superior = valor_max + margem

    fig, ax = plt.subplots(figsize = (8, 6))

    ax.scatter(y_array, predicao, alpha = 0.5, color = 'blue', s = 25, edgecolor = 'white', linewidth = 0.5, label = 'Predições')
    ax.plot([limite_inferior, limite_superior], [limite_inferior, limite_superior], color = 'red', linestyle = '--', linewidth = 2, label = 'Predição Perfeita')

    ax.set_xlim(limite_inferior, limite_superior)
    ax.set_ylim(limite_inferior, limite_superior)
    ax.set_title('Predição vs Real', fontweight = 'bold', pad = 11)
    ax.set_xlabel('Valor Real')
    ax.set_ylabel('Valor Predito pelo Modelo')
    ax.legend(loc = 'upper left')
    ax.grid(alpha = 0.2)

    plt.tight_layout()
    plt.savefig('predicao.png', dpi = 150, bbox_inches = 'tight')
    plt.show()

def optuna_regressao(study, nome_modelo):

    df = study.trials_dataframe()

    trials_df = df[df['state'] == 'COMPLETE'].copy()

    if trials_df.empty:
        print('Nenhum Trial com status COMPLETE foi encontrado')
        return
    
    direcao = study.direction.name

    if (trials_df['value'] < 0).any():
        scores = -trials_df['value']
        rmse = -study.best_value
    else:
        scores = trials_df['value']
        rmse = study.best_value

    if direcao == 'MAXIMIZE':
        curva_melhor = trials_df['value'].cummin()
        label_melhor = 'Melhor RMSE Acumulado (Mín)'
    else:
        curva_melhor = trials_df['value'].cummin()
        label_melhor = 'Melhor RMSE Acumulado (Mín)'

    nome_limpo = nome_modelo.split(' + ')[0]
    
    fig, axes = plt.subplots(1, 2, figsize = (16, 5))
    fig.suptitle(f'Histórico de Otimização de Regressão - {nome_limpo}', fontsize = 14, fontweight = 'bold', y = 1.05)

    axes[0].plot(trials_df['number'], scores, color = 'green', linewidth = 1, alpha = 0.5, label = 'Trial Individual')
    axes[0].plot(trials_df['number'], curva_melhor, color = 'blue', linewidth = 2.5, label = label_melhor)
    axes[0].axhline(y = rmse, color = 'red', linestyle = '--', linewidth = 2, label = f'Melhor RMSE Geral - {rmse:.4f}')
    axes[0].set_title('Convergência dos Trials', fontweight = 'bold', pad = 10)
    axes[0].set_xlabel('Número do Trial')
    axes[0].set_ylabel('RMSE (CV)')
    axes[0].legend(loc = 'best')
    axes[0].grid(alpha = 0.2)

    axes[1].hist(scores, bins = 20, color = 'green', alpha = 0.85, edgecolor = 'white')
    axes[1].axvline(x = rmse, color = 'red', linestyle = '--', linewidth = 2, label = f'Melhor RMSE Geral - {rmse:.4f}')
    axes[1].set_title('Distribuição dos Scores', fontweight = 'bold', pad = 10)
    axes[1].set_xlabel('Score da Métrica RMSE (CV)')
    axes[1].set_ylabel('Frequência de Ocorrências')
    axes[1].legend(loc = 'best')
    axes[1].grid(alpha = 0.2)

    plt.tight_layout()
    plt.savefig(f'optuna_{nome_limpo}.png', dpi = 150, bbox_inches = 'tight')
    plt.show()