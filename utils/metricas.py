import numpy as np
from sklearn.metrics import (
    classification_report, roc_auc_score,
    root_mean_squared_error, r2_score, mean_absolute_error,
    silhouette_score, calinski_harabasz_score, davies_bouldin_score
)

def metricas_classificacao(pipeline, X_test, y_test, predicao):

    report = classification_report(y_test, predicao, output_dict = True)

    metricas_gerais = {
        'Acurácia (%)': np.round(report['accuracy'] * 100, 2),
        'F1 Macro': np.round(report['macro avg']['f1-score'], 4),
        'Precision Macro': np.round(report['macro avg']['precision'], 4),
        'Recall Macro': np.round(report['macro avg']['recall'], 4),
    }

    metricas_por_classe = {
        classe: {
            'precision': np.round(valores['precision'], 4),
            'recall': np.round(valores['recall'], 4),
            'f1': np.round(valores['f1-score'], 4),
            'suporte': int(valores['support'])
        } 
        for classe, valores in report.items() 
        if classe not in ['accuracy', 'macro avg', 'weighted avg']
    }

    auc = None

    if hasattr(pipeline, 'predict_proba'):
        try:
            probabilidades = pipeline.predict_proba(X_test)

            if probabilidades.shape[1] == 2:
                auc_valor = roc_auc_score(
                y_test, probabilidades[:, 1]
                )

            else:
                auc_valor = roc_auc_score(
                y_test, probabilidades,
                multi_class = 'ovr',
                average = 'macro',
                )
            
            auc = np.round(auc_valor, 4)

        except Exception:
            auc = None
        
    metricas_gerais['AUC'] = auc

    return metricas_gerais, metricas_por_classe

def metricas_regressao(y_test, predicao):

    rmse = np.round(root_mean_squared_error(y_test, predicao), 4)
    r2 = np.round(r2_score(y_test, predicao), 4)
    mae = np.round(mean_absolute_error(y_test, predicao), 4)

    residuos = y_test - predicao

    metricas_gerais = {
        'RMSE': rmse,
        'MAE': mae,
        'R² Teste': r2,
        'Resíduo Médio': np.round(residuos.mean(), 4),
        'Resíduo Desvio Padrão': np.round(residuos.std(), 4)
    }

    return metricas_gerais

def metricas_agrupamento(df_trans, rotulos):

    clusters = len(set(rotulos)) - (1 if -1 in rotulos else 0)

    if clusters < 2:
        return {
            'Clusters': clusters,
            'Silhouette': None,
            'Calinski-Harabasz': None,
            'Davies_Bouldin': None
        }

    metricas_gerais = {
            'Clusters': clusters,
            'Silhouette': np.round(silhouette_score(df_trans, rotulos), 4),
            'Calinski-Harabasz': np.round(calinski_harabasz_score(df_trans, rotulos), 4),
            'Davies_Bouldin': np.round(davies_bouldin_score(df_trans, rotulos), 4)
        }
    
    return metricas_gerais