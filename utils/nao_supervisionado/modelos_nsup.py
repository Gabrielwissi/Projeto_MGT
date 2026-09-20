from sklearn.decomposition import KernelPCA, PCA
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering, MeanShift
from kmodes.kmodes import KModes
from kmodes.kprototypes import KPrototypes

def modelos_agrupamento():

    modelos_numericos = {
        'KMeans': {
            'modelo': KMeans,
            'hiperparams': {
                'n_clusters': {'tipo': 'int', 'default': 5, 'low': 2, 'high': 15}
            }
        },
        'Hierárquico': {
            'modelo': AgglomerativeClustering,
            'hiperparams': {
                'n_clusters': {'tipo': 'int', 'default': 5, 'low': 2, 'high': 15},
                'linkage': {'tipo': 'categorical', 'default': 'ward', 'choices': ['ward', 'complete', 'average']}
            }
        },
        'DBSCAN': {
            'modelo': DBSCAN,
            'hiperparams': {
                'eps': {'tipo': 'float', 'default': 0.5, 'low': 0.1, 'high': 2.0},
                'min_samples': {'tipo': 'int', 'default': 5, 'low': 3, 'high': 20}
            }
        },
        'MeanShift': {
            'modelo': MeanShift,
            'hiperparams': {}
        }
    }

    modelos_categoricos = {
    'KModes': {
            'modelo': KModes,
            'hiperparams': {
                'n_clusters': {'tipo': 'int', 'default': 5, 'low': 3, 'high': 15},
                'init': {'tipo': 'categorical', 'default': 'Cao', 'choices': ['Cao', 'Huang']}
            }
        },
    'KPrototype': {
            'modelo': KPrototypes,
            'hiperparams': {
                'n_clusters': {'tipo': 'int', 'default': 5, 'low': 2, 'high': 15},
                'gamma': {'tipo': 'float', 'default': None}
            }
        }
    }

    reducoes = {
    '': None,
    'PCA': PCA(n_components = 2, random_state = 42),
    'KPCA': KernelPCA(n_components = 2, kernel = 'rbf', random_state = 42, n_jobs = -1)
    }

    return modelos_numericos, modelos_categoricos, reducoes

def modelos_associacao():

    modelos = {
        'Apriori': 'apriori',
        'Eclat': 'fpgrowth'
    }

    return modelos