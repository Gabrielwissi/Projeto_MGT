from sklearn.decomposition import KernelPCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

# ALGORITMOS DE CLASSIFICAÇÃO
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.neural_network import MLPClassifier

# ALGORITMOS DE REGRESSÃO
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from sklearn.neural_network import MLPRegressor

def modelos_supervisionados(tipo):
        
        if tipo == 'Classificação':
            
            modelos = {
                'Naive Bayes': {
                    'modelo': GaussianNB,
                    'hiperparams': {
                        'var_smoothing': {'tipo': 'float', 'default': 1e-9, 'low': 1e-12, 'high': 1e-6, 'log': True}
                    }      
                },
                'Regressão Logística': {
                    'modelo': LogisticRegression,
                    'hiperparams': {
                        'C': {'tipo': 'float', 'default': 1.0, 'low': 0.001, 'high': 100, 'log': True},
                        'penalty': {'tipo': 'categorical', 'default': 'l2', 'choices': ['l1', 'l2']}
                    }
                },
                'SVC (SVM)':{
                    'modelo': SVC,
                    'hiperparams': {
                        'C': {'tipo': 'float', 'default': 1.0, 'low': 0.01, 'high': 100, 'log': True},
                        'gamma': {'tipo': 'categorical', 'default': 'scale', 'choices': ['scale', 'auto']},
                        'kernel': {'tipo': 'categorical', 'default': 'rbf', 'choices': ['rbf', 'linear', 'poly']}       
                    }      
                },
                'KNN': {
                    'modelo': KNeighborsClassifier,
                    'hiperparams': {
                        'n_neighbors': {'tipo': 'int', 'default': 5, 'low': 3, 'high': 30},
                        'weights': {'tipo': 'categorical', 'default': 'uniform', 'choices': ['uniform', 'distance']}
                    }
                },
                'Árvores de decisão': {
                     'modelo': DecisionTreeClassifier,
                     'hiperparams': {
                          'max_depth': {'tipo': 'int', 'default': None, 'low': 3, 'high': 30},
                          'min_samples_split': {'tipo': 'int', 'default': 2, 'low': 2, 'high': 20},
                          'min_samples_leaf': {'tipo': 'int', 'default': 1, 'low': 1, 'high': 10}
                    }
                },
                'Random Forest': {
                     'modelo': RandomForestClassifier,
                     'hiperparams': {
                          'n_estimators': {'tipo': 'int', 'default': 100, 'low': 100, 'high': 600},
                          'max_depth': {'tipo': 'int', 'default': None, 'low': 3, 'high': 30},
                          'min_samples_split': {'tipo': 'int', 'default': 2, 'low': 2, 'high': 30},
                          'max_features': {'tipo': 'categorical', 'default': 'sqrt', 'choices': ['sqrt', 'log2']},
                    }
                },
                'XGBoost': {
                     'modelo': XGBClassifier,
                     'hiperparams': {
                          'n_estimators': {'tipo': 'int', 'default': 100, 'low': 100, 'high': 800},
                          'max_depth': {'tipo': 'int', 'default': 6, 'low': 3, 'high': 12},
                          'learning_rate': {'tipo': 'float', 'default': 0.3, 'low': 0.01, 'high': 0.3, 'log': True},
                          'subsample': {'tipo': 'float', 'default': 1.0, 'low': 0.6, 'high': 1.0},
                          'colsample_bytree': {'tipo': 'float', 'default': 1.0, 'low': 0.6, 'high': 1.0}
                    }
                },
                'LightGBM': {
                     'modelo': LGBMClassifier,
                     'hiperparams': {
                          'n_estimators': {'tipo': 'int', 'default': 100, 'low': 100, 'high': 800},
                          'num_leaves': {'tipo': 'int', 'default': 31, 'low': 15, 'high': 150},
                          'max_depth': {'tipo': 'int', 'default': -1, 'low': 3, 'high': 15},
                          'learning_rate': {'tipo': 'float', 'default': 0.1, 'low': 0.01, 'high': 0.3, 'log ': True},
                    }
                },
                'CatBoost': {
                     'modelo': CatBoostClassifier,
                     'hiperparams': {
                          'iterations': {'tipo': 'int', 'default': 1000, 'low': 200, 'high': 1500},
                          'depth': {'tipo': 'int', 'default': 6, 'low': 4, 'high': 10},
                          'l2_leaf_reg': {'tipo': 'float', 'default': 3, 'low': 1, 'high': 10},
                          'learning_rate': {'tipo': 'float', 'default': 0.03, 'low': 0.01, 'high': 0.3},
                    }
                },
                'Rede Neural (MLP)': {
                     'modelo': MLPClassifier,
                     'hiperparams': {
                          'hidden_layer_sizes': {'tipo': 'categorical', 'default': (100,), 'choices': [(50,), (100,), (100, 50)]},
                          'alpha': {'tipo': 'float', 'default': 0.0001, 'low': 1e-5, 'high': 1e-1, 'log': True},
                          'learning_rate_init': {'tipo': 'float', 'default': 0.001, 'low': 1e-4, 'high': 1e-1, 'log': True},
                    }
                }
            }

            reducoes = {
            '': None,
            'LDA': LDA(n_components = 1),
            'KPCA': KernelPCA(n_components = 2, kernel = 'rbf', random_state = 42, n_jobs = -1)
            }

        elif tipo == 'Regressão':

            modelos = {
            'Regressão Linear': {
                 'modelo': LinearRegression,
                 'hiperparams': {}
            },
            'Ridge': {
                 'modelo': Ridge,
                 'hiperparams': {
                       'alpha': {'tipo': 'float', 'default': 1.0, 'low': 0.001, 'high': 100, 'log': True}
                }
            },
            'SVR (SVC)': {
                 'modelo': SVR,
                 'hiperparams': {
                       'C': {'tipo': 'float', 'default': 1.0, 'low': 0.01, 'high': 100, 'log': True},
                       'epsilon': {'tipo': 'float', 'default': 0.1, 'low': 0.01, 'high': 1.0, 'log': True},
                       'gamma': {'tipo': 'categorical', 'default': 'scale', 'choices': ['scale', 'auto']}
                }
            },
            'KNN': {
                    'modelo': KNeighborsRegressor,
                    'hiperparams': {
                        'n_neighbors': {'tipo': 'int', 'default': 5, 'low': 3, 'high': 30},
                        'weights': {'tipo': 'categorical', 'default': 'uniform', 'choices': ['uniform', 'distance']}
                    }
                },
                'Árvores de decisão': {
                     'modelo': DecisionTreeRegressor,
                     'hiperparams': {
                          'max_depth': {'tipo': 'int', 'default': None, 'low': 3, 'high': 30},
                          'min_samples_split': {'tipo': 'int', 'default': 2, 'low': 2, 'high': 20},
                          'min_samples_leaf': {'tipo': 'int', 'default': 1, 'low': 1, 'high': 10}
                    }
                },
                'Random Forest': {
                     'modelo': RandomForestRegressor,
                     'hiperparams': {
                          'n_estimators': {'tipo': 'int', 'default': 100, 'low': 100, 'high': 600},
                          'max_depth': {'tipo': 'int', 'default': None, 'low': 3, 'high': 30},
                          'min_samples_split': {'tipo': 'int', 'default': 2, 'low': 2, 'high': 30},
                          'max_features': {'tipo': 'categorical', 'default': 'sqrt', 'choices': ['sqrt', 'log2']},
                    }
                },
                'XGBoost': {
                     'modelo': XGBRegressor,
                     'hiperparams': {
                          'n_estimators': {'tipo': 'int', 'default': 100, 'low': 100, 'high': 800},
                          'max_depth': {'tipo': 'int', 'default': 6, 'low': 3, 'high': 12},
                          'learning_rate': {'tipo': 'float', 'default': 0.3, 'low': 0.01, 'high': 0.3},
                          'subsample': {'tipo': 'float', 'default': 1.0, 'low': 0.6, 'high': 1.0},
                          'colsample_bytree': {'tipo': 'float', 'default': 1.0, 'low': 0.6, 'high': 1.0}
                    }
                },
                'LightGBM': {
                     'modelo': LGBMRegressor,
                     'hiperparams': {
                          'n_estimators': {'tipo': 'int', 'default': 100, 'low': 100, 'high': 800},
                          'num_leaves': {'tipo': 'int', 'default': 31, 'low': 15, 'high': 150},
                          'max_depth': {'tipo': 'int', 'default': -1, 'low': 3, 'high': 15},
                          'learning_rate': {'tipo': 'float', 'default': 0.1, 'low': 0.01, 'high': 0.3},
                    }
                },
                'CatBoost': {
                     'modelo': CatBoostRegressor,
                     'hiperparams': {
                          'iterations': {'tipo': 'int', 'default': 1000, 'low': 200, 'high': 1500},
                          'depth': {'tipo': 'int', 'default': 6, 'low': 4, 'high': 10},
                          'l2_leaf_reg': {'tipo': 'float', 'default': 3, 'low': 1, 'high': 10},
                          'learning_rate': {'tipo': 'float', 'default': 0.03, 'low': 0.01, 'high': 0.3},
                    }
                },
                'Rede Neural (MLP)': {
                     'modelo': MLPRegressor,
                     'hiperparams': {
                          'hidden_layer_sizes': {'tipo': 'categorical', 'default': (100,), 'choices': [(50,), (100,), (100, 50)]},
                          'alpha': {'tipo': 'float', 'default': 0.0001, 'low': 1e-5, 'high': 1e-1, 'log': True},
                          'learning_rate_init': {'tipo': 'float', 'default': 0.001, 'low': 1e-4, 'high': 1e-1, 'log': True},
                          'max_iter': {'tipo': 'int', 'default': 1000, 'low': 200, 'high': 1000},
                          'early_stopping': {'tipo': 'categorical', 'default': True, 'choices': [True]},
                    }
                }
            }

            reducoes = {
            '': None,
            'KPCA': KernelPCA(n_components = 2, kernel = 'rbf', random_state = 42, n_jobs = -1)
            }

        else:
            raise ValueError("Tipo incorreto, escolha entre 'Classificação' e 'Regressão'.")

        return modelos, reducoes