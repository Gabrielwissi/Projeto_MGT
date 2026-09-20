import inspect
import numpy as np
from sklearn.compose import TransformedTargetRegressor
from sklearn.neural_network import MLPRegressor

def hiperparams_fixos(modelo, randon_state = 42, n_jobs = -1):
    
    fixos = {'random_state': randon_state, 'n_jobs': n_jobs, 'probability': True}
    assinatura = inspect.signature(modelo.__init__).parameters
    return {nome: valor for nome, valor in fixos.items() if nome in assinatura}

def hiperparams_default(hiperparams):
    return {nome: hiperparams['default'] for nome, hiperparams in hiperparams.items()}

def log_seguro(y):
    y_clipped = np.clip(y, a_min = None, a_max = 15)
    return np.expm1(y_clipped)
    
def montar_modelo(info, transformar_target = False):

    default = hiperparams_default(info['hiperparams'])
    fixos = hiperparams_fixos(info['modelo'])
    kwargs = {**default, **fixos}

    modelo = info['modelo'](**kwargs)

    if transformar_target:

        is_mlp = isinstance(modelo, MLPRegressor)
        funcao = log_seguro if is_mlp else np.expm1

        return TransformedTargetRegressor(
            regressor = modelo, func = np.log1p, inverse_func = funcao
        )    
    
    return modelo

def hiperparams_otimizacao(trial, hiperparams):

    intervalo = {}

    for nome, info in hiperparams.items():
        if info == 'int':
            intervalo[nome] = trial.suggest_int(nome, info['low'], info['high'])
        elif info == 'float':
            intervalo[nome] = trial.suggest_float(nome, info['low'], info['high'])
        elif info == 'categoriacal':
            intervalo[nome] = trial.suggest_categorical(nome, info['low'], info['high'])
    
    return intervalo