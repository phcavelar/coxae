import math

import torch.nn as nn
import torch.nn.functional as F

from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from .utils import variance_score, calculate_concrete_alpha_decay
from .wrappers import CoxPHRegression
from .feature_selection import DummyFeatureSelector, SurvivalSelectKBest

__DEFAULT_ENCODING_DIMENSIONALITY = 128

__DEFAULT_NUM_EPOCHS = 256
__DEFAULT_START_TEMP = 10.0
__DEFAULT_MIN_TEMP = 0.01
__DEFAULT_CONCRETE_ALPHA_DECAY = calculate_concrete_alpha_decay(__DEFAULT_START_TEMP, __DEFAULT_MIN_TEMP, __DEFAULT_NUM_EPOCHS)
_DEFAULT_NONLINEARITY = F.relu
_DEFAULT_NONLINEARITY_CLASS = nn.ReLU

_DEFAULT_AE_KWARGS = {
    "hidden_dims": [512],
    "encoding_dim": __DEFAULT_ENCODING_DIMENSIONALITY,
    "nonlinearity": _DEFAULT_NONLINEARITY,
    "final_nonlinearity": lambda x:x,
    "dropout_rate": 0.3,
    "bias": True,
}

_DEFAULT_COXAE_KWARGS = {
    **_DEFAULT_AE_KWARGS,
    "cox_hidden_dims": [],
}

_DEFAULT_CONCRETECOXAE_KWARGS = {
    **_DEFAULT_COXAE_KWARGS,
    "num_features": 1000,
    "start_temp": __DEFAULT_START_TEMP,
    "min_temp": __DEFAULT_MIN_TEMP,
    "alpha": __DEFAULT_CONCRETE_ALPHA_DECAY,
}

_DEFAULT_AE_OPT_KWARGS = {
    "lr":1e-3,
    "weight_decay": 1e-4
}

_DEFAULT_AE_TRAIN_KWARGS = {
    "epochs":__DEFAULT_NUM_EPOCHS,
    "noise_std": 0.2
}

_DEFAULT_DIMENSIONALITY_REDUCER = PCA(n_components=__DEFAULT_ENCODING_DIMENSIONALITY)

_DEFAULT_CLUSTERER = KMeans(2)

_DEFAULT_SCALER = StandardScaler()

_DEFAULT_INPUT_FEATURE_SELECTOR = SurvivalSelectKBest(variance_score, k=1000)

_DEFAULT_ENCODING_FEATURE_SELECTOR = DummyFeatureSelector()

_DEFAULT_COXREGRESSOR = CoxPHRegression()

_DEFAULT_PENALISED_COXREGRESSOR = CoxPHRegression(penalizer=0.1)