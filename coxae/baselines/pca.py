from copy import deepcopy
from typing import Union

import numpy as np
import pandas as pd

from sklearn.exceptions import NotFittedError

import lifelines

from ..base import SurvivalClustererMixin, HazardRegressorMixin
from ..defaults import _DEFAULT_DIMENSIONALITY_REDUCER, _DEFAULT_CLUSTERER, _DEFAULT_SCALER, _DEFAULT_INPUT_FEATURE_SELECTOR, _DEFAULT_ENCODING_FEATURE_SELECTOR, _DEFAULT_COXREGRESSOR 
from ..preprocessing import preprocess_input_to_dict, stack_dicts

class PCAClustering(SurvivalClustererMixin,HazardRegressorMixin):

    def __init__(self,
            *args,
            dimensionality_reducer = None,
            clusterer = None,
            scaler = None,
            input_feature_selector = None,
            encoding_feature_selector = None,
            cox_regressor = None,
            **kwargs):
        super().__init__(*args, **kwargs)
        # Avoiding mutable default values
        dimensionality_reducer = _DEFAULT_DIMENSIONALITY_REDUCER if dimensionality_reducer is None else dimensionality_reducer
        clusterer = _DEFAULT_CLUSTERER if clusterer is None else clusterer
        scaler = _DEFAULT_SCALER if scaler is None else scaler
        input_feature_selector = _DEFAULT_INPUT_FEATURE_SELECTOR if input_feature_selector is None else input_feature_selector
        encoding_feature_selector = _DEFAULT_ENCODING_FEATURE_SELECTOR if encoding_feature_selector is None else encoding_feature_selector
        cox_regressor = _DEFAULT_COXREGRESSOR if cox_regressor is None else cox_regressor
        
        if not hasattr(dimensionality_reducer, "fit") or not hasattr(dimensionality_reducer, "transform"):
            raise ValueError('Only dimensionality reduction methods with separate "fit" and "transform" methods should be used by this class')
        self.dimensionality_reducer = dimensionality_reducer

        if not hasattr(clusterer, "fit") or not hasattr(clusterer, "predict"):
            raise ValueError('Only clusterers with separate "fit" and "predict" methods should be used by this class')
        self.clusterer = clusterer

        self.scaler = scaler
        self.scalers = {}
        self.input_feature_selector = input_feature_selector
        self.input_feature_selectors = {}
        self.encoding_feature_selector = encoding_feature_selector
        self.cox_regressor = cox_regressor

        self.fitted = False

    # Mixin functions
    
    def __fit_dict_steps(self, X:dict[str,np.ndarray]) -> None:
        self.scalers = {k:deepcopy(self.scaler) for k in X}
        self.input_feature_selectors = {k:deepcopy(self.input_feature_selector) for k in X}

    def fit(self, X: Union[np.ndarray,dict[str,np.ndarray]], durations: np.ndarray, events: np.ndarray, *args, ae_opt_kwargs=None, ae_train_opts=None, **kwargs):
        X = preprocess_input_to_dict(X)
        self.__fit_dict_steps(X)
        X = {k:self.scalers[k].fit_transform(X[k]) for k in self.scalers}
        X = {k:self.input_feature_selectors[k].fit_transform(X[k], durations, events) for k in self.input_feature_selectors}
        X = stack_dicts(X)
        integrated_values = self.dimensionality_reducer.fit_transform(X)
        self.encoding_feature_selector.fit(integrated_values, durations, events)
        significant_factors = self.encoding_feature_selector.transform(integrated_values)
        self.cox_regressor.fit(significant_factors, durations, events)
        self.clusterer.fit(significant_factors)
        self.fitted = True
        return self

    def integrate(self, X:Union[np.ndarray,dict[str,np.ndarray]], durations:np.ndarray=None, events:np.ndarray=None, *args, **kwargs) -> np.ndarray:
        self.check_fitted()
        X = preprocess_input_to_dict(X)
        scaled_X = {k:self.scalers[k].transform(X[k]) for k in self.scalers}
        selected_X = {k:self.input_feature_selectors[k].transform(scaled_X[k], durations, events) for k in self.input_feature_selectors}
        stacked_X = stack_dicts(selected_X)
        integrated_values = self.dimensionality_reducer.transform(stacked_X)
        significant_factors = self.encoding_feature_selector.transform(integrated_values)
        return significant_factors
    
    def hazard(self, X:Union[np.ndarray,dict[str,np.ndarray]], durations:np.ndarray=None, events:np.ndarray=None, *args, **kwargs) -> np.ndarray:
        significant_factors = self.integrate(X, durations=durations, events=events, *args, **kwargs)
        return self.cox_regressor.hazard(significant_factors)
    
    def cluster(self, X:Union[np.ndarray,dict[str,np.ndarray]], durations:np.ndarray=None, events:np.ndarray=None, *args, **kwargs) -> np.ndarray:
        significant_factors = self.integrate(X, durations=durations, events=events, *args, **kwargs)
        return self.clusterer.predict(significant_factors)

    def check_fitted(self):
        if not self.fitted:
            raise NotFittedError("This {name} instance is not fitted yet. Call 'fit' with appropriate arguments before using this estimator.".format(name=type(self).__name__))