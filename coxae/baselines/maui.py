from copy import deepcopy
from typing import Union

import numpy as np
import pandas as pd

from sklearn.cluster import KMeans
from sklearn.exceptions import NotFittedError

import lifelines, lifelines.exceptions

from ..base import SurvivalClustererMixin, HazardRegressorMixin
from ..defaults import _DEFAULT_SCALER, _DEFAULT_INPUT_FEATURE_SELECTOR, _DEFAULT_CLUSTERER, _DEFAULT_COXREGRESSOR, _DEFAULT_PENALISED_COXREGRESSOR
from ..significant_factor_selection import get_significant_factors
from ..preprocessing import preprocess_input_to_dict, stack_dicts

import warnings

try:
    import maui
    import maui.utils

    _DEFAULT_MAUI_KWARGS = {
        "n_hidden": [512],
        "n_latent": 128,
        "epochs": 400,
    }

    _DEFAULT_MAUI_MAPPINGS = {
        "all": lambda x: pd.DataFrame(x[:,:].T)
    }

    class MauiClustering(SurvivalClustererMixin,HazardRegressorMixin):

        def __init__(self,
                *args,
                maui_kwargs:dict = None,
                maui_omics_layer_mappings:dict = None,
                scaler = None,
                input_feature_selector = None,
                clusterer = None,
                cox_regressor = None,
                significance_alpha = 0.05,
                only_significant:bool = True,
                **kwargs):
            super().__init__(*args, **kwargs)
            maui_kwargs = _DEFAULT_MAUI_KWARGS if maui_kwargs is None else maui_kwargs
            maui_omics_layer_mappings = _DEFAULT_MAUI_MAPPINGS if maui_omics_layer_mappings is None else maui_omics_layer_mappings
            scaler = _DEFAULT_SCALER if scaler is None else scaler
            input_feature_selector = _DEFAULT_INPUT_FEATURE_SELECTOR if input_feature_selector is None else input_feature_selector
            clusterer = _DEFAULT_CLUSTERER if clusterer is None else clusterer
            cox_regressor = _DEFAULT_COXREGRESSOR if cox_regressor is None else cox_regressor
            
            self.only_significant = only_significant
            self.maui_kwargs = maui_kwargs
            self.maui_omics_layer_mappings = maui_omics_layer_mappings
            self.__init_maui(**maui_kwargs)
            
            if not hasattr(clusterer, "fit") or not hasattr(clusterer, "predict"):
                raise ValueError('Only clusterers with separate "fit" and "predict" methods should be used by this class')
            self.clusterer = clusterer

            self.scaler = scaler
            self.scalers = {}
            self.input_feature_selector = input_feature_selector
            self.input_feature_selectors = {}
            self.cox_regressor = cox_regressor

            self.significance_alpha = significance_alpha

            self.fitted = False

        # Mixin functions

        def __fit_dict_steps(self, X:dict[str,np.ndarray]) -> None:
            self.scalers = {k:deepcopy(self.scaler) for k in X}
            self.input_feature_selectors = {k:deepcopy(self.input_feature_selector) for k in X}

        def fit(self, X:Union[np.ndarray,dict[str,np.ndarray]], durations:np.ndarray, events:np.ndarray, *args, **kwargs):
            X = preprocess_input_to_dict(X)
            self.__fit_dict_steps(X)
            selected_X = {k:self.input_feature_selectors[k].fit_transform(X[k], durations, events) for k in self.input_feature_selectors}
            X = {k:self.scalers[k].fit_transform(selected_X[k]) for k in self.scalers}

            if X.keys != self.maui_omics_layer_mappings.keys and "all" in self.maui_omics_layer_mappings:
                self.maui_omics_layer_mappings = {
                    k: lambda x: pd.DataFrame(x[:,:].T)
                    for k in X
                }

            z = self.maui_model.fit_transform(self.__preprocess_inputs(X)).values
            if self.only_significant:
                cph_p_values = [
                    lifelines.CoxPHFitter()
                    .fit(
                        pd.DataFrame(
                            {
                                "LF": z[:, i],
                                "durations": durations,
                                "events": events,
                            }
                        ),
                        "durations",
                        "events"
                    )
                    .summary.loc["LF"].p
                    for i in range(self.maui_model.n_latent)
                ]
                self.significant_factors = [
                    i for i in range(self.maui_model.n_latent) if cph_p_values[i] < self.significance_alpha
                ]
                # Add a guard in case no factors are found to be significant
                self.significant_factors = list(range(self.maui_model.n_latent)) if len(self.significant_factors) == 0 else self.significant_factors
            else:
                self.significant_factors = [
                    i for i in range(self.maui_model.n_latent)
                ]
            # This check was not done by the original Maui authors but was deemed necessary for evaluating their model:
            try:
                self.cox_regressor.fit(z[:,self.significant_factors], durations, events)
            except lifelines.exceptions.ConvergenceError:
                self.cox_regressor = _DEFAULT_PENALISED_COXREGRESSOR
                self.cox_regressor.fit(z[:,self.significant_factors], durations, events)
            self.clusterer.fit_predict(z[:,self.significant_factors])
            self.fitted = True
    
        def __preprocess_input_for_maui(self, X: dict[str,np.ndarray], durations:np.ndarray=None, events:np.ndarray=None) -> np.ndarray:
            selected_X = {k:self.input_feature_selectors[k].transform(X[k], durations, events) for k in self.input_feature_selectors}
            scaled_X = {k:self.scalers[k].transform(selected_X[k]) for k in self.scalers}
            return scaled_X

        def integrate(self, X:Union[np.ndarray,dict[str,np.ndarray]], durations:np.ndarray=None, events:np.ndarray=None, *args, **kwargs) -> np.ndarray:
            self.check_fitted()
            X = preprocess_input_to_dict(X)
            scaled_X = self.__preprocess_input_for_maui(X, durations, events)
            return self.__integrate(scaled_X)[:,self.significant_factors]

        def hazard(self,X:Union[np.ndarray,dict[str,np.ndarray]], durations:np.ndarray=None, events:np.ndarray=None, *args, **kwargs) -> np.ndarray:
            significant_factors = self.integrate(X, durations=durations, events=events, *args, **kwargs)
            return self.cox_regressor.hazard(significant_factors)
        
        def cluster(self, X:Union[np.ndarray,dict[str,np.ndarray]], durations:np.ndarray=None, events:np.ndarray=None, *args, **kwargs) -> np.ndarray:
            significant_factors = self.integrate(X, durations=durations, events=events, *args, **kwargs)
            return self.clusterer.predict(significant_factors)

        def __preprocess_inputs(self, X:dict[str,np.ndarray]) -> dict[str,pd.DataFrame]:
            return {k: self.maui_omics_layer_mappings[k](X[k]) for k in self.maui_omics_layer_mappings}

        def __init_maui(self, **maui_kwargs):
            self.maui_model = maui.Maui(**maui_kwargs)

        def __integrate(self,X:dict[str,np.ndarray]):
            return self.maui_model.transform(self.__preprocess_inputs(X)).values
        
        def check_fitted(self):
            if not self.fitted:
                raise NotFittedError("This {name} instance is not fitted yet. Call 'fit' with appropriate arguments before using this estimator.".format(name=type(self).__name__))
    
except ImportError as e:
    warnings.warn("Failed to import the maui library! Maui baseline model not built! {}".format(e))
except Exception as e:
    warnings.warn("Failed to build the maui baseline model! {}".format(e))