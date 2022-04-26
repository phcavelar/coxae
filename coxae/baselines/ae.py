from copy import deepcopy
from typing import Union

import numpy as np
import pandas as pd

from sklearn.cluster import KMeans
from sklearn.exceptions import NotFittedError

import lifelines

import torch
import torch.nn.functional as F

from ..base import SurvivalClustererMixin, HazardRegressorMixin
from ..architectures import Autoencoder
from ..significant_factor_selection import get_significant_factors, get_most_significant_factor_combinations
from ..defaults import _DEFAULT_AE_KWARGS, _DEFAULT_AE_OPT_KWARGS, _DEFAULT_AE_TRAIN_KWARGS, _DEFAULT_CLUSTERER, _DEFAULT_SCALER, _DEFAULT_INPUT_FEATURE_SELECTOR, _DEFAULT_ENCODING_FEATURE_SELECTOR, _DEFAULT_COXREGRESSOR
from ..preprocessing import preprocess_input_to_dict, stack_dicts

class AutoencoderClustering(SurvivalClustererMixin,HazardRegressorMixin):

    def __init__(self,
            *args,
            d_in:int = None,
            ae_kwargs:dict = None,
            ae_opt_kwargs:dict = None,
            ae_train_kwargs:dict = None,
            clusterer = None,
            scaler = None,
            input_feature_selector = None,
            encoding_feature_selector = None,
            cox_regressor = None,
            **kwargs):
        super().__init__(*args, **kwargs)
        # Avoiding mutable default values
        ae_kwargs = _DEFAULT_AE_KWARGS if ae_kwargs is None else ae_kwargs
        ae_opt_kwargs = _DEFAULT_AE_OPT_KWARGS if ae_opt_kwargs is None else ae_opt_kwargs
        ae_train_kwargs = _DEFAULT_AE_TRAIN_KWARGS if ae_train_kwargs is None else ae_train_kwargs
        clusterer = _DEFAULT_CLUSTERER if clusterer is None else clusterer
        scaler = _DEFAULT_SCALER if scaler is None else scaler
        input_feature_selector = _DEFAULT_INPUT_FEATURE_SELECTOR if input_feature_selector is None else input_feature_selector
        encoding_feature_selector = _DEFAULT_ENCODING_FEATURE_SELECTOR if encoding_feature_selector is None else encoding_feature_selector
        cox_regressor = _DEFAULT_COXREGRESSOR if cox_regressor is None else cox_regressor
        
        # Initialisation
        self.ae_kwargs = ae_kwargs
        if d_in is not None:
            self.__init_ae(d_in, **ae_kwargs)
            self.ae_initialised = True
        else:
            self.ae_initialised = False

        self.ae_opt_kwargs = ae_opt_kwargs
        if self.ae_initialised and self.ae_opt_kwargs is not None:
            self.__init_ae_opt(**ae_opt_kwargs)
            self.ae_opt_initialised = True
        else:
            self.ae_opt_initialised = False
        
        self.ae_train_opts = ae_train_kwargs

        if not hasattr(clusterer, "fit") or not hasattr(clusterer, "predict"):
            raise ValueError('Only clusterers with separate "fit" and "predict" methods should be used by this class')
        self.clusterer = clusterer

        self.fitted = False

        self.scaler = scaler
        self.scalers = {}
        self.input_feature_selector = input_feature_selector
        self.input_feature_selectors = {}
        self.encoding_feature_selector = encoding_feature_selector
        self.cox_regressor = cox_regressor

    # Mixin functions
    
    def __fit_dict_steps(self, X:dict[str,np.ndarray]) -> None:
        self.scalers = {k:deepcopy(self.scaler) for k in X}
        self.input_feature_selectors = {k:deepcopy(self.input_feature_selector) for k in X}

    def fit(self, X: Union[np.ndarray,dict[str,np.ndarray]], durations: np.ndarray, events: np.ndarray, *args, ae_opt_kwargs=None, ae_train_opts=None, **kwargs):
        X = preprocess_input_to_dict(X)
        self.__fit_dict_steps(X)
        selected_X = {k:self.input_feature_selectors[k].fit_transform(X[k], durations, events) for k in self.input_feature_selectors}
        scaled_X = {k:self.scalers[k].fit_transform(selected_X[k]) for k in self.scalers}
        X = stack_dicts(scaled_X)
        
        if not self.ae_initialised:
            self.__init_ae(X.shape[-1], **self.ae_kwargs)
        if not self.ae_opt_initialised:
            self.ae_opt_kwargs = self.ae_opt_kwargs if ae_opt_kwargs is None else ae_opt_kwargs
            self.__init_ae_opt(**self.ae_opt_kwargs)

        self.__train_ae(X, durations, events, *args, **self.ae_train_opts)
        integrated_values = self.__integrate(X)
        self.encoding_feature_selector.fit(integrated_values, durations, events)
        significant_factors = self.encoding_feature_selector.transform(integrated_values)
        self.cox_regressor.fit(significant_factors, durations, events)
        self.clusterer.fit(significant_factors)
        self.fitted = True
        return self
    
    def __preprocess_input_for_ae(self, X: dict[str,np.ndarray], durations:np.ndarray=None, events:np.ndarray=None) -> np.ndarray:
        selected_X = {k:self.input_feature_selectors[k].transform(X[k], durations, events) for k in self.input_feature_selectors}
        scaled_X = {k:self.scalers[k].transform(selected_X[k]) for k in self.scalers}
        stacked_X = stack_dicts(scaled_X)
        return stacked_X

    def integrate(self, X:Union[np.ndarray,dict[str,np.ndarray]], durations:np.ndarray=None, events:np.ndarray=None, *args, **kwargs) -> np.ndarray:
        self.check_fitted()
        X = preprocess_input_to_dict(X)
        stacked_X = self.__preprocess_input_for_ae(X, durations, events)
        integrated_values = self.__integrate(stacked_X)
        significant_factors = self.encoding_feature_selector.transform(integrated_values)
        return significant_factors
    
    def hazard(self, X:Union[np.ndarray,dict[str,np.ndarray]], durations:np.ndarray=None, events:np.ndarray=None, *args, **kwargs) -> np.ndarray:
        significant_factors = self.integrate(X, durations=durations, events=events, *args, **kwargs)
        return self.cox_regressor.hazard(significant_factors)
    
    def cluster(self, X:Union[np.ndarray,dict[str,np.ndarray]], durations:np.ndarray=None, events:np.ndarray=None, *args, **kwargs) -> np.ndarray:
        significant_factors = self.integrate(X, durations=durations, events=events, *args, **kwargs)
        return self.clusterer.predict(significant_factors)

    # AE-related functions

    def __init_ae(self, input_dim:int,
                hidden_dims:list = _DEFAULT_AE_KWARGS["hidden_dims"],
                encoding_dim:int = _DEFAULT_AE_KWARGS["encoding_dim"],
                nonlinearity:callable = _DEFAULT_AE_KWARGS["nonlinearity"],
                final_nonlinearity:callable = _DEFAULT_AE_KWARGS["final_nonlinearity"],
                dropout_rate:float = _DEFAULT_AE_KWARGS["dropout_rate"],
                bias:bool = _DEFAULT_AE_KWARGS["bias"],
                **kwargs) -> None:
        self.ae = Autoencoder(
            input_dim = input_dim,
            hidden_dims = hidden_dims,
            encoding_dim = encoding_dim,
            nonlinearity = nonlinearity,
            final_nonlinearity = final_nonlinearity,
            dropout_rate = dropout_rate,
            bias = bias, 
            **kwargs
        )

    def __init_ae_opt(self, lr=_DEFAULT_AE_OPT_KWARGS["lr"], weight_decay=_DEFAULT_AE_OPT_KWARGS["weight_decay"], **kwargs) -> None:
        self.opt = torch.optim.Adam(
            self.ae.parameters(),
            lr=lr,
            weight_decay=weight_decay,
            **kwargs
        )
    
    def __train_ae(self, X:np.ndarray, durations:np.ndarray, events:np.ndarray, *args, epochs=_DEFAULT_AE_TRAIN_KWARGS["epochs"], noise_std=_DEFAULT_AE_TRAIN_KWARGS["noise_std"], **kwargs) -> None:
        
        tX_clean = torch.tensor(X, dtype=torch.float32)

        losses = []
        self.ae.train()
        for e in range(epochs):
            self.opt.zero_grad()
            
            tX_noise = tX_clean + torch.normal(torch.zeros_like(tX_clean), noise_std)
            
            y = self.ae(tX_noise)
            
            reconstruction_loss = F.mse_loss(y, tX_clean)
            
            loss = reconstruction_loss
            
            loss.backward()
            self.opt.step()

            losses.append(loss.detach().numpy().item())

    def __integrate(self,X):
        self.ae.eval()
        return self.ae.encode(torch.tensor(X, dtype=torch.float32)).detach().numpy()
    
    def check_fitted(self):
        if not self.fitted:
            raise NotFittedError("This {name} instance is not fitted yet. Call 'fit' with appropriate arguments before using this estimator.".format(name=type(self).__name__))