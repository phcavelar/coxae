from copy import deepcopy
from typing import Union

import numpy as np
import scipy as sp

from sklearn.exceptions import NotFittedError

import torch
import torch.nn.functional as F
import pycox

from .base import SurvivalClustererMixin, HazardRegressorMixin
from .architectures import CoxAutoencoder, ConcreteCoxAutoencoder
from .defaults import _DEFAULT_COXAE_KWARGS, _DEFAULT_CONCRETECOXAE_KWARGS, _DEFAULT_AE_OPT_KWARGS, _DEFAULT_AE_TRAIN_KWARGS, _DEFAULT_CLUSTERER, _DEFAULT_SCALER, _DEFAULT_INPUT_FEATURE_SELECTOR, _DEFAULT_ENCODING_FEATURE_SELECTOR
from .preprocessing import preprocess_input_to_dict, stack_dicts

class CoxAutoencoderClustering(SurvivalClustererMixin,HazardRegressorMixin):

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
        ae_kwargs = _DEFAULT_COXAE_KWARGS if ae_kwargs is None else ae_kwargs
        ae_opt_kwargs = _DEFAULT_AE_OPT_KWARGS if ae_opt_kwargs is None else ae_opt_kwargs
        ae_train_kwargs = _DEFAULT_AE_TRAIN_KWARGS if ae_train_kwargs is None else ae_train_kwargs
        clusterer = _DEFAULT_CLUSTERER if clusterer is None else clusterer
        scaler = _DEFAULT_SCALER if scaler is None else scaler
        input_feature_selector = _DEFAULT_INPUT_FEATURE_SELECTOR if input_feature_selector is None else input_feature_selector
        encoding_feature_selector = _DEFAULT_ENCODING_FEATURE_SELECTOR if encoding_feature_selector is None else encoding_feature_selector
        
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

        if not self.ae_initialised:
            self.__init_ae(X.shape[-1], **self.ae_kwargs)
        if not self.ae_opt_initialised:
            self.ae_opt_kwargs = self.ae_opt_kwargs if ae_opt_kwargs is None else ae_opt_kwargs
            self.__init_ae_opt(**self.ae_opt_kwargs)
        self.ae_train_opts = self.ae_train_opts if ae_train_opts is None else ae_train_opts

        self.__train_ae(X, durations, events, *args, **self.ae_train_opts)
        integrated_values = self.__integrate(X)
        self.encoding_feature_selector.fit(integrated_values, durations, events)
        significant_factors = self.encoding_feature_selector.transform(integrated_values)
        self.clusterer.fit(significant_factors)
        self.fitted = True
        return self
    
    def integrate(self, X: Union[np.ndarray,dict[str,np.ndarray]], durations:np.ndarray=None, events:np.ndarray=None, *args, **kwargs) -> np.ndarray:
        self.check_fitted()
        X = preprocess_input_to_dict(X)
        scaled_X = {k:self.scalers[k].transform(X[k]) for k in self.scalers}
        selected_X = {k:self.input_feature_selectors[k].transform(scaled_X[k], durations, events) for k in self.input_feature_selectors}
        stacked_X = stack_dicts(selected_X)
        integrated_values = self.__integrate(stacked_X)
        significant_factors = self.encoding_feature_selector.transform(integrated_values)
        return significant_factors
    
    def hazard(self,X: Union[np.ndarray,dict[str,np.ndarray]], durations:np.ndarray=None, events:np.ndarray=None, *args, **kwargs) -> np.ndarray:
        self.check_fitted()
        X = preprocess_input_to_dict(X)
        scaled_X = {k:self.scalers[k].transform(X[k]) for k in self.scalers}
        selected_X = {k:self.input_feature_selectors[k].transform(scaled_X[k], durations, events) for k in self.input_feature_selectors}
        stacked_X = stack_dicts(selected_X)
        return self.__hazard(stacked_X)
    
    def cluster(self, X: Union[np.ndarray,dict[str,np.ndarray]], durations:np.ndarray=None, events:np.ndarray=None, *args, **kwargs) -> np.ndarray:
        significant_factors = self.integrate(X, durations=durations, events=events, *args, **kwargs)
        return self.clusterer.predict(significant_factors)

    # AE-related functions

    def __init_ae(self, input_dim:int,
                hidden_dims:list = _DEFAULT_COXAE_KWARGS["hidden_dims"],
                encoding_dim:int = _DEFAULT_COXAE_KWARGS["encoding_dim"],
                cox_hidden_dims:list = _DEFAULT_COXAE_KWARGS["cox_hidden_dims"],
                nonlinearity:callable = _DEFAULT_COXAE_KWARGS["nonlinearity"],
                final_nonlinearity:callable = _DEFAULT_COXAE_KWARGS["final_nonlinearity"],
                dropout_rate:float = _DEFAULT_COXAE_KWARGS["dropout_rate"],
                bias:bool = _DEFAULT_COXAE_KWARGS["bias"],
                **kwargs) -> None:
        self.ae = CoxAutoencoder(
            input_dim = input_dim,
            hidden_dims = hidden_dims,
            encoding_dim = encoding_dim,
            cox_hidden_dims = cox_hidden_dims,
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
    
    def __train_ae(self, X: np.ndarray, durations: np.ndarray, events: np.ndarray, *args, epochs=_DEFAULT_AE_TRAIN_KWARGS["epochs"], noise_std=_DEFAULT_AE_TRAIN_KWARGS["noise_std"], **kwargs) -> None:
        
        tX_clean = torch.tensor(X, dtype=torch.float32)
        tT = torch.tensor(durations, dtype=torch.float32)
        tE = torch.tensor(events)

        reconstruction_losses = []
        cox_losses = []
        self.ae.train()
        for e in range(epochs):
            self.opt.zero_grad()
            
            tX_noise = tX_clean + torch.normal(torch.zeros_like(tX_clean), noise_std)
            
            y = self.ae(tX_noise)
            
            reconstruction_loss = F.mse_loss(y, tX_clean)
            
            cox_loss = pycox.models.loss.cox_ph_loss(self.ae.cox(tX_clean), tT, tE)
            
            loss = reconstruction_loss + cox_loss
            
            loss.backward()
            self.opt.step()

            reconstruction_losses.append(reconstruction_loss.detach().numpy().item())
            cox_losses.append(cox_loss.detach().numpy().item())
        self.reconstruction_losses = reconstruction_losses
        self.cox_losses = cox_losses

    def __integrate(self,X):
        self.ae.eval()
        return self.ae.encode(torch.tensor(X, dtype=torch.float32)).detach().numpy()

    def __hazard(self,X):
        self.ae.eval()
        return np.exp(self.ae.cox(torch.tensor(X, dtype=torch.float32)).detach().numpy())
    
    def check_fitted(self):
        if not self.fitted:
            raise NotFittedError("This {name} instance is not fitted yet. Call 'fit' with appropriate arguments before using this estimator.".format(name=type(self).__name__))


class ConcreteCoxAutoencoderClustering(SurvivalClustererMixin,HazardRegressorMixin):

    def __init__(self,
            *args,
            d_in:int = None,
            ae_kwargs:dict = None,
            ae_opt_kwargs = None,
            ae_train_kwargs = None,
            clusterer = None,
            scaler = None,
            input_feature_selector = None,
            encoding_feature_selector = None,
            cox_regressor = None,
            **kwargs):
        super().__init__(*args, **kwargs)
        # Avoiding mutable default values
        ae_kwargs = _DEFAULT_COXAE_KWARGS if ae_kwargs is None else ae_kwargs
        ae_opt_kwargs = _DEFAULT_AE_OPT_KWARGS if ae_opt_kwargs is None else ae_opt_kwargs
        ae_train_kwargs = _DEFAULT_AE_TRAIN_KWARGS if ae_train_kwargs is None else ae_train_kwargs
        clusterer = _DEFAULT_CLUSTERER if clusterer is None else clusterer
        scaler = _DEFAULT_SCALER if scaler is None else scaler
        input_feature_selector = _DEFAULT_INPUT_FEATURE_SELECTOR if input_feature_selector is None else input_feature_selector
        encoding_feature_selector = _DEFAULT_ENCODING_FEATURE_SELECTOR if encoding_feature_selector is None else encoding_feature_selector
        
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

        if not self.ae_initialised:
            self.__init_ae(X.shape[-1], **self.ae_kwargs)
        if not self.ae_opt_initialised:
            self.ae_opt_kwargs = self.ae_opt_kwargs if ae_opt_kwargs is None else ae_opt_kwargs
            self.__init_ae_opt(**self.ae_opt_kwargs)

        self.__train_ae(X, durations, events, *args, **self.ae_train_opts)
        integrated_values = self.__integrate(X)
        self.encoding_feature_selector.fit(integrated_values, durations, events)
        significant_factors = self.encoding_feature_selector.transform(integrated_values)
        self.clusterer.fit(significant_factors)
        self.fitted = True
        return self
    
    def integrate(self, X:Union[np.ndarray,dict[str,np.ndarray]], durations:np.ndarray=None, events:np.ndarray=None, *args, **kwargs) -> np.ndarray:
        self.check_fitted()
        X = preprocess_input_to_dict(X)
        scaled_X = {k:self.scalers[k].transform(X[k]) for k in self.scalers}
        selected_X = {k:self.input_feature_selectors[k].transform(scaled_X[k], durations, events) for k in self.input_feature_selectors}
        stacked_X = stack_dicts(selected_X)
        integrated_values = self.__integrate(stacked_X)
        significant_factors = self.encoding_feature_selector.transform(integrated_values)
        return significant_factors
    
    def hazard(self,X:Union[np.ndarray,dict[str,np.ndarray]], durations:np.ndarray=None, events:np.ndarray=None, *args, **kwargs) -> np.ndarray:
        self.check_fitted()
        X = preprocess_input_to_dict(X)
        scaled_X = {k:self.scalers[k].transform(X[k]) for k in self.scalers}
        selected_X = {k:self.input_feature_selectors[k].transform(scaled_X[k], durations, events) for k in self.input_feature_selectors}
        stacked_X = stack_dicts(selected_X)
        return self.__hazard(stacked_X)
    
    def cluster(self, X:Union[np.ndarray,dict[str,np.ndarray]], durations:np.ndarray=None, events:np.ndarray=None, *args, **kwargs) -> np.ndarray:
        significant_factors = self.integrate(X, durations=durations, events=events, *args, **kwargs)
        return self.clusterer.predict(significant_factors)

    # AE-related functions

    def __init_ae(self, input_dim:int,
                num_features:int = _DEFAULT_CONCRETECOXAE_KWARGS["num_features"],
                hidden_dims:list = _DEFAULT_CONCRETECOXAE_KWARGS["hidden_dims"],
                encoding_dim:int = _DEFAULT_CONCRETECOXAE_KWARGS["encoding_dim"],
                cox_hidden_dims:list = _DEFAULT_CONCRETECOXAE_KWARGS["cox_hidden_dims"],
                nonlinearity:callable = _DEFAULT_CONCRETECOXAE_KWARGS["nonlinearity"],
                final_nonlinearity:callable = _DEFAULT_CONCRETECOXAE_KWARGS["final_nonlinearity"],
                dropout_rate:float = _DEFAULT_CONCRETECOXAE_KWARGS["dropout_rate"],
                bias:bool = _DEFAULT_CONCRETECOXAE_KWARGS["bias"],
                start_temp:float = _DEFAULT_CONCRETECOXAE_KWARGS["start_temp"],
                min_temp:float = _DEFAULT_CONCRETECOXAE_KWARGS["min_temp"],
                alpha:float = _DEFAULT_CONCRETECOXAE_KWARGS["alpha"],
                **kwargs) -> None:
        self.ae = ConcreteCoxAutoencoder(
            input_dim = input_dim,
            hidden_dims = hidden_dims,
            encoding_dim = encoding_dim,
            cox_hidden_dims = cox_hidden_dims,
            nonlinearity = nonlinearity,
            final_nonlinearity = final_nonlinearity,
            dropout_rate = dropout_rate,
            bias = bias,
            start_temp = start_temp,
            min_temp = min_temp,
            alpha = alpha,
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
        tT = torch.tensor(durations, dtype=torch.float32)
        tE = torch.tensor(events)

        reconstruction_losses = []
        cox_losses = []
        self.ae.train()
        for e in range(epochs):
            self.opt.zero_grad()
            
            tX_noise = tX_clean + torch.normal(torch.zeros_like(tX_clean), noise_std)
            
            y = self.ae(tX_noise)
            
            reconstruction_loss = F.mse_loss(y, tX_clean)
            
            cox_loss = pycox.models.loss.cox_ph_loss(self.ae.cox(tX_clean), tT, tE)
            
            loss = reconstruction_loss + cox_loss
            
            loss.backward()
            self.opt.step()

            self.ae.update_temperature()

            reconstruction_losses.append(reconstruction_loss.detach().numpy().item())
            cox_losses.append(cox_loss.detach().numpy().item())
        self.reconstruction_losses = reconstruction_losses
        self.cox_losses = cox_losses

    def __integrate(self,X):
        self.ae.eval()
        return self.ae.encode(torch.tensor(X, dtype=torch.float32)).detach().numpy()

    def __hazard(self,X):
        self.ae.eval()
        return np.exp(self.ae.cox(torch.tensor(X, dtype=torch.float32)).detach().numpy())

    def check_fitted(self):
        if not self.fitted:
            raise NotFittedError("This {name} instance is not fitted yet. Call 'fit' with appropriate arguments before using this estimator.".format(name=type(self).__name__))