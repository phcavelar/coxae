import numpy as np
import pandas as pd

from sklearn.cluster import KMeans
from sklearn.exceptions import NotFittedError

import lifelines

import torch
import torch.nn.functional as F

from ..base import SurvivalClustererMixin, HazardPredictorMixin
from ..architectures import Autoencoder
from ..significant_factor_selection import get_significant_factors, get_most_significant_factor_combinations

_DEFAULT_AE_KWARGS = {
    "hidden_dims": [512],
    "encoding_dim": 128,
    "nonlinearity": F.relu,
    "final_nonlinearity": lambda x:x,
    "dropout_rate": 0.3,
    "bias": True,
}

_DEFAULT_AE_OPT_KWARGS = {
    "lr":1e-3,
    "weight_decay": 1e-4
}

_DEFAULT_AE_TRAIN_KWARGS = {
    "epochs":256,
    "noise_std": 0.2
}

class AutoencoderClustering(SurvivalClustererMixin,HazardPredictorMixin):

    def __init__(self,
            *args,
            d_in:int = None,
            ae_kwargs:dict = _DEFAULT_AE_KWARGS,
            ae_opt_kwargs = _DEFAULT_AE_OPT_KWARGS,
            ae_train_kwargs = _DEFAULT_AE_TRAIN_KWARGS,
            only_significant:bool = True,
            limit_significant:int = -1,
            get_most_significant_combination_time_limit:float = 0.0,
            clusterer = KMeans(2),
            **kwargs):
        super().__init__(*args, **kwargs)
        
        self.only_significant = only_significant
        self.limit_significant = limit_significant
        self.get_most_significant_combination_time_limit = get_most_significant_combination_time_limit

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

    # Mixin functions

    def fit(self, X: np.array, durations: np.array, events: np.array, *args, ae_opt_kwargs=None, ae_train_opts=None, **kwargs):
        if not self.ae_initialised:
            self.__init_ae(X.shape[-1], **self.ae_kwargs)
        if not self.ae_opt_initialised:
            self.ae_opt_kwargs = self.ae_opt_kwargs if ae_opt_kwargs is None else ae_opt_kwargs
            self.__init_ae_opt(**self.ae_opt_kwargs)
        self.ae_train_opts = self.ae_train_opts if ae_train_opts is None else ae_train_opts
        self.__train_ae(X, durations, events, *args, **self.ae_train_opts)
        integrated_values = self.__integrate(X)
        if self.only_significant:
            self.significant_indexes, self.significant_indexes_p_values = get_significant_factors(integrated_values, durations, events)
            ordering = np.argsort(self.significant_indexes_p_values).tolist()
            self.significant_indexes = [self.significant_indexes[i] for i in ordering[:self.limit_significant]]
            self.significant_indexes_p_values = [self.significant_indexes_p_values[i] for i in ordering[:self.limit_significant]]
            if self.get_most_significant_combination_time_limit > 0:
                significant_indexes_combinations, significant_indexes_combinations_p_values = get_most_significant_factor_combinations(integrated_values, durations, events, self.significant_indexes, time_limit=self.self.get_most_significant_combination_time_limit)
                combination_ordering = np.argsort(significant_indexes_combinations_p_values).tolist()
                best_combination = significant_indexes_combinations[combination_ordering[0]]
                self.significant_indexes_p_values = [self.significant_indexes_p_values[i] for i in self.significant_indexes if i in best_combination]
                self.significant_indexes = best_combination
            # Add a guard in case no factors are found to be significant
            self.significant_indexes = [i for i in range(integrated_values.shape[1])] if self.significant_indexes == [] else self.significant_indexes
        else:
            self.significant_indexes = [i for i in range(integrated_values.shape[1])]
        significant_factors = integrated_values[:,self.significant_indexes]
        self.cox_ph_model = lifelines.CoxPHFitter().fit(
            pd.DataFrame(
                {
                    **{i: integrated_values[:, i] for i in self.significant_indexes},
                    "durations": durations,
                    "events": events,
                }
            ),
            "durations",
            "events"
        )
        self.clusterer.fit(significant_factors)
        self.fitted = True
    
    def predict(self, X: np.array, *args, **kwargs):
        self.check_fitted()
        integrated_values = self.__integrate(X)
        significant_factors = integrated_values[:,self.significant_indexes]
        clusters = self.clusterer.predict(significant_factors)
        return clusters

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
    
    def __train_ae(self, X: np.array, durations: np.array, events: np.array, *args, epochs=_DEFAULT_AE_TRAIN_KWARGS["epochs"], noise_std=_DEFAULT_AE_TRAIN_KWARGS["noise_std"], **kwargs) -> None:
        
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

    def integrate(self,X):
        self.check_fitted()
        return self.__integrate(X)

    def __calculate_hazard(self,X):
        self.ae.eval()
        integrated_values = self.__integrate(X)
        return np.exp(
            self.cox_ph_model.predict_log_partial_hazard(
                pd.DataFrame(
                    {
                        **{i: integrated_values[:, i] for i in self.significant_indexes},
                    }
                )
            )
        )
    def calculate_hazard(self,X):
        self.check_fitted()
        return self.__calculate_hazard(X)
    
    def check_fitted(self):
        if not self.fitted:
            raise NotFittedError("This {name} instance is not fitted yet. Call 'fit' with appropriate arguments before using this estimator.".format(name=type(self).__name__))