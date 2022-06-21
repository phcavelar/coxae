from copy import deepcopy
from typing import Union

import numpy as np
import pandas as pd

from sklearn.cluster import KMeans
from sklearn.exceptions import NotFittedError

import lifelines, lifelines.exceptions

from ..base import SurvivalClustererMixin, HazardRegressorMixin
from ..defaults import _DEFAULT_SCALER, _DEFAULT_INPUT_FEATURE_SELECTOR, _DEFAULT_CLUSTERER, _DEFAULT_COXREGRESSOR, _DEFAULT_PENALISED_COXREGRESSOR, _DEFAULT_COXAE_KWARGS, _DEFAULT_AE_TRAIN_KWARGS, _DEFAULT_AE_OPT_KWARGS, _DEFAULT_NONLINEARITY_CLASS
from ..significant_factor_selection import get_significant_factors
from ..preprocessing import preprocess_input_to_dict, stack_dicts, stack_dicts_indexes

import warnings

try:
    # Code taken or adapted from the original publication
    from hierae.model.autoencoders import HierarchicalSAE, HierarchicalSAENet
    from hierae.utils.utils import hierarchical_sae_criterion
    from skorch.callbacks import EarlyStopping, LRScheduler, Checkpoint
    import torch.optim

    _DEFAULT_HIERAE_KWARGS = {
        "max_epochs": _DEFAULT_AE_TRAIN_KWARGS["epochs"],
        "lr": _DEFAULT_AE_OPT_KWARGS["lr"],
        "optimizer": torch.optim.Adam,
        "verbose": 0,
        "batch_size": -1,
        "module__block_embedding_dimension": _DEFAULT_COXAE_KWARGS["encoding_dim"],
        "module__block_hidden_layers": len(_DEFAULT_COXAE_KWARGS["hidden_dims"]),
        "module__block_hidden_layer_size":  _DEFAULT_COXAE_KWARGS["hidden_dims"][0] if len(_DEFAULT_COXAE_KWARGS["hidden_dims"])>0 else 128,
        "module__common_embedding_dimension": _DEFAULT_COXAE_KWARGS["encoding_dim"],
        "module__common_hidden_layers": 0,
        "module__common_hidden_layer_size": _DEFAULT_COXAE_KWARGS["hidden_dims"][0] if len(_DEFAULT_COXAE_KWARGS["hidden_dims"])>0 else 128,
        "module__block_activation": _DEFAULT_NONLINEARITY_CLASS,
        "module__common_activation": _DEFAULT_NONLINEARITY_CLASS,
        "module__hazard_hidden_layer_size": _DEFAULT_COXAE_KWARGS["cox_hidden_dims"][0] if len(_DEFAULT_COXAE_KWARGS["cox_hidden_dims"])>0 else 64,
        "module__hazard_activation": _DEFAULT_NONLINEARITY_CLASS,
        "module__hazard_hidden_layers": len(_DEFAULT_COXAE_KWARGS["cox_hidden_dims"]),
        "module__lambda_q": 0.001,
        #"train_split": StratifiedSkorchSurvivalSplit(
        #    10, stratified=True
        #),
        "callbacks": [
            #("seed", FixRandomSeed(config["seed"])),
            (
                "es",
                EarlyStopping(
                    patience=10,
                    monitor="valid_loss",
                    #load_best=True, # Doesn't seem to have a "load_best" parameter
                ),
            ),
            (
                "sched",
                LRScheduler(
                    torch.optim.lr_scheduler.ReduceLROnPlateau,
                    patience=3,
                    cooldown=5,
                    monitor="valid_loss",
                    verbose=False,
                ),
            ),
            ( # This was aadded since EarlyStopping didn't seem to have a "load_best" parameter
                "ckp",
                Checkpoint(
                    monitor='valid_loss_best',
                    f_params='hierae_params.pt',
                    f_optimizer='hierae_optimizer.pt',
                    f_criterion='hierae_criterion.pt',
                    f_history='hierae_history.json',
                    load_best=True,
                ),
            ),
        ],
    }

    class HierAEClustering(SurvivalClustererMixin,HazardRegressorMixin):

        def __init__(self,
                *args,
                hierae_kwargs:dict = None,
                scaler = None,
                input_feature_selector = None,
                clusterer = None,
                use_separate_cox_regressor = False,
                cox_regressor = None,
                significance_alpha = 0.05,
                only_significant:bool = False,
                **kwargs):
            super().__init__(*args, **kwargs)
            hierae_kwargs = _DEFAULT_HIERAE_KWARGS if hierae_kwargs is None else hierae_kwargs
            scaler = _DEFAULT_SCALER if scaler is None else scaler
            input_feature_selector = _DEFAULT_INPUT_FEATURE_SELECTOR if input_feature_selector is None else input_feature_selector
            clusterer = _DEFAULT_CLUSTERER if clusterer is None else clusterer
            cox_regressor = _DEFAULT_COXREGRESSOR if cox_regressor is None and not use_separate_cox_regressor else cox_regressor
            
            self.only_significant = only_significant
            self.hierae_kwargs = hierae_kwargs
            
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
            scaled_X = {k:self.scalers[k].fit_transform(selected_X[k]) for k in self.scalers}

            X = stack_dicts(scaled_X)
            blocks = stack_dicts_indexes(scaled_X)
            #y_str = events.astype(str) + "|" + durations.astype(str)
            y_str = np.char.add(
                np.char.add(
                    events.astype(str),
                    "|"
                ),
                durations.astype(str)
            )

            self.net = HierarchicalSAENet(
                module=HierarchicalSAE,
                criterion=hierarchical_sae_criterion,
                module__blocks=blocks,
                **self.hierae_kwargs
            )
            self.net.fit(
                X.astype(np.float32),
                y_str.astype(str),
            )
            hazard, original_common, decoded, original_blocks, blocks_decoded, block_hazards, z = self.net.infer(X.astype(np.float32))
            z = z.detach().cpu().numpy()
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
                    i for i in range(self.net.module_.common_embedding_dimension)
                ]
            if self.cox_regressor is not None:
                self.cox_regressor.fit(z[:,self.significant_factors], durations, events)
            self.clusterer.fit_predict(z[:,self.significant_factors])
            self.fitted = True
    
        def __preprocess_input_for_hierae(self, X: dict[str,np.ndarray], durations:np.ndarray=None, events:np.ndarray=None) -> np.ndarray:
            selected_X = {k:self.input_feature_selectors[k].fit_transform(X[k], durations, events) for k in self.input_feature_selectors}
            scaled_X = {k:self.scalers[k].fit_transform(selected_X[k]) for k in self.scalers}
            X = stack_dicts(scaled_X)
            return X

        def integrate(self, X:Union[np.ndarray,dict[str,np.ndarray]], durations:np.ndarray=None, events:np.ndarray=None, *args, **kwargs) -> np.ndarray:
            self.check_fitted()
            X = preprocess_input_to_dict(X)
            X = self.__preprocess_input_for_hierae(X, durations, events)
            return self.__integrate(X)[:,self.significant_factors]

        def hazard(self,X:Union[np.ndarray,dict[str,np.ndarray]], durations:np.ndarray=None, events:np.ndarray=None, *args, **kwargs) -> np.ndarray:
            if self.cox_regressor is not None:
                significant_factors = self.integrate(X, durations=durations, events=events, *args, **kwargs)
                return self.cox_regressor.hazard(significant_factors)
            self.check_fitted()
            X = preprocess_input_to_dict(X)
            X = self.__preprocess_input_for_hierae(X, durations, events)
            hazard, original_common, decoded, original_blocks, blocks_decoded, block_hazards, z = self.net.infer(X.astype(np.float32))
            return hazard.detach().cpu().numpy()
        
        def cluster(self, X:Union[np.ndarray,dict[str,np.ndarray]], durations:np.ndarray=None, events:np.ndarray=None, *args, **kwargs) -> np.ndarray:
            significant_factors = self.integrate(X, durations=durations, events=events, *args, **kwargs)
            return self.clusterer.predict(significant_factors)

        def __integrate(self,X:np.ndarray):
            hazard, original_common, decoded, original_blocks, blocks_decoded, block_hazards, z = self.net.infer(X.astype(np.float32))
            return z.detach().cpu().numpy()
        
        def check_fitted(self):
            if not self.fitted:
                raise NotFittedError("This {name} instance is not fitted yet. Call 'fit' with appropriate arguments before using this estimator.".format(name=type(self).__name__))
except Exception as e:
    raise e
except ImportError as e:
    warnings.warn("Failed to import the HierAE library! HierAE baseline model not built! {}".format(e))
except Exception as e:
    warnings.warn("Failed to build the HierAE baseline model! {}".format(e))


    