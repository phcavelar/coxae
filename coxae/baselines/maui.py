import numpy as np
import pandas as pd

from sklearn.cluster import KMeans
from sklearn.exceptions import NotFittedError

import lifelines

from ..base import SurvivalClustererMixin, HazardPredictorMixin
from ..significant_factor_selection import get_significant_factors

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

    class MauiClustering(SurvivalClustererMixin,HazardPredictorMixin):

        def __init__(self,
                *args,
                maui_kwargs:dict = _DEFAULT_MAUI_KWARGS,
                maui_omics_layer_mappings:dict =_DEFAULT_MAUI_MAPPINGS,
                clusterer = KMeans(2),
                significance_alpha = 0.05,
                only_significant:bool = True,
                **kwargs):
            super().__init__(*args, **kwargs)
            
            self.only_significant = only_significant
            self.maui_kwargs = maui_kwargs
            self.maui_omics_layer_mappings = maui_omics_layer_mappings
            self.__init_maui(**maui_kwargs)
            
            if not hasattr(clusterer, "fit") or not hasattr(clusterer, "predict"):
                raise ValueError('Only clusterers with separate "fit" and "predict" methods should be used by this class')
            self.clusterer = clusterer

            self.significance_alpha = significance_alpha

            self.fitted = False

        # Mixin functions

        def fit(self, X: np.array, durations: np.array, events: np.array, *args, **kwargs):
            z = self.maui_model.fit_transform(self.preprocess_inputs(X)).values
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
            self.cox_ph_model = lifelines.CoxPHFitter().fit(
                pd.DataFrame(
                    {
                        **{i: z[:, i] for i in self.significant_factors},
                        "durations": durations,
                        "events": events,
                    }
                ),
                "durations",
                "events"
            )
            self.clusterer.fit_predict(z[:,self.significant_factors])
            self.fitted = True
        
        def predict(self, X: np.array, *args, **kwargs):
            significant_factors = self.integrate(X)[:,self.significant_factors]
            clusters = self.clusterer.predict(significant_factors)
            return clusters

        def __init_maui(self, **maui_kwargs):
            self.maui_model = maui.Maui(**maui_kwargs)

        def preprocess_inputs(self,X):
            return {k:self.maui_omics_layer_mappings[k](X) for k in self.maui_omics_layer_mappings}

        def __integrate(self,X):
            return self.maui_model.transform(self.preprocess_inputs(X)).values

        def integrate(self,X):
            self.check_fitted()
            return self.__integrate(X)

        def __calculate_hazard(self,X):
            z = self.__integrate(X)
            return np.exp(
                    self.cox_ph_model.predict_log_partial_hazard(
                    pd.DataFrame(
                        {
                            **{i: z[:, i] for i in self.significant_factors},
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
    
except ImportError as e:
    warnings.warn("Failed to import the maui library! Maui baseline model not built! {}".format(e))
except Exception as e:
    warnings.warn("Failed to build the maui baseline model! {}".format(e))