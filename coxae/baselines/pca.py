import numpy as np
import pandas as pd

from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.exceptions import NotFittedError

import lifelines

from ..base import SurvivalClustererMixin, HazardPredictorMixin
from ..significant_factor_selection import get_significant_factors, get_most_significant_factor_combinations

class PCAClustering(SurvivalClustererMixin,HazardPredictorMixin):

    def __init__(self,
            *args,
            dimensionality_reducer = PCA(n_components=128),
            only_significant:bool = True,
            limit_significant:int = -1,
            get_most_significant_combination_time_limit:float = 0.0,
            clusterer = KMeans(2),
            **kwargs):
        super().__init__(*args, **kwargs)
        
        self.only_significant = only_significant
        self.limit_significant = limit_significant
        self.get_most_significant_combination_time_limit = get_most_significant_combination_time_limit

        if not hasattr(dimensionality_reducer, "fit") or not hasattr(dimensionality_reducer, "transform"):
            raise ValueError('Only dimensionality reduction methods with separate "fit" and "transform" methods should be used by this class')
        self.dimensionality_reducer = dimensionality_reducer

        if not hasattr(clusterer, "fit") or not hasattr(clusterer, "predict"):
            raise ValueError('Only clusterers with separate "fit" and "predict" methods should be used by this class')
        self.clusterer = clusterer

        self.fitted = False

    # Mixin functions

    def fit(self, X: np.array, durations: np.array, events: np.array, *args, ae_opt_kwargs=None, ae_train_opts=None, **kwargs):
        integrated_values = self.dimensionality_reducer.fit_transform(X)
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
            self.significant_indexes = [i for i in range(integrated_values.shape[1])] if len(self.significant_indexes) == 0 else self.significant_indexes
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

    def __integrate(self,X):
        return self.dimensionality_reducer.transform(X)

    def integrate(self,X):
        self.check_fitted()
        return self.__integrate(X)

    def __calculate_hazard(self,X):
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