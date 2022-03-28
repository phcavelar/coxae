import numpy as np
from sklearn.feature_selection import SelectKBest, SelectPercentile

from .base import SurvivalClustererMixin, HazardRegressorMixin, SurvivalSelectorMixin
from .significant_factor_selection import get_significant_factors, get_most_significant_factor_combinations

class SurvivalSelectPercentile(SurvivalSelectorMixin):
    def __init__(self, score, percentile=10):
        self.selector = SelectPercentile(score, percentile=percentile)
    
    def fit(self, X, durations, events):
        self.selector.fit(X, X[:,0])
        return self
    
    def transform(self, X, durations=None, events=None):
        return self.selector.transform(X)
    
    def inverse_transform(self, X, durations=None, events=None):
        return self.selector.inverse_transform(X)

class SurvivalSelectKBest(SurvivalSelectorMixin):
    def __init__(self, score, k=1000):
        self.selector = SelectKBest(score, k=k)
    
    def fit(self, X, durations, events):
        self.selector.k = self.selector.k if self.selector.k < X.shape[1] else X.shape[1]
        self.selector.fit(X, X[:,0])
        return self
    
    def transform(self, X, durations=None, events=None):
        return self.selector.transform(X)
    
    def inverse_transform(self, X, durations=None, events=None):
        return self.selector.inverse_transform(X)

class DummyFeatureSelector(SurvivalSelectorMixin):
    def __init__(self):
        pass
    
    def fit(self, X, durations, events):
        return self
    
    def transform(self, X, durations=None, events=None):
        return X
    
    def inverse_transform(self, X, durations=None, events=None):
        return X

class CoxPHFeatureSelector(SurvivalSelectorMixin):

    def __init__(self, limit_significant:int=-1, get_most_significant_combination_time_limit:float=-1., fallback_strategy="best_k"):
        assert fallback_strategy in {"best_k", "fail"}
        self.fallback_strategy = fallback_strategy
        self.limit_significant = limit_significant
        self.get_most_significant_combination_time_limit = get_most_significant_combination_time_limit

    def fit(self, X, durations, events):
        significant_indexes, significant_indexes_p_values = get_significant_factors(X, durations, events)
        ordering = np.argsort(significant_indexes_p_values).tolist()
        significant_indexes = [significant_indexes[i] for i in ordering[:self.limit_significant]]
        significant_indexes_p_values = [significant_indexes_p_values[i] for i in ordering[:self.limit_significant]]
        if self.get_most_significant_combination_time_limit > 0:
            significant_indexes_combinations, significant_indexes_combinations_p_values = get_most_significant_factor_combinations(X, durations, events, significant_indexes, time_limit=self.self.get_most_significant_combination_time_limit)
            combination_ordering = np.argsort(significant_indexes_combinations_p_values).tolist()
            best_combination = significant_indexes_combinations[combination_ordering[0]]
            significant_indexes_p_values = [significant_indexes_p_values[i] for i in significant_indexes if i in best_combination]
            significant_indexes = best_combination
        # Add a guard in case no factors are found to be significant
        if significant_indexes == []:
            # Set a significance_threshold above 1 to select all factors as significant and get their p_values
            significant_indexes, significant_indexes_p_values = get_significant_factors(X, durations, events, significance_threshold=1.1)
            if self.fallback_strategy == "best_k":
                ordering = np.argsort(significant_indexes_p_values).tolist()
                significant_indexes = [significant_indexes[i] for i in ordering[:self.limit_significant]]
                significant_indexes_p_values = [significant_indexes_p_values[i] for i in ordering[:self.limit_significant]]
            else:
                raise ValueError("Fit method failed to find significant indexes and fallback strategy \"{}\" failed.".format(self.fallback_strategy))
        self.significant_indexes = significant_indexes
        self.significant_indexes_p_values = significant_indexes_p_values
        return self
    
    def transform(self, X, durations=None, events=None):
        return X[:, self.significant_indexes]
    
    def inverse_transform(self, X, durations=None, events=None):
        new_X = np.zeros_like(X)
        new_X[:, self.significant_indexes] = X[:, self.significant_indexes]
        return new_X