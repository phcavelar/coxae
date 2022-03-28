from typing import Union
import copy
import warnings
from collections import defaultdict
import platform
import inspect
import re

from abc import ABCMeta, abstractmethod

import numpy as np
import lifelines

class HazardRegressorMixin:
    """"""

    __NOTIMPLEMENTED_MESSAGE = "HazardRegressorMixin is a Mix-in class and should implement the hazard method. See the documentation on the HazardRegressorMixin class for more information."

    def hazard(self,X:Union[np.array,dict[str,np.array]], durations:np.array=None, events:np.array=None, *args, **kwargs) -> np.array:
        raise NotImplementedError(HazardRegressorMixin.__NOTIMPLEMENTED_MESSAGE)

    def concordance_index(self, hazards:np.array, durations:np.array, events:np.array, weightings=None) -> float:
        """Calculates the concordance index on pre-calculated hazard values.
        See https://lifelines.readthedocs.io/en/latest/lifelines.utils.html?highlight=concordance_index#lifelines.utils.concordance_index for more information.
        """
        return lifelines.utils.concordance_index(durations, -hazards, events)


class SurvivalClustererMixin:
    """A Mixin class for survival-based clustering"""

    __NOTIMPLEMENTED_MESSAGE = "SurvivalClustererMixin is a Mix-in class and should implement the fit, fit_cluster and cluster methods. See the documentation on the SurvivalClustererMixin class for more information."

    def fit(self, X:Union[np.array,dict[str,np.array]], durations:np.array=None, events:np.array=None, *args, **kwargs):
        """Fits a model to cluster values on input features X, using information about survival on durations and events to inform model training.

        Arguments:
            X {np.array} -- pre-processed input features
            durations {np.array} -- durations array
            events {np.arrray} -- events array 0/1
        """
        raise NotImplementedError(SurvivalClustererMixin.__NOTIMPLEMENTED_MESSAGE)

    def cluster(self, X:Union[np.array,dict[str,np.array]], durations:np.array=None, events:np.array=None, *args, **kwargs):
        """Predicts cluster labels.

        Arguments:
            X {np.array} -- pre-processed input features
            durations {np.array} -- durations array, unused
            events {np.arrray} -- events array 0/1, unused
        """
        raise NotImplementedError(SurvivalClustererMixin.__NOTIMPLEMENTED_MESSAGE)

    def fit_cluster(self, X:Union[np.array,dict[str,np.array]], durations:np.array=None, events:np.array=None, *args, **kwargs):
        """Perform clustering and returns cluster labels.

        Arguments:
            X {np.array} -- pre-processed input features
            durations {np.array} -- durations array, unused
            events {np.arrray} -- events array 0/1, unused
        """
        self.fit(X, durations, events, *args, **kwargs)
        return self.cluster(X)

    def logrank_p_score(self, clusters:np.array, durations:np.array, events:np.array, t_0:float=-1, weightings:str=None):
        """Performs a log-rank test based on pre-calculated subgroups.
        See https://lifelines.readthedocs.io/en/latest/lifelines.statistics.html?highlight=statistics#lifelines.statistics.logrank_test and https://lifelines.readthedocs.io/en/latest/lifelines.statistics.html?highlight=statistics#lifelines.statistics.multivariate_logrank_test for more information.
        """
        test_results = lifelines.statistics.multivariate_logrank_test(
            durations,
            clusters,
            events,
            t_0,
            weightings
        )
        return test_results.test_statistic, test_results.p_value


class SurvivalTransformerMixin:
    """A Mixin class for survival-based transformers"""

    __NOTIMPLEMENTED_MESSAGE = "SurvivalTransformerMixin is a Mix-in class and should implement the fit, fit_transform and transform methods. See the documentation on the SurvivalTransformerMixin class for more information."

    def fit(self, X:Union[np.array,dict[str,np.array]], durations:np.array=None, events:np.array=None, *args, **kwargs):
        """Fits a model to cluster values on input features X, using information about survival on durations and events to inform model training.

        Arguments:
            X {np.array} -- pre-processed input features
            durations {np.array} -- durations array
            events {np.arrray} -- events array 0/1
        """
        raise NotImplementedError(SurvivalTransformerMixin.__NOTIMPLEMENTED_MESSAGE)

    def transform(self, X:Union[np.array,dict[str,np.array]], durations:np.array=None, events:np.array=None, *args, **kwargs):
        """ a model to cluster values on input features X, using information about survival on durations and events to inform model clustering.

        Arguments:
            X {np.array} -- pre-processed input features
            durations {np.array} -- durations array, unused
            events {np.arrray} -- events array 0/1, unused
        """
        raise NotImplementedError(SurvivalTransformerMixin.__NOTIMPLEMENTED_MESSAGE)

    def fit_transform(self, X:Union[np.array,dict[str,np.array]], durations:np.array=None, events:np.array=None, **fit_params):
        """
        Fit to data, then transform it.

        Arguments:
            X {np.array} -- pre-processed input features
            durations {np.array} -- durations array, unused
            events {np.arrray} -- events array 0/1, unused
        """
        self.fit(X, durations, events, **fit_params)
        return self.transform(X)


class SurvivalSelectorMixin(SurvivalTransformerMixin):
    """A Mixin class for survival-based feature selectors"""

    __NOTIMPLEMENTED_MESSAGE = "SurvivalSelectorMixin is a Mix-in class and should implement the ivnerse_transform method as well as those methods required by SurvivalTransformerMixin. See the documentation on the SurvivalSelectorMixin class for more information."

    def inverse_transform(self, X):
        """
        Reverses the transform operation.

        Arguments:
            X {np.array} -- pre-processed input features
            durations {np.array} -- durations array, unused
            events {np.arrray} -- events array 0/1, unused
        """
        raise NotImplementedError(SurvivalSelectorMixin.__NOTIMPLEMENTED_MESSAGE)