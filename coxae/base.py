import numpy as np
import lifelines

class SurvivalClustererMixin:
    """A Mixin class for survival-based clustering"""

    __NOTIMPLEMENTED_MESSAGE = "SurvivalClustererMixin is a Mix-in class and should implement the fit, fit_transform and transform methods. See the documentation on the SurvivalClustererMixin class for more information."

    def fit(self, X:np.array, durations:np.array, events:np.array, *args, **kwargs):
        """Fits a model to cluster values on input features X, using information about survival on durations and events to inform model training.

        Arguments:
            X {np.array} -- pre-processed input features
            durations {np.array} -- durations array
            events {np.arrray} -- events array 0/1
        
        Returns:
            SurvivalClustererMixin -- The fitted model class.
        """
        raise NotImplementedError(SurvivalClustererMixin.__NOTIMPLEMENTED_MESSAGE)

    def predict(self, X:np.array, *args, **kwargs):
        """ a model to cluster values on input features X, using information about survival on durations and events to inform model clustering.

        Arguments:
            X {np.array} -- pre-processed input features
            durations {np.array} -- durations array
            events {np.arrray} -- events array 0/1
        
        Returns:
            SurvivalClustererMixin -- The fitted model class.
        """
        raise NotImplementedError(SurvivalClustererMixin.__NOTIMPLEMENTED_MESSAGE)

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

class HazardPredictorMixin:
    __NOTIMPLEMENTED_MESSAGE = "HazardPredictorMixin is a Mix-in class and should implement the calculate_hazard method. See the documentation on the HazardPredictorMixin class for more information."

    def calculate_hazard(self,X:np.array) -> np.array:
        raise NotImplementedError(HazardPredictorMixin.__NOTIMPLEMENTED_MESSAGE)

    def concordance_index(self, hazards:np.array, durations:np.array, events:np.array):
        """Calculates the concordance index on pre-calculated hazard values.
        See https://lifelines.readthedocs.io/en/latest/lifelines.utils.html?highlight=concordance_index#lifelines.utils.concordance_index for more information.
        """
        return lifelines.utils.concordance_index(durations, -hazards, events)
