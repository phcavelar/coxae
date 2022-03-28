import numpy as np
import pandas as pd
import lifelines

from .base import HazardRegressorMixin

class CoxPHRegression(HazardRegressorMixin):
    """Scikit-learn style wrapper for lifelines' CoxPHFitter class"""

    def __init__(self, **kwargs):
        self.cox_ph_model = lifelines.CoxPHFitter(**kwargs)

    def fit(self, X, durations, events):
        self.cox_ph_model.fit(
            self.build_dataframe(X,durations,events),
            "durations",
            "events"
        )
        return self
    
    def hazard(self, X, durations=None, events=None):
        return np.exp(
            self.cox_ph_model.predict_log_partial_hazard(
                self.build_dataframe(X)
            )
        )
    
    def build_dataframe(self, X, durations=None, events=None):
        dfdict = {
            **{i: X[:, i] for i in range(X.shape[1])},
        }
        if durations is not None and events is not None:
            dfdict["durations"] = durations
            dfdict["events"] = events
        elif durations is not None or events is not None:
            raise ValueError("Either both durations and events are None or both should be defined")
        return pd.DataFrame(
            dfdict
        )