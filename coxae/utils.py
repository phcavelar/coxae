import signal
import contextlib
import math

import numpy as np
import scipy.stats
import pandas as pd
import lifelines

class TimeoutException(Exception): pass

@contextlib.contextmanager
def time_limited_execution(seconds):
    if seconds>=0:
        def signal_handler(signum, frame):
            raise TimeoutException("Timed out!")
        signal.signal(signal.SIGALRM, signal_handler)
        signal.alarm(seconds)
    try:
        yield
    finally:
        if seconds>=0:
            signal.alarm(0)

def get_kmfs(subgroups, durations, events):
    """
    Gets the clinical dataset and subgroups (already aligned) and returns a
    Kaplan Meier estimator and its respective samples for each of these groups
    """
    target_features = pd.DataFrame({
        "duration": durations,
        "observed": events,
        "subgroup": subgroups,
    })
    kmfs = []
    grouped_samples = []
    for i in range(1+max(subgroups)):
        samples = target_features[target_features["subgroup"]==i]
        kmf = lifelines.KaplanMeierFitter(label=i)
        kmf.fit(durations=samples["duration"],
                event_observed=samples["observed"])
        grouped_samples.append(samples)
        kmfs.append(kmf)
    return kmfs, grouped_samples

def variance_score(X, y=None):
    return np.var(X, axis=0)

def mad_score(X, y=None):
    return scipy.stats.median_absolute_deviation(X, axis=0)

def calculate_concrete_alpha_decay(start_temp, min_temp, num_epochs, steps_per_epoch=1):
    return math.exp(math.log(min_temp / start_temp) / (num_epochs * steps_per_epoch))