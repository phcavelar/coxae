import signal
import contextlib

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