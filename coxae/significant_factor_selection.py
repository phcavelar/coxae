import itertools

import numpy as np
import pandas as pd

import lifelines

from .utils import TimeoutException, time_limited_execution

def get_significant_factors(integrated_values, duration, observed, significance_threshold = 0.05):

    # logRank_Pvalues is a vector that will store the log-rank p-values of each 
    # univariate Cox-PH models built
    logRank_Pvalues = list()
    # significant_factors is a vector that will store every factor for which a 
    # significant Cox-PH model was built (e.g. log-rank P < 0.05)
    significant_factor_indexes = list()
    # Iterate over all factors/features of the actual dataframe
    for i in range(integrated_values.shape[1]):
        # Create a df with 3 columns, one for factor, one for OS_MONTHS
        # and one for OS_STATUS
        factor_df = pd.DataFrame(
            {
                "duration": duration,
                "observed": observed,
                "Factor_{}".format(i): integrated_values[:,i]
            }
        )
        # Build univariate COX-PH models
        cox_ph = lifelines.fitters.coxph_fitter.CoxPHFitter().fit(factor_df, "duration", "observed")
        log_rank_p_value = cox_ph.summary["p"].mean()
        if (log_rank_p_value<significance_threshold):
            significant_factor_indexes.append(i)
            logRank_Pvalues.append(log_rank_p_value)

    return significant_factor_indexes, logRank_Pvalues

def get_most_significant_factor_combinations(integrated_values, duration, observed, significant_factor_indexes=None, logRank_Pvalues=None, significance_threshold = 0.05, time_limit=-1, allow_keyboard_interrupt=False):
    if (
            (significant_factor_indexes is not None and logRank_Pvalues is None)
            or
            (significant_factor_indexes is None and logRank_Pvalues is not None)
            ):
            raise ValueError(
                "If specifying either significant_factor_indexes or logRank_Pvalues, both must not be None, whereas {} was given as None".format(
                    "logRank_Pvalues" if logRank_Pvalues is None else "significant_factor_indexes"
                )
            )
    elif significant_factor_indexes is None and logRank_Pvalues is None:
        significant_factor_indexes, logRank_Pvalues = get_significant_factors(integrated_values, duration, observed, significance_threshold = significance_threshold)

    # logRank_Pvalues is a vector that will store the log-rank p-values of each 
    # univariate Cox-PH models built
    combinations_logRank_Pvalues = [p for p in logRank_Pvalues]
    # significant_factors is a vector that will store every factor for which a 
    # significant Cox-PH model was built (log-rank P < 0.05)
    combinations_significant_factor_indexes = [[i] for i in significant_factor_indexes]
    number_of_significant_factors = len(significant_factor_indexes)
    try:
        with time_limited_execution(time_limit):
            # Iterate over all factors/features of the actual dataframe
            for k in range(2, number_of_significant_factors+1):
                for factor_indexes in itertools.combinations(significant_factor_indexes, k):
                    # Create a df with 3 columns, one for factor, one for OS_MONTHS
                    # and one for OS_STATUS
                    factor_df = pd.DataFrame(
                        {
                            "duration": duration,
                            "observed": observed,
                            **{
                                "Factor_{}".format(i): integrated_values[:,i]
                                for i in factor_indexes
                            }

                        }
                    )
                    # Build univariate COX-PH models
                    cox_ph = lifelines.fitters.coxph_fitter.CoxPHFitter().fit(factor_df, "duration", "observed")
                    log_rank_p_value = cox_ph.summary["p"].mean()
                    if (log_rank_p_value<significance_threshold):
                        combinations_significant_factor_indexes.append(factor_indexes)
                        combinations_logRank_Pvalues.append(log_rank_p_value)
    except KeyboardInterrupt:
        if not allow_keyboard_interrupt:
            raise
    except TimeoutException:
        pass
    return combinations_significant_factor_indexes, combinations_logRank_Pvalues