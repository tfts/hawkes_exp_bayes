'''
Illustration of the non-convex, noisy shape of the Hawkes process log-likelihood function (cf. Fig. 3 of the paper)
'''
import os

import numpy as np
import pandas as pd

from tick.hawkes import SimuHawkesExpKernels


GIVEN_BETAS = [1.2] 
END_TIME = 1000
MAX_JUMPS = 100
REPETITIONS = 100
INTENSITY_PARAMETERS = {
    "mu": [0.1],
    "alpha": [[0.5]]
}
USE_OZAKI_LOGLIK = True
SMALL_STEP = .01
LARGE_STEP = .2
LARGEST_STEP = .4

def __ozaki_loglik(m, a, b, t_is):
    t_n = max(t_is)
    l = -m*t_n
    term_2 = 0
    for t_i in t_is:
        term_2 += np.exp(-b*(t_n - t_i)) - 1
    l = l + a/b * term_2
    term_3 = 0
    for t_i_index, t_i in enumerate(t_is):
        a_i = 0
        for t_j in t_is[:t_i_index]:
            a_i += np.exp(-b*(t_i - t_j))
        term_3 += np.log(m + a*a_i)
    return l + term_3

def __get_bootstrapped_ci(values, aggregator=np.mean):
    agg_value = aggregator(values)
    bootstrap_results = []
    for _ in range(REPETITIONS):
        agg_value_bootstrap = aggregator(np.random.choice(values, len(values)))
        bootstrap_results.append(agg_value_bootstrap - agg_value)
    upper_value = agg_value - np.percentile(bootstrap_results, 2.5)
    lower_value = agg_value - np.percentile(bootstrap_results, 97.5)
    return agg_value, upper_value, lower_value


for given_beta in GIVEN_BETAS:
    INTENSITY_PARAMETERS['beta'] = [[given_beta] * len(INTENSITY_PARAMETERS['mu'])] * len(INTENSITY_PARAMETERS['mu'])
    betas_to_plot = {
        'Broad': np.arange(given_beta - .3, given_beta + .3 + SMALL_STEP, SMALL_STEP), 
        'Far': np.arange(.6, 5. + LARGE_STEP, LARGE_STEP), 
        'Farthest': np.arange(.4, 20. + LARGEST_STEP, LARGEST_STEP), 
    }
    for plot_type, beta_list in betas_to_plot.items():
        df = {'Beta': [], 'LogLik': [], 'Upper': [], 'Lower': []}
        for beta in beta_list:
            # bootstrap
            scores = []
            for _ in range(REPETITIONS):
                # get events
                hawkes_exp_simu = SimuHawkesExpKernels(adjacency=list(INTENSITY_PARAMETERS["alpha"] / np.array(INTENSITY_PARAMETERS["beta"])), 
                                                        decays=INTENSITY_PARAMETERS["beta"], 
                                                        baseline=INTENSITY_PARAMETERS["mu"], 
                                                        max_jumps=MAX_JUMPS, 
                                                        verbose=False)
                hawkes_exp_simu.simulate()
                # get loglik
                scores.append(-__ozaki_loglik(INTENSITY_PARAMETERS["mu"][0], INTENSITY_PARAMETERS["alpha"][0][0], beta, hawkes_exp_simu.timestamps[0]))
            mean_score, upper_score, lower_score = __get_bootstrapped_ci(scores)
            df['Beta'].append(beta)
            df['LogLik'].append(mean_score)
            df['Upper'].append(upper_score)
            df['Lower'].append(lower_score)
        # plot
        pd.DataFrame(df).to_csv('loglik_plot_{}.csv'.format(plot_type), index=False)
        os.system('R -f loglik_plot.R --slave --args {} {} {}'.format(plot_type, 'mean_ci', given_beta))
        os.system('R -f loglik_plot.R --slave --args {} {} {}'.format(plot_type, 'mean', given_beta))
