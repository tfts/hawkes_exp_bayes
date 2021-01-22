'''
Illustration of our Bayesian approach working on synthetic data to quantify uncertainty in inferred directions of temporal dependency between Hawkes process dimensions.
- cf. Section 4.1 "Synthetic Data" and Fig. 4 in the paper
'''
import os
import sys
import time

import hyperopt
import lmfit
import numpy as np
import pandas as pd

from scipy.stats import lomax
from tick.hawkes import HawkesExpKern, SimuHawkesExpKernels, SimuHawkesMulti

import constants
const = constants.Constants(n_dims=2)


# fct to optimize
def __loglik_wrapper(params, event_times_dict_list):
    learner = HawkesExpKern(decays=[[params["beta"]] * const.n_dims] * const.n_dims)
    learner.fit(event_times_dict_list)
    return -learner.score()

intensity_parameters = {
    'mu': [0.1, 0.5],
    'alpha': [[0.1, None], [0.7, 0.2]],
    'beta': [[1.2] * 2] * 2
}
bayes_rounds = 100
prior_alpha = 1
user_hyp = 1.5

df = {'Alpha': [], 'Accuracy': []}
for alpha_perc in np.linspace(.75, 1.25, 10):
    print('alpha_perc', alpha_perc)
    intensity_parameters['alpha'][0][1] = intensity_parameters['alpha'][1][0] * alpha_perc
    ground_truth = intensity_parameters['alpha'][0][1] < intensity_parameters['alpha'][1][0]
    hawkes_exp_simu = SimuHawkesExpKernels(adjacency=list(intensity_parameters['alpha'] / np.array(intensity_parameters['beta'])), 
                                    decays=intensity_parameters['beta'], 
                                    baseline=intensity_parameters['mu'], 
                                    end_time=const.simulation_end, 
                                    verbose=False)
    multi = SimuHawkesMulti(hawkes_exp_simu, n_simulations=bayes_rounds, n_threads=4)
    multi.simulate()

    fitted_betas = []
    for realization_i in range(1, bayes_rounds):
        parameters = lmfit.Parameters()
        parameters.add('beta', min=0, value=const.initial_beta_value)
        minimizer = lmfit.Minimizer(__loglik_wrapper, parameters, 
                                    fcn_args=(multi.timestamps[:realization_i], ))
        result = minimizer.minimize(method='lbfgsb')
        fitted_betas.append(result.params['beta'].value)
    
    post_alpha = prior_alpha + len(fitted_betas)
    post_beta = user_hyp + sum(fitted_betas)
    post_distr = lomax(c=post_alpha, scale=post_beta)
    resulting_beta_95CI = post_distr.interval(alpha=.95)
    for a_beta in np.linspace(*resulting_beta_95CI, num=100):
        learner = HawkesExpKern([[a_beta] * const.n_dims] * const.n_dims)
        learner.fit(multi.timestamps)
        alpha_matrix = (learner.adjacency * np.array(learner.decays)).tolist()
        df['Alpha'].append(alpha_perc)
        df['Accuracy'].append(1 if (alpha_matrix[0][1] < alpha_matrix[1][0]) == ground_truth else 0)

pd.DataFrame(df).to_csv('uncertainty_influence.csv', index=False)
os.system('R -f uncertainty_influence.R --slave')
