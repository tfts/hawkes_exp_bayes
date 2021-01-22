'''
Illustration of our Bayesian approach working on real-world earthquake data to quantify uncertainty in inferred directions of temporal dependency between Hawkes process dimensions (in this case, earthquake regions).
    - cf. Section 4.1 "Earthquakes and Aftershocks" in the paper
'''
import os

import lmfit
import numpy as np
import pandas as pd

from scipy.stats import lomax
from tick.hawkes import HawkesExpKern, SimuHawkesExpKernels, SimuHawkesMulti

import constants

const = constants.Constants(n_dims=2)
SPLIT_SEQS = True
DATASET = 'ogata1982'
DATASET_SUFFIX = {'standard': '', 'restricted_1982': '_fig7'}['standard']
SPLIT_NR = 4
USER_HYP = 0.00633
CONVERT_TIMESCALE_TO_DECADE = True


# fct to optimize
def __loglik_wrapper(params, event_times, one_dim=False):
    hawkes_exp = HawkesExpKern([[params['beta']] * const.n_dims] * const.n_dims if not one_dim else [[params['beta']]])
    hawkes_exp.fit(event_times)
    score = hawkes_exp.score()
    return -score


# load dataset
dim_names = []
timestamps = []
with open('data/earthquakes/{0}{1}.txt'.format(DATASET, DATASET_SUFFIX), 'r') as f:
    for line in f:
        raw_data = line.strip().split(',')
        dim_names.append(raw_data[0])
        timestamps.append(np.genfromtxt(raw_data[1:]) / (365.25 * 10 if CONVERT_TIMESCALE_TO_DECADE else 1))

if SPLIT_SEQS:
    temp_timestamps = []
    for dim_i in range(len(dim_names)):
        temp_timestamps.append(np.array_split(timestamps[dim_i], SPLIT_NR))
    new_timestamps = []
    for split_i in range(SPLIT_NR):
        realization = []
        realization_begin = min([min(dim[split_i]) for dim in temp_timestamps])
        for dim_i, dim in enumerate(temp_timestamps):
            realization.append(dim[split_i] - realization_begin)
        new_timestamps.append(realization)
    timestamps = new_timestamps
else:
    timestamps = [timestamps]

# encode hypotheses: 
user_hyp = USER_HYP
prior_alpha = 1

# fit data
fitted_betas = []
for realization_i in range(1, len(timestamps) + 1 if not SPLIT_SEQS else len(timestamps)):
    parameters = lmfit.Parameters()
    parameters.add('beta', min=0, value=const.initial_beta_value)
    minimizer = lmfit.Minimizer(__loglik_wrapper, parameters, 
                                fcn_args=(timestamps[:realization_i], ))
    result = minimizer.minimize(method='lbfgsb')
    fitted_betas.append(result.params['beta'].value)

# print results
print('fitted_betas', fitted_betas)
post_alpha = prior_alpha + len(fitted_betas)
post_beta = user_hyp + sum(fitted_betas)
post_distr = lomax(c=post_alpha, scale=post_beta)
resulting_beta = post_distr.mean()
resulting_beta_95CI = post_distr.interval(alpha=.95)
print('dimensions:', list(enumerate(dim_names)))
print('user hyp: {}, posterior beta: {} with 95% CI at {}'.format(user_hyp, resulting_beta, resulting_beta_95CI))
learner = HawkesExpKern([[resulting_beta] * const.n_dims] * const.n_dims)
learner.fit(timestamps)
print('parameters:')
print('  mu:', learner.baseline.tolist())
print('  alphas:', (learner.adjacency * np.array(learner.decays)).tolist())
print('  betas:', learner.decays)
print()
for a_beta in np.linspace(*resulting_beta_95CI):
    print('#'*3, a_beta, '#'*3)
    learner = HawkesExpKern([[a_beta] * const.n_dims] * const.n_dims)
    learner.fit(timestamps)
    print('parameters:')
    print('  mu:', learner.baseline.tolist())
    print('  alphas:', (learner.adjacency * np.array(learner.decays)).tolist())
    print('  betas:', learner.decays)
    print('Wrong direction of influence captured!\n' if learner.adjacency[0][1] >= learner.adjacency[1][0] else '\n')
