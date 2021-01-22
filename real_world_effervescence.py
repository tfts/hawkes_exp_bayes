'''
Illustration of our Bayesian approach working on real-world data on the manifestation of collective effervescence on Twitter to address stationarity breaks.
    - cf. Section 4.3 "Strength of Collective Effervescence" in the paper

Run this script with "python3 real_world_effervescence.py 20 25" to reproduce Fig. 6 of the paper. 
To experiment with different activity bounds, run first the pre-processing script "TweetTS_preprocessing.R" in the folder "data/effervescence" with e.g. "R -f TweetTS_preprocessing.R --args 25 30"
'''
import os
import subprocess
import sys

from collections import Counter

import lmfit
import numpy as np
import pandas as pd
import pymc3 as pm3

from tick.hawkes import HawkesExpKern, SimuHawkesExpKernels

import constants

USER_HYP = {'before': 1.5, 'after': 1} # {'before': 1, 'after': 1.5} posterior-contrarian, {'before': 1.5, 'after': 1} posterior-like
const = constants.Constants(n_dims=1)
SUBSAMPLE_SIZE = 0 # 0 or eg 10**9 corresponds to no subsample
BURNIN_REALIZATIONS = 1 # 1 corresponds to no burn-in
INPUT_FILENAME = 'data/effervescence/nov_8d.csv'

# count the number of lines (i.e. realizations) in a file
def __file_len(fname):
    with open(fname) as f:
        for i, _ in enumerate(f):
            pass
    return i + 1

# fct to optimize
def __loglik_wrapper(params, event_times, one_dim=False):
    hawkes_exp = HawkesExpKern([[params['beta']] * const.n_dims] * const.n_dims if not one_dim else [[params['beta']]])
    hawkes_exp.fit(event_times)
    score = hawkes_exp.score()
    return -score


realizations_total = __file_len(INPUT_FILENAME)
realizations = []
with open(INPUT_FILENAME) as f:
    for line_i, line in enumerate(f):
        tweet_dates = [pd.Timestamp(i).to_datetime64() for i in line.strip().split(',')]
        realization = ((np.array(tweet_dates) - (pd.Timestamp('2015-11-05 21:16:00') if line_i < (realizations_total / 2) else pd.Timestamp('2015-11-13 21:16:00')).to_datetime64()) / np.timedelta64(1, 'm')).tolist() # converting to days past first tweet date
        realizations.append([np.array(realization)])

if SUBSAMPLE_SIZE:
    realization_indexes = np.random.choice(range(int(len(realizations) / 2)), SUBSAMPLE_SIZE, replace=False)
    realizations = [realizations[i] for i in realization_indexes] + [realizations[i] for i in realization_indexes + int(len(realizations) / 2)]

fitted_betas = []
for realization_i in range(BURNIN_REALIZATIONS, len(realizations)):
    parameters = lmfit.Parameters()
    parameters.add('beta', min=0, value=const.initial_beta_value)
    minimizer = lmfit.Minimizer(__loglik_wrapper, parameters, 
                                fcn_args=(realizations[:realization_i], ))
    result = minimizer.minimize(method='lbfgsb')
    fitted_betas.append(result.params['beta'].value)

beta_1_result = None
beta_2_result = None
tau_result = None
with pm3.Model() as model:
    beta_1 = pm3.Exponential('beta_1', USER_HYP['before'])
    beta_2 = pm3.Exponential('beta_2', USER_HYP['after'])
    tau = pm3.DiscreteUniform('tau', lower=0, upper=len(fitted_betas) - 1)
    idx = np.arange(len(fitted_betas))
    lambda_ = pm3.math.switch(tau > idx, beta_1, beta_2)

    observation = pm3.Exponential('obs', lambda_, observed=fitted_betas)
    
    trace = pm3.sample(20000, tune=10000, cores=6)

beta_1_result = 1 / trace['beta_1'].mean()
beta_2_result = 1 / trace['beta_2'].mean() 
tau_result = np.median(trace['tau'])

result_str = '### range {}-{} ###\n'.format(sys.argv[1], sys.argv[2])
result_str += 'len(realizations) = {}\n'.format(len(realizations))
result_str += 'fitted_betas {}\n'.format(fitted_betas)
result_str += '    1st .5 mean is {}, 2nd is {}\n'.format(np.mean(fitted_betas[:int(len(fitted_betas) / 2)]), np.mean(fitted_betas[int(len(fitted_betas) / 2):]))
result_str += "mcmc'd beta_1 {}\n".format(beta_1_result)
result_str += "mcmc'd beta_2 {}\n".format(beta_2_result)
result_str += "mcmc'd tau {}, histogram: {}\n\n".format(tau_result, list(zip(*np.histogram(trace['tau']))))
with open('real_world_effervescence.txt', 'a') as f:
    f.write(result_str)

result_data = (1 / trace['beta_1']).tolist() + (1 / trace['beta_2']).tolist() + trace['tau'].tolist()
result_labels = ['Beta 1'] * len(trace['beta_1']) + ['Beta 2'] * len(trace['beta_2']) + ['Tau'] * len(trace['tau'])
pd.DataFrame({'Parameter': result_labels, 'Value': result_data}).to_csv('real_world_effervescence_{}-{}.csv'.format(sys.argv[1], sys.argv[2]), index=False)
os.system('R -f real_world_effervescence.R --slave --args {} {}'.format(sys.argv[1], sys.argv[2]))
