'''
Illustration of the uncertainty surrounding point estimations of the decay value (with and without stationarity breaks) in a Hawkes process.
This code produces normalized decay distributions which deviate from the standard Gaussian, as exemplified in Fig. 1 in the paper.
'''
import functools
import os
import sys
import time

import hyperopt
import lmfit
import numpy as np
import pandas as pd

from scipy import stats
from tick.hawkes import HawkesExpKern, SimuHawkesExpKernels, SimuHawkesMulti

import constants
const = constants.Constants(n_dims=1)

SAMPLE_SIZE = 100
BETA_INCREMENT = 1
SPLIT_INDEX = int(const.n_realizations / 2)


# fct to optimize
def __loglik_wrapper(params, event_times_dict_list):
    learner = HawkesExpKern(decays=[[params["beta"]] * const.n_dims] * const.n_dims)
    learner.fit(event_times_dict_list)
    return -learner.score()

df = []
df_splitbeta = []
intensity_parameters = {
    "mu": [0.1],
    "alpha": [[0.5]],
    "beta": [[1.2]]
}
print('true beta is', intensity_parameters["beta"])
for i, initial_beta_i in enumerate(np.linspace(0.01, 10, num=SAMPLE_SIZE)):  
    print('round', i)
    
    # simulation
    hawkes_exp_kernels = SimuHawkesExpKernels(adjacency=list(intensity_parameters["alpha"] / np.array(intensity_parameters["beta"])), 
                                            decays=intensity_parameters["beta"], 
                                            baseline=intensity_parameters["mu"], 
                                            end_time=const.simulation_end,
                                            verbose=False)
    multi = SimuHawkesMulti(hawkes_exp_kernels, n_simulations=const.n_realizations, n_threads=8)
    multi.simulate()

    # LBFGSB
    parameters = lmfit.Parameters()
    parameters.add("beta", min=0, value=1.) 
    minimizer = lmfit.Minimizer(__loglik_wrapper, parameters, 
                                fcn_args=(multi.timestamps, ))
    start_time = time.time()
    result = minimizer.minimize(method="lbfgsb")
    end_time = time.time()
    df.append(result.params["beta"].value)
    
    # simulation but with split beta
    hawkes_exp_simu = SimuHawkesExpKernels(adjacency=list(intensity_parameters['alpha'] / np.array(intensity_parameters['beta'])), 
                            decays=intensity_parameters['beta'], 
                            baseline=intensity_parameters['mu'], 
                            max_jumps=const.max_jumps, 
                            verbose=False)
    multi = SimuHawkesMulti(hawkes_exp_simu, n_simulations=const.n_realizations, n_threads=8)
    multi.simulate()
    hawkes_exp_simu = SimuHawkesExpKernels(adjacency=list(intensity_parameters['alpha'] / np.array(intensity_parameters['beta'])), 
                                        decays=(np.array(intensity_parameters['beta']) + BETA_INCREMENT).tolist(), 
                                        baseline=intensity_parameters['mu'], 
                                        max_jumps=const.max_jumps, 
                                        verbose=False)
    multi_other_beta = SimuHawkesMulti(hawkes_exp_simu, n_simulations=const.n_realizations, n_threads=8)
    multi_other_beta.simulate()
    combined_timestamps = multi.timestamps[SPLIT_INDEX:] + multi_other_beta.timestamps[SPLIT_INDEX:]
    
    # LBFGSB
    parameters = lmfit.Parameters()
    parameters.add("beta", min=0, value=1.)
    minimizer = lmfit.Minimizer(__loglik_wrapper, parameters, 
                                fcn_args=(combined_timestamps, ))
    start_time = time.time()
    result = minimizer.minimize(method="lbfgsb")
    end_time = time.time()
    df_splitbeta.append(result.params["beta"].value)

rounded_beta = np.round(intensity_parameters["beta"][0][0], 4)
pd.DataFrame({"Beta": df, "SplitBeta": df_splitbeta}).to_csv('uncertainty_basics_beta{}.csv'.format(rounded_beta), index=False)
os.system('R -f uncertainty_basics.R --slave --args ' + str(rounded_beta))
