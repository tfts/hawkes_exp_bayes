'''
Illustration of our Bayesian approach working on synthetic data to:
1. diagnose misestimation 
    - cf. "# inference with exponential loglik and gamma prior" in this file
    - cf. Section 4.2 "Synthetic Data" in the paper
2. address stationarity breaks
    - cf. "# inference with exponential likelihood and bi-exponential prior" in this file 
    - cf. Section 4.3 "Synthetic Data" in the paper
'''
import os

import hyperopt
import lmfit
import numpy as np
import pandas as pd
import pymc3 as pm3

from sklearn.metrics import mean_squared_error
from tick.hawkes import HawkesExpKern, SimuHawkesExpKernels, SimuHawkesMulti

import constants

bayes_rounds = 100
user_hyp = 1 
print('user_hyp', user_hyp) 
beta_given = .8
print('real_beta', beta_given)
prior_std = lik_std = 1
prior_alpha = 1
const = constants.Constants(n_dims=1)
intensity_parameters = {
    'mu': [1.2],
    'alpha': [[.6]],
    'beta': [[beta_given]]
}
GRID_TO_SEARCH = np.logspace(-1, 2, num=10) 
REPETITIONS = 100
POSTERIOR_SAMPLE_NR = 10
DO_EXP_GAMMA = True
DO_BI_ADV = True
BI_ADV_DEBUG = True
BETA_INCREMENT = 1

# fct to optimize
def __loglik_wrapper(params, event_times_dict_list):
    hawkes_exp = HawkesExpKern([[params['beta']] * const.n_dims] * const.n_dims)
    score = hawkes_exp.score(events=event_times_dict_list, 
                             baseline=intensity_parameters['mu'], 
                             adjacency=intensity_parameters['alpha'] / np.array(intensity_parameters['beta']))
    return -score

def __parameter_rmse(actual_values, fitted_values):
    return np.sqrt(mean_squared_error(actual_values, fitted_values))

def __grid(data):
    learner_scores = []
    for beta_in_grid in GRID_TO_SEARCH:
        learner = HawkesExpKern([[beta_in_grid] * const.n_dims] * const.n_dims)
        learner.fit(data)
        learner_scores.append(-learner.score())
    return GRID_TO_SEARCH[np.argmin(learner_scores)]

def __lbgfsb(data):
    parameters = lmfit.Parameters()
    parameters.add('beta', min=0, value=const.initial_beta_value)
    minimizer = lmfit.Minimizer(__loglik_wrapper, parameters, 
                                fcn_args=(data, ))
    result = minimizer.minimize(method='lbfgsb')
    return result.params['beta'].value

def __hyperopt(data):
    param_space = {"beta": hyperopt.hp.lognormal("beta", const.initial_beta_value, const.initial_beta_value * const.initial_variance)}
    result = hyperopt.fmin(
        fn=lambda many_params: __loglik_wrapper(many_params, data),
        space=param_space,
        algo=hyperopt.tpe.suggest,
        max_evals=const.max_hyperopt_evals,
        verbose=False)
    return result["beta"]

FITTING_METHODS = {'Fitted_Beta': __lbgfsb, 'Grid_Beta': __grid, 'Hyperopt': __hyperopt}


# inference with exponential loglik and gamma prior
if DO_EXP_GAMMA:
    df = {'Mode': [], 'RMSE': [], 'Difference': []}
    for i in range(REPETITIONS):
        print('round', i)
        hawkes_exp_simu = SimuHawkesExpKernels(adjacency=list(intensity_parameters['alpha'] / np.array(intensity_parameters['beta'])), 
                                       decays=intensity_parameters['beta'], 
                                       baseline=intensity_parameters['mu'], 
                                       max_jumps=const.max_jumps, 
                                       verbose=False)
        multi = SimuHawkesMulti(hawkes_exp_simu, n_simulations=bayes_rounds, n_threads=4)
        multi.simulate()
        
        fitted_betas = {i: [] for i in FITTING_METHODS}
        predictive_density_means = {i: [] for i in FITTING_METHODS}
        post_alpha = {i: prior_alpha for i in FITTING_METHODS}
        post_beta = {i: user_hyp for i in FITTING_METHODS}
        for realization_i in range(1, bayes_rounds):
            for method_name, method_fct in FITTING_METHODS.items():
                fitted_betas[method_name].append(method_fct(multi.timestamps[:realization_i]))
        
        for method_name in FITTING_METHODS:
            post_alpha[method_name] = post_alpha[method_name] + len(fitted_betas[method_name])
            post_beta[method_name] = post_beta[method_name] + sum(fitted_betas[method_name])
            predictive_density_mean = post_beta[method_name] / (post_alpha[method_name] - 1) 
            predictive_density_means[method_name].append(predictive_density_mean)

            df['Mode'].append(method_name)
            df['RMSE'].append(__parameter_rmse(intensity_parameters['beta'], [predictive_density_means[method_name]]))
            df['Difference'].append(user_hyp - predictive_density_mean)
    
    pd.DataFrame(df).to_csv('misestimation_stationarity_ExpGamma.csv', index=False)
    os.system('R -f misestimation_stationarity.R --slave --args ExpGamma Difference')


# inference with exponential likelihood and bi-exponential prior
if DO_BI_ADV:
    SPLIT_INDEX = int(bayes_rounds / 2)
    df = {'Mode': [], 'Accuracy': [], 'RMSE': []}
    for i in range(REPETITIONS):
        print('\nround', i)
        hawkes_exp_simu = SimuHawkesExpKernels(adjacency=list(intensity_parameters['alpha'] / np.array(intensity_parameters['beta'])), 
                                       decays=intensity_parameters['beta'], 
                                       baseline=intensity_parameters['mu'], 
                                       max_jumps=const.max_jumps, 
                                       verbose=False)
        multi = SimuHawkesMulti(hawkes_exp_simu, n_simulations=bayes_rounds, n_threads=8)
        multi.simulate()
        hawkes_exp_simu = SimuHawkesExpKernels(adjacency=list(intensity_parameters['alpha'] / np.array(intensity_parameters['beta'])), 
                                            decays=(np.array(intensity_parameters['beta']) + BETA_INCREMENT).tolist(), 
                                            baseline=intensity_parameters['mu'], 
                                            max_jumps=const.max_jumps, 
                                            verbose=False)
        multi_other_beta = SimuHawkesMulti(hawkes_exp_simu, n_simulations=bayes_rounds, n_threads=8)
        multi_other_beta.simulate()
        combined_timestamps = multi.timestamps[SPLIT_INDEX:] + multi_other_beta.timestamps[SPLIT_INDEX:]
        
        fitted_betas = {i: [] for i in FITTING_METHODS}
        for realization_i in range(1, bayes_rounds):
            for method_name, method_fct in FITTING_METHODS.items():
                fitted_betas[method_name].append(method_fct(combined_timestamps[:realization_i]))
        
        for method_name in FITTING_METHODS:
            beta_1_result = None
            beta_2_result = None
            tau_result = None
            
            with pm3.Model() as model:
                beta_1 = pm3.Exponential('beta_1', user_hyp)
                beta_2 = pm3.Exponential('beta_2', user_hyp - .3)
                tau = pm3.DiscreteUniform('tau', lower=0, upper=len(fitted_betas[method_name]) - 1)
                idx = np.arange(len(fitted_betas[method_name]))
                lambda_ = pm3.math.switch(tau > idx, beta_1, beta_2)
                
                observation = pm3.Exponential('obs', lambda_, observed=fitted_betas[method_name])
                
                trace = pm3.sample(10000, tune=5000, cores=6) 

            beta_1_result = 1 / trace['beta_1'].mean() 
            beta_2_result = 1 / trace['beta_2'].mean()
            tau_result = np.median(trace['tau'])
            
            df['Mode'].append(method_name)
            df['RMSE'].append(__parameter_rmse(np.array(intensity_parameters['beta']).flatten().tolist() + (np.array(intensity_parameters['beta']) + BETA_INCREMENT).flatten().tolist() + [SPLIT_INDEX / bayes_rounds], [beta_1_result, beta_2_result, tau_result / bayes_rounds]))
            df['Accuracy'].append(1 if beta_1_result < beta_2_result else 0)
            if BI_ADV_DEBUG:
                print('#'*5, method_name, '#'*5)
                print('fitted_betas 1/2', np.mean(fitted_betas[method_name][:SPLIT_INDEX]), '2/2', np.mean(fitted_betas[method_name][SPLIT_INDEX:]))
                print("mcmc'd beta_1", beta_1_result)
                print("mcmc'd beta_2", beta_2_result)
                print("mcmc'd tau {}, histogram: {}".format(tau_result, list(zip(*np.histogram(trace['tau'])))))
                print()

    pd.DataFrame(df).to_csv('misestimation_stationarity_BiExpAdv.csv', index=False)
    os.system('R -f misestimation_stationarity.R --slave --args BiExpAdv Accuracy')
