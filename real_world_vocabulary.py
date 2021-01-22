'''
Illustration of our Bayesian approach working on real-world vocabulary learning data (from the Duolingo language learning app) to diagnose misestimation when quantifying language learning progress.
    - cf. Section 4.2 "Vocabulary Learning Intensity" and Fig. 5 in the paper

Note that the pre-processing scripts are included in the same folder where the source data lies.
'''
import os

import lmfit
import numpy as np
import pandas as pd

from tick.hawkes import HawkesExpKern, SimuHawkesExpKernels

import constants

const = constants.Constants(n_dims=1)

USER_HYPS = {'c': 1, 'a': 2}
EVENTS_PER_USER = 10 # 10, 11, 12
PRINT_RESULTS = True
REPETITIONS = 100
METRIC_LIST = [j + i for j in ['a', 'c'] for i in ['_fitted_beta', '_spectral_radius', '_bayes_beta']] # the 1st and 2nd metrics will be 1 or 0 according to whether the hypothesized relational ordering of difficulty is correct, and the 3rd is the bayesian fitted ratio (for comparison with the encoded 1:2 hypothesis)

# fct to optimize
def __loglik_wrapper(params, event_times, one_dim=False):
    hawkes_exp = HawkesExpKern([[params['beta']] * const.n_dims] * const.n_dims if not one_dim else [[params['beta']]])
    hawkes_exp.fit(event_times)
    score = hawkes_exp.score()
    return -score if (not np.isnan(score)) and score != 0 else 10**6


df = pd.read_csv('data/vocabulary/learning_traces_en_a1a2c1c2.csv', header=None, names=['cerf', 'word', 'p_recall', 'timestamp', 'delta', 'user_id', 'learning_language', 'ui_language', 'lexeme_id', 'lexeme_string', 'history_seen', 'history_correct', 'session_seen', 'session_correct'])
df_c = df[(df['cerf'] == 'C1') | (df['cerf'] == 'C2')]
df_a = df[(df['cerf'] == 'A1') | (df['cerf'] == 'A2')]
df_result = {'Metric': [], 'Value': []}
for repetition_i in range(REPETITIONS):
    print('repetition', repetition_i)
    c_user_event_counts = df_c.groupby('user_id').size()
    c_users_filtered = set(c_user_event_counts[c_user_event_counts == EVENTS_PER_USER].index)
    c_users_realizations = []
    for c_user in c_users_filtered:
        c_users_realizations.append(df_c[df_c['user_id'] == c_user]['timestamp'])
    c_users_realizations = [[(i.to_numpy() - min(i)).astype(np.float)] for i in c_users_realizations]
    a_user_event_counts = df_a.groupby('user_id').size()
    a_users_filtered = set(np.random.choice(a_user_event_counts[a_user_event_counts == EVENTS_PER_USER].index, len(c_users_realizations), replace=False))
    a_users_realizations = []
    for a_user in a_users_filtered:
        a_users_realizations.append(df_a[df_a['user_id'] == a_user]['timestamp'])
    a_users_realizations = [[(i.to_numpy() - min(i)).astype(np.float)] for i in a_users_realizations]
    realizations = {'a': a_users_realizations, 'c': c_users_realizations}
    repetition_result = {i: None for i in METRIC_LIST}
    for dataset, timestamps in realizations.items():
        # encode hypotheses: 
        user_hyp = USER_HYPS[dataset]
        prior_alpha = 1

        # fit data
        fitted_betas = []
        for realization_i in range(1, len(timestamps)):
            parameters = lmfit.Parameters()
            parameters.add('beta', min=0, value=const.initial_beta_value)
            minimizer = lmfit.Minimizer(__loglik_wrapper, parameters, 
                                        fcn_args=(timestamps[:realization_i], ))
            result = minimizer.minimize(method='lbfgsb')
            fitted_betas.append(result.params['beta'].value)

        # print results
        post_alpha = prior_alpha + len(fitted_betas)
        post_beta = user_hyp + sum(fitted_betas)
        resulting_beta = post_beta / (post_alpha - 1)
        learner = HawkesExpKern(resulting_beta)
        learner.fit(timestamps)
        hawkes_exp_kernels = SimuHawkesExpKernels(adjacency=learner.adjacency, decays=learner.decays, baseline=learner.baseline)
        spectral_radius = hawkes_exp_kernels.spectral_radius()
        repetition_result[dataset + '_fitted_beta'] = np.mean(fitted_betas)
        repetition_result[dataset + '_spectral_radius'] = spectral_radius
        repetition_result[dataset + '_bayes_beta'] = resulting_beta
        if PRINT_RESULTS:
            print('{0}  {1}  {0}'.format('#' * 5, dataset))
            print('fitted_betas', fitted_betas)
            print('mean fitted_betas', np.mean(fitted_betas))
            print('user hyp: {}, posterior beta: {}'.format(user_hyp, resulting_beta))
            print('parameters:')
            print('  mu:', learner.baseline.tolist())
            print('  alphas:', (learner.adjacency * np.array(learner.decays)).tolist())
            print('  betas:', learner.decays)
            print('spectral_radius: ', spectral_radius)
            print()
    df_result['Metric'].append('fitted_beta')
    df_result['Value'].append(1 if repetition_result['a_fitted_beta'] > repetition_result['c_fitted_beta'] else 0)
    df_result['Metric'].append('spectral_radius')
    df_result['Value'].append(1 if repetition_result['a_spectral_radius'] < repetition_result['c_spectral_radius'] else 0)
    df_result['Metric'].append('bayes_beta')
    df_result['Value'].append(repetition_result['c_bayes_beta'] / repetition_result['a_bayes_beta'])
    df_result['Metric'].append('lm_beta')
    df_result['Value'].append(repetition_result['c_fitted_beta'] / repetition_result['a_fitted_beta'])

pd.DataFrame(df_result).to_csv('real_world_vocabulary.csv', index=False)
os.system('R -f real_world_vocabulary.R --slave')
