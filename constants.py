'''
Constants
'''
class Constants():    
    def __init__(self, n_dims=2, simulation_end=1000):
        self.bootstrap_repetitions = 100
        self.gam_repetitions = 2
        self.initial_beta_value = 1.
        self.initial_variance = 1.
        self.initial_other_value = .1
        self.max_beta_value = 100
        self.max_hyperopt_evals = 100
        self.n_cores = 4
        self.n_realizations = 100
        self.n_realizations_to_compare = 10
        self.n_dims = n_dims
        self.simulation_end = simulation_end
        self.max_jumps = 100
