library(dplyr)
library(gtools)

BAYES_TYPE <- commandArgs(trailingOnly=T)[1]
METRIC_TYPE <- commandArgs(trailingOnly=T)[2]
PATH_PREFIX <- paste0('misestimation_stationarity_', BAYES_TYPE)
BAYESIAN_INFERENCE_TASKS <- list(ExpGamma='"Misaligned Hypotheses"', BiExpAdv='"Stationarity Break"')
FITTING_METHODS <- list(Fitted_Beta='L-BFGS-B', Grid_Beta='Grid Search', Hyperopt='Hyperopt')

raw_data <- read.csv(paste0(PATH_PREFIX, '.csv'))
for (fitting_method in names(FITTING_METHODS)) {
    raw_data_metric <- (raw_data %>% filter(grepl(fitting_method, Mode)))[[METRIC_TYPE]] 
    bayes_bootstrap <- rdirichlet(100000, rep(1, length(raw_data_metric))) %*% raw_data_metric
    cat(paste0('For the Bayesian inference task ', BAYESIAN_INFERENCE_TASKS[[BAYES_TYPE]], ' with the fitting method ', FITTING_METHODS[[fitting_method]], ', the mean ', METRIC_TYPE, ' is ', mean(bayes_bootstrap), ' (standard deviation ', sd(bayes_bootstrap), ')\n'))
}
