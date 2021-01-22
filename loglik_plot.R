library(dplyr)
library(ggpubr)
library(tidyr)

PLOT_TYPE <- commandArgs(trailingOnly=T)[1]
AGGREGATION_TYPE <- commandArgs(trailingOnly=T)[2]
GIVEN_BETA <- commandArgs(trailingOnly=T)[3]
PATH_PREFIX <- paste0('loglik_plot_', PLOT_TYPE)

raw_data <- read.csv(paste0(PATH_PREFIX, '.csv'))

if (AGGREGATION_TYPE == 'mean_ci') {
    loglik_plot <- ggplot(raw_data) + geom_pointrange(aes(x=Beta, y=LogLik, ymin=Lower, ymax=Upper), fatten=.5)
} else {
    loglik_plot <- ggplot(raw_data) + geom_point(aes(x=Beta, y=LogLik))
}
loglik_plot <- loglik_plot +
    geom_line(aes(x=Beta, y=LogLik)) +
    geom_vline(xintercept=as.numeric(GIVEN_BETA), colour='#CC79A7', linetype='dashed') + 
    ylab('Negative Log Likelihood') +
    xlab(expression(beta)) +
    theme_pubr() +
    font('title', size = 10) + font('subtitle', size = 12) + font('caption', size = 12) + font('xlab', size = 13) + font('ylab', size = 13) + font('xy.text', size = 13) + font('legend.title', size = 13) + font('legend.text', size = 13)
ggsave(filename=paste0('loglik_plot_', GIVEN_BETA, '_', PLOT_TYPE, '_', AGGREGATION_TYPE, '.png'), plot=loglik_plot, width=4, height=2.5)
