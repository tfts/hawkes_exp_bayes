library(dplyr)
library(ggpubr)
library(gtools)
library(tidyr)

METRICS <- list(fitted_beta='Mean Fitted\nBeta', spectral_radius='Spectral\nRadius', bayes_beta='Bayesian Decay\nRatio', lm_beta='Fitted Decay\nRatio')

raw_data <- read.csv('real_world_vocabulary.csv') %>% mutate(Metric=recode(Metric, !!!METRICS)) %>% filter(grepl('Ratio', Metric))

bayes_ratio <- (raw_data %>% filter(grepl('Bayesian', Metric)))$Value
fitted_ratio <- (raw_data %>% filter(grepl('Fitted', Metric)))$Value
bayes_ratio_bootstrap <- rdirichlet(100000, rep(1, length(bayes_ratio))) %*% bayes_ratio
fitted_ratio_bootstrap <- rdirichlet(100000, rep(1, length(fitted_ratio))) %*% fitted_ratio
df <- data.frame(Metric=c('Bayesian Decay\nRatio', 'Fitted Decay\nRatio'), Value=c(mean(bayes_ratio_bootstrap), mean(fitted_ratio_bootstrap)), Upper=c(quantile(bayes_ratio_bootstrap, .975)[[1]], quantile(fitted_ratio_bootstrap, .975)[[1]]), Lower=c(quantile(bayes_ratio_bootstrap, .025)[[1]], quantile(fitted_ratio_bootstrap, .025)[[1]]))

df_plot <- ggplot(df, aes(Metric, Value)) + 
    geom_pointrange(aes(ymin=Lower, ymax=Upper)) + 
    ylab('Estimated Difficulty Ratio') + 
    theme_pubr() + 
    font('title', size = 11) + font('subtitle', size = 11) + font('caption', size = 11) + font('xlab', size = 12) + font('ylab', size = 12) + font('xy.text', size = 12) + font('legend.title', size = 12) + font('legend.text', size = 12)
ggsave(plot=df_plot, filename=paste0('real_world_vocabulary_ratio.png'), width=4, height=2.5)
