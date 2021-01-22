library(dplyr)
library(ggpubr)
library(tidyr)

CORRECT_BETA <- as.numeric(commandArgs(trailingOnly=T)[1])
NORMAL_SAMPLES <- 1000000
raw_data <- read.csv(paste0('uncertainty_basics_beta', CORRECT_BETA, '.csv'))

produce_plot <- function(df, annotation, include_norm01, filename_suffix) {
    if (include_norm01) {
        the_plot <- ggplot(df) + geom_density(aes(Norm), na.rm=T, color='#999999') + geom_density(aes(SplitBeta), na.rm=T, color='#F0E442') + geom_density(aes(Beta), na.rm=T)
        resulting_filename <- paste0('uncertainty_basics_beta', CORRECT_BETA, '_NORM', filename_suffix, '.png')
    } else {
        the_plot <- ggplot(df) + geom_density(aes(Beta), na.rm=T)
        resulting_filename <- paste0('uncertainty_basics_beta', CORRECT_BETA, '_NONORM', filename_suffix, '.png')
    }
    the_plot <- the_plot +
        xlab('Decay') +
        ylab('Density') + 
        theme_pubr() +
        font('title', size = 9) + font('subtitle', size = 11) + font('caption', size = 11) + font('xlab', size = 10) + font('ylab', size = 10) + font('xy.text', size = 10) + font('legend.title', size = 10) + font('legend.text', size = 10)
    ggsave(filename=resulting_filename, plot=the_plot, width=4, height=2.5)
}

mean_beta <- mean(raw_data$Beta)
sd_beta <- sd(raw_data$Beta)
raw_data <- raw_data %>% mutate(Beta=(Beta - mean_beta) / sd_beta)
mean_splitbeta <- mean(raw_data$SplitBeta)
sd_splitbeta <- sd(raw_data$SplitBeta)
raw_data <- raw_data %>% mutate(SplitBeta=(SplitBeta - mean_splitbeta) / sd_splitbeta)
raw_data$Norm <- rep(NA, NROW(raw_data))
raw_data <- rbind(raw_data, data.frame(Beta=rep(NA, NORMAL_SAMPLES), SplitBeta=rep(NA, NORMAL_SAMPLES), Norm=rnorm(NORMAL_SAMPLES)))
produce_plot(raw_data, (CORRECT_BETA - mean_beta) / sd_beta, T, '')
