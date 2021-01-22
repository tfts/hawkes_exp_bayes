library(dplyr)
library(ggpubr)
library(tidyr)

lower_lim <- commandArgs(trailingOnly=T)[1]
upper_lim <- commandArgs(trailingOnly=T)[2]

raw_data <- read.csv(paste0('real_world_effervescence_', lower_lim, '-', upper_lim, '.csv'))
filtered_data <- raw_data %>% filter(grepl('Beta', Parameter))
filtered_data <- rbind(filtered_data %>% filter(Parameter == 'Beta 1') %>% filter(Value < quantile(Value, probs=.95)[[1]]), filtered_data %>% filter(Parameter == 'Beta 2') %>% filter(Value < quantile(Value, probs=.95)[[1]]))
beta_densities <- ggdensity(filtered_data, x='Value', linetype='Parameter', ylab='Density') + scale_linetype_manual(values=c('solid', 'dashed'), labels=c('Before', 'After')) + labs(linetype = 'Posterior\nwrt. Shock') + font('title', size = 10) + font('subtitle', size = 12) + font('caption', size = 12) + font('xlab', size = 13) + font('ylab', size = 13) + font('xy.text', size = 13) + font('legend.title', size = 13) + font('legend.text', size = 13)
ggsave(filename=paste0('real_world_effervescence_', lower_lim, '-', upper_lim, '.png'), plot=beta_densities, width=4, height=2.5)
