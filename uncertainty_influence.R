library(dplyr)
library(ggpubr)
library(tidyr)

raw_data <- read.csv('uncertainty_influence.csv') %>% mutate(Alpha = round(Alpha, 2))
ggerrorplot(raw_data, x='Alpha', y='Accuracy', desc_stat='mean_ci', xlab=expression(alpha['12'] ~ textstyle('as a % of') ~ alpha['21'])) + font('title', size = 10) + font('subtitle', size = 12) + font('caption', size = 12) + font('xlab', size = 13) + font('ylab', size = 13) + font('xy.text', size = 13) + font('legend.title', size = 13) + font('legend.text', size = 13) + theme(axis.text.x = element_text(angle = 90, hjust=1, vjust=.5))
ggsave(filename=paste0('uncertainty_influence.png'), width=4, height=2.5)
