#0. David Garcia kindly provided the Tweet timestamps of the dataset underlying the paper "D. Garcia and B. Rimé, 'Collective emotions and social resilience in the digital traces after a terrorist attack,' Psychological Science, 2019.". The file "TweetTS.csv" contains those Tweet timestamps.

#1. prepare dataset of tweets before and after
library(lubridate)
TweetsTS <- read.csv('TweetTS.csv', stringsAsFactors=F)
TweetsTS$ts <- as_datetime(TweetsTS$ts)
write.csv(TweetsTS[TweetsTS$ts >= as_datetime('2015-11-01 21:16:00') & TweetsTS$ts < as_datetime('2015-11-13 21:16:00'), ], file='nov_before.csv')
write.csv(TweetsTS[TweetsTS$ts >= as_datetime('2015-11-13 21:16:00') & TweetsTS$ts < as_datetime('2015-11-25 21:16:00'), ], file='nov_after.csv')
write.csv(TweetsTS[TweetsTS$ts >= as_datetime('2015-11-05 21:16:00') & TweetsTS$ts < as_datetime('2015-11-13 21:16:00'), ], file='nov_before_8d.csv')
write.csv(TweetsTS[TweetsTS$ts >= as_datetime('2015-11-13 21:16:00') & TweetsTS$ts < as_datetime('2015-11-21 21:16:00'), ], file='nov_after_8d.csv')

#2. extract timestamps of users with minimum activity
library(dplyr)

MIN_ACTIVITY <- as.numeric(commandArgs(trailingOnly=T)[1]) # 20 
MAX_ACTIVITY <- as.numeric(commandArgs(trailingOnly=T)[2]) # 25
OUTPUT_FILE <- 'nov_8d.csv'

before <- read.csv('nov_before_8d.csv', stringsAsFactors=F)
after <- read.csv('nov_after_8d.csv', stringsAsFactors=F)
filtered_users_before <- (as.data.frame(table(before$uid)) %>% filter(Freq >= MIN_ACTIVITY & Freq < MAX_ACTIVITY))$Var1 
filtered_users_after <- (as.data.frame(table(after$uid)) %>% filter(Freq >= MIN_ACTIVITY & Freq < MAX_ACTIVITY))$Var1
common_users <- intersect(filtered_users_before, filtered_users_after)
str_to_write <- ''
for (common_user in common_users) {
    str_to_write <- paste0(str_to_write, paste(unlist(sort(t((before %>% filter(uid == common_user))$ts))), collapse=','), '\n')
}
for (common_user in common_users) {
    str_to_write <- paste0(str_to_write, paste(unlist(sort(t((after %>% filter(uid == common_user))$ts))), collapse=','), '\n')
}
cat(str_to_write, file=OUTPUT_FILE)
