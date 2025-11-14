

#library(Rmosek)
#source("~/mosek/11.0/tools/platform/osxaarch64/rmosek/builder.R")
#attachbuilder()
#install.rmosek()


############################################
library(close)
df <- read.csv('data/OA_kfr_black_pooled_p25.csv')
head(df)

Ys <- df$kfr_black_pooled_p25
sigmas <- df$kfr_black_pooled_p25_se

close_results <- close::compute_close(Ys,sigmas)

posterior_means <- close_results$close_npmle

plot(sigmas, Ys)
plot(sigmas, posterior_means)

df['close_posterior_means'] = posterior_means
#write.csv(df, 'OA_close_npmle_data.csv')
