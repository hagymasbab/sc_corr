data {
  int<lower=1> n_trial;
  real<lower=-1,upper=1> sc_corr;
  vector[2] sc_mean_vec;
  int<lower=1> n_bins;
}

parameters {
  real<lower=-1,upper=1> mp_corr;
  real<lower=0.1> base_rate;
  real<lower=0> threshold;
  real<lower=1,upper=3> exponent;
}

model {
  vector[2] mp_mean_vec;
  vector[2] mp_var_vec;
  matrix[2,2] mp_cov_mat;
  matrix[2,n_trial] rate_sums;
  matrix[2,n_trial] spike_counts;
  vector[2] act_mp;
  vector[2] act_rate;
  vector[2] sc_sd_vec;
  real corr_value;

  # sample MP correlation and rate parameter priors
  base_rate ~ normal(10,2);
  threshold ~ normal(1.9,0.1);
  exponent ~ normal(1.1,0.1);
  mp_corr ~ uniform(-1,1);

  # construct MP mean and covariance (only instantaneous)
  for (l in 1:2){
    mp_mean_vec[l] <- 1.0;
    mp_var_vec[l] <- 1.0;
    for (m in 1:2){
      # TODO do this for non-unit variance
      mp_cov_mat[l,m] <- if_else (l==m, mp_var_vec[l], mp_corr);
    }
  }

  # calculate spike counts for each trial
  for (i in 1:n_trial){
    for (j in 1:n_bins){
      act_mp ~ multi_normal(mp_mean_vec,mp_cov_mat);
      for (k in 1:2){
        act_rate[k] <- if_else (act_mp[k] > threshold, base_rate * (act_mp[k] - threshold)^exponent, 0);
        rate_sums[i,k] <- rate_sums[i,k] + act_rate[k];
      }
    }
    for (k in 1:2){
      spike_counts[i,k] <- floor(rate_sums[i,k]);
    }
  }

  # calculate SC correlation and mean
  for (k in 1:2){
    sc_mean_vec[k] ~ normal(mean(spike_counts[k,:]),1);
    sc_sd_vec[k] <- sd(spike_counts[k,:]);
  }
  corr_value <- 0.0;
  for (i in 1:n_trial){
    corr_value <- corr_value + (spike_counts[1,i] - sc_mean_vec[1]) * (spike_counts[1,i] - sc_mean_vec[1]) /(sc_sd_vec[1]*sc_sd_vec[2]);
  }
  sc_corr ~ normal(corr_value,1);
}