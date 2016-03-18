functions {
  
}

data {
  int<lower=1> n_trial;
  int<lower=1> n_unit;
  int<lower=1> n_bin;
  int<lower=1> n_sample;
  corr_matrix[n_unit] sc_corr_mat;
  vector[n_unit] sc_mean_vec;
  vector[n_unit]<lower=0> sc_var_vec;
}

parameters {
  cov_matrix[n_unit] mp_cov_mat;
  vector[n_unit] mp_mean_vec;

  real<lower=0.1> base_rate;
  real<lower=0> threshold;
  real<lower=1,upper=3> exponent;
}

model {
  
  matrix[n_unit,n_sample] mp[n_bin];
  matrix[n_unit,n_sample] rate[n_bin];
  matrix[n_unit,n_sample] sc;

  vector[n_unit] sc_mean_true;
  vector[n_unit] sc_var_true;
  matrix[n_unit,n_unit] sc_corr_true;

  # sample MP statistics priors


  # rate parameter priors
  base_rate ~ normal(10,2);
  threshold ~ normal(1.9,0.1);
  exponent ~ normal(1.1,0.1);
  

  # take n_unit x n_sample samples for each bin

  # push the samples through the nonlinearity

  # sum the rates in the small bins and add random initial value

  # discretise the rate into spike counts

  # calculate mean, variance and correlation of samples

  # sample the observations from the statistics sampling distributions

}