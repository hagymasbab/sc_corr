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

  # real base_rate_prior_mean;
  # real base_rate_prior_var;
  # real threshold_prior_mean;
  # real threshold_prior_var;
  # real exponent_prior_mean;
  # real exponent_prior_var;

  real mp_mean_prior_mean;
  real mp_mean_prior_var;
  real mp_var_prior_shape;
  real mp_var_prior_scale;
  real mp_corr_prior_conc;
}

transformed data {
  # correlations shifted to [0,1]

  # quick and dirty rate parameters
  real base_rate = 10;
  real threshold = 1.9;
  real exponent = 1.1;
}

parameters {
  cholesky_factor_corr[n_unit] mp_corr_chol;
  vector[n_unit] mp_mean_vec;
  vector<lower=0>[n_unit] mp_var_vec;

  # real<lower=0.1> base_rate;
  # real<lower=0> threshold;
  # real<lower=1,upper=3> exponent;
}

model {
  
  matrix[n_unit,n_unit] mp_cov;

  matrix[n_unit,n_sample] mp[n_bin];
  matrix[n_unit,n_sample] rate[n_bin];
  matrix[n_unit,n_sample] sc;

  vector[n_unit] sc_mean_true;
  vector[n_unit] sc_var_true;
  matrix[n_unit,n_unit] sc_corr_true;

  real sc_mean_sampling_std;
  real sc_var_sampling_shape;
  real sc_var_sampling_rate;
  real sc_corr_sampling_alpha;
  real sc_corr_sampling_beta;

  # sample MP mean, variance, correlation
  mp_mean_vec ~ multi_normal(mp_mean_prior_mean, sqrt(mp_mean_prior_var));
  mp_var_vec ~ inv_gamma(mp_var_prior_shape, mp_var_prior_scale);
  mp_corr_chol ~ lkj_corr_cholesky(mp_corr_prior_conc);

  # rate parameter priors
  # base_rate ~ normal(base_rate_prior_mean, base_rate_prior_var);
  # threshold ~ normal(threshold_prior_mean, threshold_prior_var);
  # exponent ~ normal(exponent_prior_mean, exponent_prior_var);
  
  # construct MP covariance
  mp_cov <- quad_form_diag(mp_corr_chol * mp_corr_chol', sqrt(mp_var_vec));

  # construct sigma points

  # push the sigma points through the nonlinearity

  # calculate rate moments and multiply with number of bins

  # calculate mean, variance and correlation of samples

  # sample the observations from the statistics sampling distributions

}

generated quantities{
  # get back MP correlations from Cholesky factors
}