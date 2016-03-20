functions {
  vector sqrt(vector v){
    vector[num_elements(v)] rv;
    for (i in 1:num_elements(v)){
      rv[i] <- sqrt(v[i]);
    }
    return rv;
  }
}

data {
  int<lower=1> n_trial;
  int<lower=1> n_unit;
  int<lower=1> n_bin;

  corr_matrix[n_unit] sc_corr_mat;
  vector[n_unit] sc_mean_vec;
  vector<lower=0>[n_unit] sc_var_vec;

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
  matrix[n_unit,n_unit] sc_shifted_corr_vals;
  sc_shifted_corr_vals <- (sc_corr_mat + 1) ./ 2;
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
  real base_rate;
  real threshold;
  real exponent;
  
  matrix[n_unit,n_unit] mp_cov;

  matrix[n_unit,n_unit] mp_cov_sqrt;
  vector[n_unit] sigma_points[n_unit * 2 + 1];
  vector[n_unit] tr_sigma_points[n_unit * 2 + 1];

  vector[n_unit] rate_mean;
  matrix[n_unit,n_unit] rate_cov;

  vector[n_unit] sc_mean_true;
  vector[n_unit] sc_var_true;
  matrix[n_unit,n_unit] sc_corr_true;

  real sc_mean_sampling_std;
  real sc_var_sampling_shape;
  real sc_var_sampling_rate;
  real sc_corr_sampling_alpha;
  real sc_corr_sampling_beta;

  real tr_corr_mean;
  real tr_corr_var;

  # quick and dirty rate parameters
  base_rate <- 10;
  threshold <- 1.9;
  exponent <- 1.1;

  # sample MP mean, variance, correlation
  mp_mean_vec ~ multi_normal(rep_vector(mp_mean_prior_mean,n_unit), diag_matrix(rep_vector(sqrt(mp_mean_prior_var),n_unit)));
  mp_var_vec ~ inv_gamma(mp_var_prior_shape, mp_var_prior_scale);
  mp_corr_chol ~ lkj_corr_cholesky(mp_corr_prior_conc);

  # rate parameter priors
  # base_rate ~ normal(base_rate_prior_mean, base_rate_prior_var);
  # threshold ~ normal(threshold_prior_mean, threshold_prior_var);
  # exponent ~ normal(exponent_prior_mean, exponent_prior_var);
  
  # construct MP covariance
  mp_cov <- quad_form_diag(multiply_lower_tri_self_transpose(mp_corr_chol), sqrt(mp_var_vec));

  # # construct sigma points
  # mp_cov_sqrt <- quad_form_sym(diag_matrix(sqrt(eigenvalues_sym(mp_cov))), eigenvectors_sym(mp_cov)');
  # sigma_points[1] <- mp_mean_vec;
  # for (i in 1:2*n_unit) {
  #   if (i <= n_unit) 
  #     sigma_points[i+1] <- sqrt(2) * row(mp_cov_sqrt,i)';
  #   else
  #     sigma_points[i+1] <- -sqrt(2) * row(mp_cov_sqrt,i-n_unit)';
  # }

  # # push the sigma points through the nonlinearity
  # for (i in 1:2*n_unit+1) {
  #   tr_sigma_points[i] <- (sigma_points[i] - threshold);
  #   for (j in 1:n_unit){
  #     tr_sigma_points[i,j] <- if_else(tr_sigma_points[i,j] > 0, base_rate * tr_sigma_points[i,j]^exponent, 0);
  #   }
  # }

  # # calculate rate moments and multiply with number of bins
  # rate_mean <- tr_sigma_points[1];
  # for (i in 2:2*n_unit+1) {
  #   rate_mean <- rate_mean + tr_sigma_points[i];
  # }
  # rate_mean <- rate_mean ./ (2*n_unit+1);

  # rate_cov <- diag_matrix(rep_vector(0.0,n_unit));
  # for (i in 2:2*n_unit+1) {
  #     rate_cov <- rate_cov + (tr_sigma_points[i] - rate_mean) * (tr_sigma_points[i] - rate_mean)';
  # }
  # rate_cov <- rate_cov ./ (2*n_unit+1); # covariance is normalised by N here

  # substitute for testing 
  rate_mean <- mp_mean_vec;
  rate_cov <- mp_cov;

  # calculate mean, variance and correlation of samples
  sc_mean_true <- rate_mean;
  sc_var_true <- diagonal(rate_cov);
  sc_corr_true <- rate_cov ./ prod(sqrt(sc_var_true));

  # sample the observations from the statistics sampling distributions
  for (i in 1:n_unit){
    sc_mean_sampling_std <- sqrt(sc_var_true[i]) / sqrt(n_trial);
    sc_var_sampling_shape <- (n_trial - 1) / 2.0;
    sc_var_sampling_rate <- n_trial / (2 * sqrt(sc_var_true[i]));
    sc_mean_vec[i] ~ normal(sc_mean_true[i], sc_mean_sampling_std);
    sc_var_vec[i] ~ gamma(sc_var_sampling_shape, sc_var_sampling_rate);
    for (j in i+1:n_unit){
      tr_corr_mean <- ( sc_corr_true[i,j] - ((1 - sc_corr_true[i,j]^2) / (2*n_trial - 2)) + 1) / 2;
      tr_corr_var <- ( (1 - sc_corr_true[i,j]^2)^2 * (1 + (11 * sc_corr_true[i,j]^2) / (2*n_trial - 2)) / (n_trial - 1) ) / 4;
      sc_corr_sampling_alpha <- (((1 - tr_corr_mean) / tr_corr_var) - (1 / tr_corr_mean)) * tr_corr_mean^2;
      sc_corr_sampling_beta <- sc_corr_sampling_alpha * ((1 / tr_corr_mean) - 1);
      sc_shifted_corr_vals[i, j] ~ beta(sc_corr_sampling_alpha, sc_corr_sampling_beta);
    }
  }

}

generated quantities{
  # get back MP correlations from Cholesky factors
  matrix[n_unit,n_unit] mp_corr_mat;
  mp_corr_mat <- multiply_lower_tri_self_transpose(mp_corr_chol);
}