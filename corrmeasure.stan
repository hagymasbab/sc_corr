functions {
  vector sqrt_vec(vector v){
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
  int<lower=1> n_obs;

  corr_matrix[n_unit] sc_corr_mat[n_obs];
  vector[n_unit] sc_mean_vec[n_obs];
  vector<lower=0>[n_unit] sc_var_vec[n_obs];

  real base_rate_prior_mean;
  # real base_rate_prior_var;
  real threshold_prior_mean;
  # real threshold_prior_var;
  real exponent_prior_mean;
  # real exponent_prior_var;

  real mp_mean_prior_mean;
  real mp_mean_prior_var;
  real mp_var_prior_shape;
  real mp_var_prior_scale;
  real mp_corr_prior_conc;

  # MC
  int n_samples;
  vector[n_unit] stdnorm_samples[n_samples];
}

transformed data {
  # int n_samples;
  matrix[n_unit,n_unit] sc_shifted_corr_vals[n_obs];

  # correlations shifted to [0,1]
  for (i in 1:n_obs) {
    sc_shifted_corr_vals[i] <- (sc_corr_mat[i] + 1) ./ 2.0;
  }

}

parameters {
  cholesky_factor_corr[n_unit] mp_corr_chol;
  vector[n_unit] mp_mean_vec;
  vector<lower=0.01>[n_unit] mp_var_vec;
}

model {
  real base_rate;
  real threshold;
  real exponent;
  
  matrix[n_unit,n_unit] mp_cov;
  matrix[n_unit,n_unit] mp_cov_cholesky;

  matrix[n_unit,n_unit] mp_cov_sqrt;
  vector[n_unit] sigma_points[n_samples];
  vector[n_unit] tr_sigma_points[n_samples];

  vector[n_unit] rate_mean;
  matrix[n_unit,n_unit] rate_cov;

  vector[n_unit] summed_rate_mean;
  matrix[n_unit,n_unit] summed_rate_cov;

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

  # sample MP mean, variance, correlation
  mp_mean_vec ~ multi_normal(rep_vector(mp_mean_prior_mean,n_unit), diag_matrix(rep_vector(sqrt(mp_mean_prior_var),n_unit)));
  mp_var_vec ~ inv_gamma(mp_var_prior_shape, mp_var_prior_scale);
  mp_corr_chol ~ lkj_corr_cholesky(mp_corr_prior_conc);

  # quick and dirty rate parameters
  base_rate <- base_rate_prior_mean;
  threshold <- threshold_prior_mean;
  exponent <- exponent_prior_mean;
  
  # construct MP covariance
  mp_cov <- quad_form_diag(multiply_lower_tri_self_transpose(mp_corr_chol), sqrt_vec(mp_var_vec));

  # construct sigma points
  mp_cov_cholesky <- cholesky_decompose(mp_cov);
  for (i in 1:n_samples) {
    sigma_points[i] <- mp_mean_vec + mp_cov_cholesky * stdnorm_samples[i];  
  }

  # push the sigma points through the nonlinearity
  for (i in 1:n_samples) {
    tr_sigma_points[i] <- (sigma_points[i] - threshold);
    for (j in 1:n_unit){
      tr_sigma_points[i,j] <- if_else(tr_sigma_points[i,j] > 0, tr_sigma_points[i,j], 0);
      tr_sigma_points[i,j] <- tr_sigma_points[i,j]^exponent;
      tr_sigma_points[i,j] <- base_rate * tr_sigma_points[i,j];
    }
  }

  # calculate rate moments
  rate_mean <- tr_sigma_points[1];
  for (i in 2:n_samples) {
    rate_mean <- rate_mean + tr_sigma_points[i];
  }
  rate_mean <- rate_mean ./ n_samples;

  rate_cov <- diag_matrix(rep_vector(0.0,n_unit));
  for (i in 1:n_samples) {
      rate_cov <- rate_cov + (tr_sigma_points[i] - rate_mean) * (tr_sigma_points[i] - rate_mean)';
  }
  rate_cov <- rate_cov ./ (n_samples - 1.0); # in UT covariance shoud be normalised by N here

  # calculate summed rate moments by multiplying with number of bins
  summed_rate_mean <- n_bin * rate_mean;
  summed_rate_cov <- n_bin * rate_cov;

  # calculate mean, variance and correlation of samples
  sc_mean_true <- summed_rate_mean;
  sc_var_true <- diagonal(summed_rate_cov);
  sc_corr_true <- summed_rate_cov ./ prod(sqrt_vec(sc_var_true));

  # sample the observations from the statistics sampling distributions
  for (o in 1:n_obs){
    for (i in 1:n_unit){
      sc_mean_sampling_std <- sqrt(sc_var_true[i]) / sqrt(n_trial);
      sc_var_sampling_shape <- (n_trial - 1) / 2.0;
      sc_var_sampling_rate <- n_trial / (2.0 * sc_var_true[i]);
      sc_mean_vec[o, i] ~ normal(sc_mean_true[i], sc_mean_sampling_std);
      sc_var_vec[o, i] ~ gamma(sc_var_sampling_shape, sc_var_sampling_rate);
      for (j in i+1:n_unit){
        tr_corr_mean <- ( sc_corr_true[i,j] - ((1 - sc_corr_true[i,j]^2) / (2*n_trial - 2)) + 1) / 2.0;
        tr_corr_var <- ( (1 - sc_corr_true[i,j]^2)^2 * (1 + (11 * sc_corr_true[i,j]^2) / (2*n_trial - 2)) / (n_trial - 1) ) / 4.0;
        sc_corr_sampling_alpha <- (((1 - tr_corr_mean) / tr_corr_var) - (1 / tr_corr_mean)) * tr_corr_mean^2;
        sc_corr_sampling_beta <- sc_corr_sampling_alpha * ((1 / tr_corr_mean) - 1);
        sc_shifted_corr_vals[o, i, j] ~ beta(sc_corr_sampling_alpha, sc_corr_sampling_beta);
      }
    }
  }

}

generated quantities{
  # get back MP correlations from Cholesky factors
  matrix[n_unit,n_unit] mp_corr_mat;
  mp_corr_mat <- multiply_lower_tri_self_transpose(mp_corr_chol);
}