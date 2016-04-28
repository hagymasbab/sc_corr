import numpy as np
import numpy.random as rnd
import scipy.stats as st
import pickle
import csnltools as ct


class correlationMeasurementModel:

    def __init__(self, stanModel):
        self.sm = pickle.load(open(stanModel, 'rb'))
        self.threshold = 1.9
        self.base_rate = 10
        self.exponent = 1.1
        self.mp_mean_prior_mean = 1
        self.mp_mean_prior_var = 2
        self.mp_var_prior_shape = 3
        self.mp_var_prior_scale = 4
        self.mp_corr_prior_conc = 3

    def generate(self, mp_corr, mp_mean, mp_var, n_trial, n_bin, seed=None):
        if seed is not None:
            rnd.seed(seed)
        n_unit = len(mp_mean)
        C_mp = np.diag(mp_var)
        C_mp = np.sqrt(C_mp).dot(mp_corr.dot(np.sqrt(C_mp)))
        # TODO us this C_mp = ct.covariance_matrix(mp_var, mp_corr)
        U = rnd.multivariate_normal(mp_mean, C_mp, (n_trial, n_bin))
        rate = np.zeros((n_trial, n_bin, n_unit))
        for t in range(n_trial):
            for b in range(n_bin):
                for c in range(n_unit):
                    if U[t, b, c] > self.threshold:
                        rate[t, b, c] = self.base_rate * (U[t, b, c] - self.threshold) ** self.exponent
        spike_count = np.floor(np.sum(rate, axis=1))  # n_trial x n_unit
        sc_mean = np.mean(spike_count, axis=0)
        sc_var = np.var(spike_count, axis=0)
        sc_corr = np.corrcoef(spike_count.T)
        return sc_corr, sc_mean, sc_var

    def infer(self, sc_corr, sc_mean, sc_var, n_trial, n_bin, n_samp_est, n_iter, n_chains, seed=None, init='random', thin=1):
        n_unit = len(sc_mean)

        if hasattr(n_samp_est, "__iter__"):
            stdnorm_samples = n_samp_est
            n_samp_est = len(stdnorm_samples)
        else:
            stdnorm_samples = rnd.normal(size=(n_samp_est, n_unit))

        corr_dat = {
            'n_trial': n_trial,
            'n_bin': n_bin,
            'n_unit': n_unit,
            'sc_corr_mat': sc_corr,
            'sc_mean_vec': sc_mean,
            'sc_var_vec': sc_var,
            'mp_mean_prior_mean': self.mp_mean_prior_mean,
            'mp_mean_prior_var': self.mp_mean_prior_var,
            'mp_var_prior_shape': self.mp_var_prior_shape,
            'mp_var_prior_scale': self.mp_var_prior_scale,
            'mp_corr_prior_conc': self.mp_corr_prior_conc,
            'exponent_prior_mean': self.exponent,
            'base_rate_prior_mean': self.base_rate,
            'threshold_prior_mean': self.threshold,
            'n_samples': n_samp_est,
            'stdnorm_samples': stdnorm_samples
        }
        if seed is not None:
            fit = self.sm.sampling(data=corr_dat, iter=n_iter, chains=n_chains, init=init, thin=thin, seed=seed)
        else:
            fit = self.sm.sampling(data=corr_dat, iter=n_iter, chains=n_chains, init=init, thin=thin)
        estimation = fit.extract(permuted=True)
        return estimation['mp_corr_mat'], estimation['mp_mean_vec'], estimation['mp_var_vec']

    def sample_mean_prior(self, n_samp, n_unit):
        return rnd.normal(self.mp_mean_prior_mean, self.mp_mean_prior_var, (n_samp, n_unit)).T

    def sample_variance_prior(self, n_samp, n_unit):
        var_samples = st.invgamma.rvs(self.mp_var_prior_shape, scale=self.mp_var_prior_scale, size=n_samp * n_unit)
        return np.reshape(var_samples, (n_unit, n_samp))

    def evaluate_corr_prior(self, mpc):
        # TODO implement this from the Stan manual
        return 1

    def rate_transform(self, mp_samples):
        rate = np.zeros(mp_samples.shape)
        for u in range(mp_samples.shape[0]):
            for s in range(mp_samples.shape[1]):
                if mp_samples[u, s] > self.threshold:
                    thr_mp = mp_samples[u, s] - self.threshold
                else:
                    thr_mp = 0
                rate[u, s] = self.base_rate * thr_mp ** self.exponent
        return rate

    def evaluate_pairwise_corr_likelihood(self, n_trial, scc, mpc, mp_means, mp_vars, n_samp):
        mp_covmat = ct.covariance_matrix(mp_vars, np.array([[1, mpc], [mpc, 1]]))
        mp_samples = rnd.multivariate_normal(mp_means, mp_covmat, n_samp)
        rate_samples = self.rate_transform(mp_samples)
        # print rate_samples
        rate_corr = ct.correlation(rate_samples.T)
        # print rate_corr
        obs_corr_mean = rate_corr - ((1 - rate_corr) ** 2 / (2 * (n_trial - 1)))
        obs_corr_var = (1 + 11 * rate_corr ** 2 / (2 * (n_trial - 1))) * ((1 - rate_corr ** 2) ** 2) / (n_trial - 1)
        # print obs_corr_mean, obs_corr_var
        beta_shape, beta_rate = ct.beta_params_from_moments((obs_corr_mean + 1) / 2.0, obs_corr_var / 4.0)
        # print beta_shape, beta_rate
        if beta_shape > 100:
            return 0
        else:
            return st.beta.pdf((scc + 1) / 2.0, beta_shape, beta_rate)

    def pairwise_corr_numerical_posterior(self, n_trial, sc_corr, n_transform_samples, n_marginal_samples, n_eval_points):
        n_unit = 2
        eval_points = np.linspace(-1, 1, n_eval_points)
        posterior = np.zeros(n_eval_points)
        # sample from priors of mean and variance
        mean_samples = self.sample_mean_prior(n_marginal_samples, n_unit)
        var_samples = self.sample_variance_prior(n_marginal_samples, n_unit)
        for i in range(n_eval_points):
            act_mpc = eval_points[i]
            marginal_likelihood = 0.0
            for l in range(n_marginal_samples):
                act_marginal_likelihood = self.evaluate_pairwise_corr_likelihood(n_trial, sc_corr, act_mpc, mean_samples[:, l], var_samples[:, l], n_transform_samples)
                # print act_marginal_likelihood
                marginal_likelihood += act_marginal_likelihood
            act_prior = self.evaluate_corr_prior(act_mpc)
            posterior[i] = marginal_likelihood * act_prior
        # posterior = posterior / np.sum(posterior)
        return posterior
