import numpy as np
import numpy.random as rnd
import scipy.stats as st
import scipy.special as fn
import pickle
import csnltools as ct
import matplotlib.pyplot as pl


def corr_error_pdf(r, trueR, n):
    num_const = (n - 2) * fn.gamma(n - 1)
    num_rho = (1 - trueR ** 2) ** ((n - 1) / 2)
    num_r = (1 - r ** 2) ** ((n - 4) / 2)
    den_const = np.sqrt(2 * np.pi) * fn.gamma(n - 0.5)
    den_rhor = (1 - trueR * r) ** (n - 1.5)  # TODO prevent overflow here
    num_hyp = fn.hyp2f1(0.5, 0.5, (2 * n - 1) / 2, (trueR * r + 1) / 2)
    return (num_const * num_rho * num_r * num_hyp) / (den_const * den_rhor)


def var_error_pdf(v, trueV, n):
    return st.gamma.pdf(v, (n - 1) / 2, 2 * trueV / n)


def mean_error_pdf(m, trueM, trueV, n):
    if trueV == 0:
        return 0
    else:
        return st.norm.pdf(m, trueM, np.sqrt(trueV) / np.sqrt(n))


def corr_dist_moments(true_corr, n_trial):
    obs_corr_mean = true_corr - ((1 - true_corr) ** 2 / (2 * (n_trial - 1)))
    obs_corr_var = (1 + 11 * true_corr ** 2 / (2 * (n_trial - 1))) * ((1 - true_corr ** 2) ** 2) / (n_trial - 1)
    return obs_corr_mean, obs_corr_var


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

        self.quick_mean = 2
        self.quick_var = 1
        self.quick_corr = 0.5
        self.quick_trialnum = 100000
        self.quick_binnum = 1

    def generate(self, mp_corr, mp_mean, mp_var, n_trial, n_bin, n_obs=1, seed=None):
        if seed is not None:
            rnd.seed(seed)
        n_unit = len(mp_mean)
        C_mp = np.diag(mp_var)
        C_mp = np.sqrt(C_mp).dot(mp_corr.dot(np.sqrt(C_mp)))
        # TODO use this C_mp = ct.covariance_matrix(mp_var, mp_corr)
        U = rnd.multivariate_normal(mp_mean, C_mp, (n_obs, n_trial, n_bin))
        # TODO use this self.rate_transform(U)
        rate = np.zeros((n_obs, n_trial, n_bin, n_unit))
        for o in range(n_obs):
            for t in range(n_trial):
                for b in range(n_bin):
                    for c in range(n_unit):
                        if U[o, t, b, c] > self.threshold:
                            rate[o, t, b, c] = self.base_rate * ((U[o, t, b, c] - self.threshold) ** self.exponent)
        spike_count = np.floor(np.sum(rate, axis=2))  # n_obs x n_trial x n_unit
        sc_mean = np.mean(spike_count, axis=1)  # n_obs x n_unit
        sc_var = np.var(spike_count, axis=1)
        sc_corr = np.zeros((n_obs, n_unit, n_unit))
        for o in range(n_obs):
            sc_corr[o, :, :] = np.corrcoef(spike_count[o, :, :].T)
        return sc_corr, sc_mean, sc_var

    def quick_sample(self, n_obs):
        n_unit = 2
        mu_mp = self.quick_mean * np.ones(n_unit)
        var_mp = self.quick_var * np.ones(n_unit)
        corr_mp = np.array([[1, self.quick_corr], [self.quick_corr, 1]])
        sc_corr, sc_mean, sc_var = self.generate(corr_mp, mu_mp, var_mp, self.quick_trialnum, self.quick_binnum, n_obs=n_obs)
        return sc_corr, sc_mean, sc_var

    def infer(self, sc_corr, sc_mean, sc_var, n_trial, n_bin, n_samp_est, n_iter, n_chains=2, seed=None, init='random', thin=1):
        n_obs = sc_mean.shape[0]
        n_unit = sc_mean.shape[1]

        if hasattr(n_samp_est, "__iter__"):
            stdnorm_samples = n_samp_est
            n_samp_est = len(stdnorm_samples)
        else:
            stdnorm_samples = rnd.normal(size=(n_samp_est, n_unit))

        corr_dat = {
            'n_obs': n_obs,
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
        # TODO do this in matrix form
        # TODO support ndarrays of more than 2 dimensions
        rate = np.zeros(mp_samples.shape)
        for u in range(mp_samples.shape[0]):
            for s in range(mp_samples.shape[1]):
                if mp_samples[u, s] > self.threshold:
                    thr_mp = mp_samples[u, s] - self.threshold
                else:
                    thr_mp = 0
                rate[u, s] = self.base_rate * thr_mp ** self.exponent
        return rate

    def evaluate_pairwise_corr_likelihood(self, n_trial, scc, sc_means, sc_vars, mpc, mp_means, mp_vars, n_samp):
        # TODO handle multiple observations
        # TODO handle more than 1 bins
        n_unit = 2
        mp_covmat = ct.covariance_matrix(mp_vars, np.array([[1, mpc], [mpc, 1]]))
        mp_samples = rnd.multivariate_normal(mp_means, mp_covmat, n_samp)
        rate_samples = self.rate_transform(mp_samples)
        rate_corr = ct.correlation(rate_samples.T)
        rate_means = np.mean(rate_samples, axis=0)
        rate_vars = np.var(rate_samples, axis=0)

        # # TODO replace this with evaluations of the three pdfs
        # obs_corr_mean, obs_corr_var = corr_dist_moments(rate_corr, n_trial)
        # if obs_corr_var == 0.0:
        #     obs_corr_var = 0.001
        # # print obs_corr_mean, obs_corr_var
        # beta_shape, beta_rate = ct.beta_params_from_moments((obs_corr_mean + 1) / 2.0, obs_corr_var / 4.0)
        # # print beta_shape, beta_rate
        # # TODO these might be inappropriate
        # if beta_shape > 100 or beta_rate > 100 or beta_shape < 0 or beta_rate < 0:
        #     return 0
        # else:
        #     return st.beta.pdf((scc + 1) / 2.0, beta_shape, beta_rate) / 2.0

        corr_like = corr_error_pdf(scc, rate_corr, n_trial)
        mean_like = 1
        var_like = 1
        for u in range(n_unit):
            mean_like *= mean_error_pdf(sc_means[0, u], rate_means[u], rate_vars[u], n_trial)
            var_like *= var_error_pdf(sc_vars[0, u], rate_vars[u], n_trial)

        return corr_like * mean_like * var_like

    def pairwise_corr_numerical_posterior(self, n_trial, sc_corr, sc_means, sc_vars, n_transform_samples, n_marginal_samples, n_eval_points):
        # TODO include single-cell statistics to likelihood
        # TODO handle multiple observations
        n_unit = 2
        eval_points = np.linspace(-1, 1, n_eval_points)
        posterior = np.zeros(n_eval_points)
        # sample from priors of mean and variance
        mean_samples = self.sample_mean_prior(n_marginal_samples, n_unit)
        var_samples = self.sample_variance_prior(n_marginal_samples, n_unit)
        for i in range(n_eval_points):
            ct.printProgress(i, n_eval_points, prefix='Progress:', suffix='Complete', barLength=50)
            act_mpc = eval_points[i]
            marginal_likelihood = 0.0
            for l in range(n_marginal_samples):
                act_marginal_likelihood = self.evaluate_pairwise_corr_likelihood(n_trial, sc_corr, sc_means, sc_vars, act_mpc, mean_samples[:, l], var_samples[:, l], n_transform_samples)
                # print act_marginal_likelihood
                if np.isnan(act_marginal_likelihood):
                    raise ValueError("nan in the likelihood")
                marginal_likelihood += act_marginal_likelihood
            act_prior = self.evaluate_corr_prior(act_mpc)
            posterior[i] = marginal_likelihood * act_prior
        # print posterior
        # posterior = posterior / np.sum(posterior)
        posterior = posterior / (np.sum(posterior) * (2.0 / len(eval_points)))
        return posterior

    def plot_inference(self, sc_corr, sc_mean, sc_var, post_corr, post_mean, post_var, true_corr=None, true_mean=None, true_var=None, n_trial=None):
        # TODO handle numerical posteriors too
        # TODO support multiple bins

        pl.figure(figsize=(20, 10))

        n_obs = sc_mean.shape[0]
        n_unit = sc_mean.shape[1]
        n_pairs = n_unit * (n_unit - 1) / 2

        nn_corr = None
        nn_mean = None
        nn_var = None
        if true_corr is not None:
            nn_corr, nn_mean, nn_var = self.generate(true_corr, true_mean, true_var, 100000, 1, n_obs=1)

        mp_col = 'red'
        sc_nn_col = 'green'
        sc_obs_col = 'black'
        hist_col = 'grey'

        num_row = 3
        num_col = max([n_unit, n_pairs])

        act_row = 0
        act_col = 1
        for i in range(n_pairs):
            pl.subplot(num_row, num_col, i + 1)
            pl.hist(post_corr[:, act_row, act_col], bins=40, normed=1, facecolor=hist_col, edgecolor=hist_col, label='Posterior')
            if true_corr is not None:
                x_dense = np.linspace(-1, 1, 200)
                samp_distr = np.zeros(len(x_dense))
                for xi in range(len(x_dense)):
                    samp_distr[xi] = corr_error_pdf(x_dense[xi], nn_corr[0, act_row, act_col], n_trial)
                pl.plot(x_dense, samp_distr, color=sc_nn_col, linewidth=2, label='SC obs. samp. dist.')
                pl.plot(true_corr[act_row, act_col] * np.ones((1, 2)).T, [0, pl.gca().get_ylim()[1]], color=mp_col, linestyle='-', linewidth=2, label='MP ground truth')
            for o in range(n_obs):
                if o == 0:
                    pl.plot(sc_corr[o, act_row, act_col] * np.ones((1, 2)).T, [0, pl.gca().get_ylim()[1]], color=sc_obs_col, linestyle='-', linewidth=2, label='SC observation')
                else:
                    pl.plot(sc_corr[o, act_row, act_col] * np.ones((1, 2)).T, [0, pl.gca().get_ylim()[1]], color=sc_obs_col, linestyle='-', linewidth=2)

            pl.xlim([-1, 1])
            if i == 0:
                pl.ylabel('MP corr')
                if sc_corr[0, 0, 1] > 0:
                    pl.legend(loc=2)
                else:
                    pl.legend(loc=1)
            pl.title('Pair %d %d' % (act_row, act_col))

            act_col += 1
            if act_col == n_unit:
                act_row += 1
                act_col = act_row + 1

        for i in range(n_unit):
            pl.subplot(num_row, num_col, num_col + i + 1)

            pl.hist(post_mean[:, i], bins=40, normed=1, facecolor=hist_col, edgecolor=hist_col)
            if true_corr is not None:
                x_dense = np.linspace(pl.gca().get_xlim()[0] - 1, np.max(sc_mean[:, i]) + 1, 200)
                samp_distr = np.zeros(len(x_dense))
                for xi in range(len(x_dense)):
                    samp_distr[xi] = mean_error_pdf(x_dense[xi], nn_mean[0, i], nn_var[0, i], n_trial)
                pl.plot(x_dense, samp_distr, color=sc_nn_col, linewidth=2)
                pl.plot(true_mean[i] * np.ones((1, 2)).T, [0, pl.gca().get_ylim()[1]], color=mp_col, linestyle='-', linewidth=2)
            for o in range(n_obs):
                pl.plot(sc_mean[o, i] * np.ones((1, 2)).T, [0, pl.gca().get_ylim()[1]], color=sc_obs_col, linestyle='-', linewidth=2)

            if i == 0:
                pl.ylabel('MP mean')
            pl.title('Unit %d' % i)

            pl.subplot(num_row, num_col, 2 * num_col + i + 1)
            pl.hist(post_var[:, i], bins=40, normed=1, facecolor=hist_col, edgecolor=hist_col)
            if true_corr is not None:
                x_dense = np.linspace(pl.gca().get_xlim()[0] - 1, np.max(sc_var[:, i]) + 1, 200)
                samp_distr = np.zeros(len(x_dense))
                for xi in range(len(x_dense)):
                    samp_distr[xi] = var_error_pdf(x_dense[xi], nn_var[0, i], n_trial)
                pl.plot(x_dense, samp_distr, color=sc_nn_col, linewidth=2)
                pl.plot(true_var[i] * np.ones((1, 2)).T, [0, pl.gca().get_ylim()[1]], color=mp_col, linestyle='-', linewidth=2)
            for o in range(n_obs):
                pl.plot(sc_var[o, i] * np.ones((1, 2)).T, [0, pl.gca().get_ylim()[1]], color=sc_obs_col, linestyle='-', linewidth=2)
            if i == 0:
                pl.ylabel('MP var')
        pl.show()

    def plot_numerical_inference(self, sc_corr, post_corr, true_corr=None, true_mean=None, true_var=None, n_trial=None):
        # TODO merge this with the previous function
        n_obs = sc_corr.shape[0]

        nn_corr = None
        nn_mean = None
        nn_var = None
        if true_corr is not None:
            nn_corr, nn_mean, nn_var = self.generate(true_corr, true_mean, true_var, 100000, 1, n_obs=1)

        mp_col = 'red'
        sc_nn_col = 'green'
        sc_obs_col = 'black'
        hist_col = 'grey'

        x = np.linspace(-1, 1, len(post_corr))
        pl.fill_between(x, post_corr, facecolor=hist_col)
        pl.plot(x, post_corr, color=hist_col, label='Posterior', linewidth=4)
        if true_corr is not None:
            x_dense = np.linspace(-1, 1, 200)
            samp_distr = np.zeros(len(x_dense))
            for xi in range(len(x_dense)):
                samp_distr[xi] = corr_error_pdf(x_dense[xi], nn_corr[0, 0, 1], n_trial)
            pl.plot(x_dense, samp_distr, color=sc_nn_col, linewidth=2, label='SC obs. samp. dist.')
            pl.plot(true_corr[0, 1] * np.ones((1, 2)).T, [0, pl.gca().get_ylim()[1]], color=mp_col, linestyle='-', linewidth=2, label='MP ground truth')
        for o in range(n_obs):
            if o == 0:
                pl.plot(sc_corr[o, 0, 1] * np.ones((1, 2)).T, [0, pl.gca().get_ylim()[1]], color=sc_obs_col, linestyle='-', linewidth=2, label='SC observation')
            else:
                pl.plot(sc_corr[o, 0, 1] * np.ones((1, 2)).T, [0, pl.gca().get_ylim()[1]], color=sc_obs_col, linestyle='-', linewidth=2)

        pl.xlim([-1, 1])
        if sc_corr[0, 0, 1] > 0:
            pl.legend(loc=2)
        else:
            pl.legend(loc=1)
        pl.show()
