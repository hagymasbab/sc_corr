import numpy as np
import numpy.random as rnd
import pickle


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

    def generate(self, mp_corr, mp_mean, mp_var, n_trial, n_bin):
        n_unit = len(mp_mean)
        C_mp = np.diag(mp_var)
        C_mp = np.sqrt(C_mp).dot(mp_corr.dot(np.sqrt(C_mp)))
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
        return (sc_corr, sc_mean, sc_var)

    def infer(self, sc_corr, sc_mean, sc_var, n_trial, n_bin, n_samp_est, n_iter, n_chains):
        n_unit = len(sc_mean)
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

        fit = self.sm.sampling(data=corr_dat, iter=n_iter, chains=n_chains)
        estimation = fit.extract(permuted=True)
        return (estimation['mp_corr_mat'], estimation['mp_mean_vec'], estimation['mp_var_vec'])
