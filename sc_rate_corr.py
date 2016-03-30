import pickle
import numpy as np
import numpy.random as rnd
import matplotlib.pyplot as pl
from scipy.io import loadmat

n_samp_est = 200
n_iter = 2000
n_chains = 2

n_unit = 2
n_pairs = n_unit * (n_unit - 1) / 2
spont_start_bin = 0
evoked_start_bin = 70
segment_length = 50

# load data
offset = 3
lsc = loadmat('../majom/data/SC_nat_atoc102a03_bin10.mat')
sc = lsc['spikeCount'][offset:offset + n_unit, :, :]
n_trial = sc.shape[1]

sc_spont = np.sum(sc[:, :, spont_start_bin:spont_start_bin + segment_length], axis=2)  # n_unit x n_trial
spont_mean = np.mean(sc_spont, axis=1)
spont_var = np.var(sc_spont, axis=1)
spont_corr = np.corrcoef(sc_spont)

sc_evoked = np.sum(sc[:, :, evoked_start_bin:evoked_start_bin + segment_length], axis=2)
evoked_mean = np.mean(sc_evoked, axis=1)
evoked_var = np.var(sc_evoked, axis=1)
evoked_corr = np.corrcoef(sc_evoked)

sm = pickle.load(open('corr_rate.pkl', 'rb'))
corr_dat = {
    'n_trial': n_trial,
    'n_bin': 1,
    'n_unit': n_unit,
    'sc_corr_mat': spont_corr,
    'sc_mean_vec': spont_mean,
    'sc_var_vec': spont_var,
    'mp_mean_prior_mean': 1,
    'mp_mean_prior_var': 1,
    'mp_var_prior_shape': 1,
    'mp_var_prior_scale': 1,
    'mp_corr_prior_conc': 2,
    'exponent_prior_mean': 1.1,
    'base_rate_prior_mean': 10,
    'threshold_prior_mean': 1.9,
    'n_samples': n_samp_est,
    'stdnorm_samples': rnd.normal(size=(n_samp_est, n_unit))
}

fit = sm.sampling(data=corr_dat, iter=n_iter, chains=n_chains)
spont_est = fit.extract(permuted=True)

corr_dat['sc_corr_mat'] = evoked_corr
corr_dat['sc_mean_vec'] = evoked_mean
corr_dat['sc_var_vec'] = evoked_var

fit = sm.sampling(data=corr_dat, iter=n_iter, chains=n_chains)
evoked_est = fit.extract(permuted=True)


num_row = 4
num_col = max([n_unit, n_pairs])
alpha = 0.5
hist_bins = 30
spont_col = 'blue'
evoked_col = 'red'

act_row = 0
act_col = 1
for i in range(n_pairs):
    pl.subplot(num_row, num_col, i + 1)
    pl.scatter(sc_spont[act_row, :], sc_spont[act_col, :], facecolor=spont_col, alpha=alpha)
    pl.scatter(sc_evoked[act_row, :], sc_evoked[act_col, :], facecolor=evoked_col, alpha=alpha)
    if i == 0:
        pl.ylabel('observed SC')
    pl.title('Pair %d %d' % (act_row, act_col))

    pl.subplot(num_row, num_col, 1 * num_col + i + 1)
    pl.hist(spont_est['mp_corr_mat'][:, act_row, act_col], bins=hist_bins, facecolor=spont_col, alpha=alpha)
    pl.hist(evoked_est['mp_corr_mat'][:, act_row, act_col], bins=hist_bins, facecolor=evoked_col, alpha=alpha)
    pl.plot(spont_corr[act_row, act_col] * np.ones((1, 2)).T, [0, pl.gca().get_ylim()[1]], color=spont_col, linestyle='-', linewidth=2)
    pl.plot(evoked_corr[act_row, act_col] * np.ones((1, 2)).T, [0, pl.gca().get_ylim()[1]], color=evoked_col, linestyle='-', linewidth=2)
    pl.xlim([-1, 1])
    if i == 0:
        pl.ylabel('MP corr')

    act_col += 1
    if act_col == n_unit:
        act_row += 1
        act_col = act_row + 1

for i in range(n_unit):
    pl.subplot(num_row, num_col, 2 * num_col + i + 1)
    pl.hist(spont_est['mp_mean_vec'][:, i], bins=hist_bins, facecolor=spont_col, alpha=alpha)
    pl.hist(evoked_est['mp_mean_vec'][:, i], bins=hist_bins, facecolor=evoked_col, alpha=alpha)
    if i == 0:
        pl.ylabel('MP mean')
    pl.title('Unit %d' % i)

    pl.subplot(num_row, num_col, 3 * num_col + i + 1)
    pl.hist(spont_est['mp_var_vec'][:, i], bins=hist_bins, facecolor=spont_col, alpha=alpha)
    pl.hist(evoked_est['mp_var_vec'][:, i], bins=hist_bins, facecolor=evoked_col, alpha=alpha)
    if i == 0:
        pl.ylabel('MP var')

pl.show()
