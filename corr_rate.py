import pickle
import numpy as np
import numpy.random as rnd
from pystan import StanModel
import matplotlib.pyplot as pl

recompile = False
# rnd.seed(15)
n_samp_gen = 1000
n_samp_est = 200

n_trial = 1000
n_bin = 1
n_unit = 3
base_rate = 10
threshold = 1.9
exponent = 1.1

n_pairs = n_unit * (n_unit - 1) / 2
mu_mp = 1 * np.ones(n_unit)
var_mp = 2 * np.ones(n_unit)
# mp_corrs = [0.8, -0.8, -0.8, 0.8, -0.8, 0.8]
mp_corrs = [0.4, -0.1, -0.5]
corrmat = np.identity(n_unit)
act_row = 0
act_col = 1
for i in range(n_pairs):
    corrmat[act_row, act_col] = mp_corrs[i]
    act_col += 1
    if act_col == n_unit:
        act_row += 1
        act_col = act_row + 1
corrmat = corrmat + corrmat.T - np.identity(n_unit)

C_mp = np.diag(var_mp)
C_mp = np.sqrt(C_mp).dot(corrmat.dot(np.sqrt(C_mp)))
U = rnd.multivariate_normal(mu_mp, C_mp, (n_samp_gen, n_bin))
rate = np.zeros((n_samp_gen, n_bin, n_unit))
for t in range(n_samp_gen):
    for b in range(n_bin):
        for c in range(n_unit):
            if U[t, b, c] > threshold:
                rate[t, b, c] = base_rate * (U[t, b, c] - threshold) ** exponent

spike_count = np.floor(np.sum(rate, axis=1))  # n_samp_gen x n_unit
sc_true_mean_vec = np.mean(spike_count, axis=0)
sc_true_var_vec = np.var(spike_count, axis=0)
sc_true_corr_mat = np.corrcoef(spike_count.T)

observed_spike_count = spike_count[0:n_trial, :]
sc_mean_vec = np.mean(observed_spike_count, axis=0)
sc_var_vec = np.var(observed_spike_count, axis=0)
sc_corr_mat = np.corrcoef(observed_spike_count.T)

stdnorm_samples = rnd.normal(size=(n_samp_est, n_unit))
# pickle.dump(stdnorm_samples, open('stdns.pkl', 'wb'))
# stdnorm_samples = pickle.load(open('stdns.pkl', 'rb'))

corr_dat = {
    'n_trial': n_trial,
    'n_bin': n_bin,
    'n_unit': n_unit,
    'sc_corr_mat': sc_corr_mat,
    'sc_mean_vec': sc_mean_vec,
    'sc_var_vec': sc_var_vec,
    'mp_mean_prior_mean': 1,
    'mp_mean_prior_var': 1,
    'mp_var_prior_shape': 1,
    'mp_var_prior_scale': 1,
    'mp_corr_prior_conc': 2,
    'exponent_prior_mean': exponent,
    'base_rate_prior_mean': base_rate,
    'threshold_prior_mean': threshold,
    'n_samples': n_samp_est,
    'stdnorm_samples': stdnorm_samples
}

if recompile:
    sm = StanModel(file='corr_rate.stan', verbose=False)
    with open('corr_rate.pkl', 'wb') as f:
        pickle.dump(sm, f)
else:
    sm = pickle.load(open('corr_rate.pkl', 'rb'))

fit = sm.sampling(data=corr_dat, iter=2000, chains=2)
estimation = fit.extract(permuted=True)
cm = estimation['mp_corr_mat']


mp_col = 'r'
sc_true_col = 'y'
sc_obs_col = 'g'

num_row = 5
num_col = max([n_unit, n_pairs])

act_row = 0
act_col = 1
for i in range(n_pairs):
    pl.subplot(num_row, num_col, i + 1)
    pl.scatter(U[:, 0, act_row], U[:, 0, act_col])
    if i == 0:
        pl.ylabel('MP samples')
    pl.title('Pair %d %d' % (act_row, act_col))

    pl.subplot(num_row, num_col, num_col + i + 1)
    pl.scatter(observed_spike_count[:, act_row], observed_spike_count[:, act_col])
    if i == 0:
        pl.ylabel('observed SC')

    pl.subplot(num_row, num_col, 2 * num_col + i + 1)
    pl.hist(estimation['mp_corr_mat'][:, act_row, act_col], bins=40)
    pl.plot(corrmat[act_row, act_col] * np.ones((1, 2)).T, [0, pl.gca().get_ylim()[1]], color=mp_col, linestyle='-', linewidth=2)
    pl.plot(sc_true_corr_mat[act_row, act_col] * np.ones((1, 2)).T, [0, pl.gca().get_ylim()[1]], color=sc_true_col, linestyle='-', linewidth=2)
    pl.plot(sc_corr_mat[act_row, act_col] * np.ones((1, 2)).T, [0, pl.gca().get_ylim()[1]], color=sc_obs_col, linestyle='-', linewidth=2)
    pl.xlim([-1, 1])
    if i == 0:
        pl.ylabel('MP corr')

    act_col += 1
    if act_col == n_unit:
        act_row += 1
        act_col = act_row + 1

for i in range(n_unit):
    pl.subplot(num_row, num_col, 3 * num_col + i + 1)
    pl.hist(estimation['mp_mean_vec'][:, i], bins=40)
    pl.plot(mu_mp[i] * np.ones((1, 2)).T, [0, pl.gca().get_ylim()[1]], color=mp_col, linestyle='-', linewidth=2)
    if n_bin < 5:
        pl.plot(sc_true_mean_vec[i] * np.ones((1, 2)).T, [0, pl.gca().get_ylim()[1]], color=sc_true_col, linestyle='-', linewidth=2)
        pl.plot(sc_mean_vec[i] * np.ones((1, 2)).T, [0, pl.gca().get_ylim()[1]], color=sc_obs_col, linestyle='-', linewidth=2)
    if i == 0:
        pl.ylabel('MP mean')
    pl.title('Unit %d' % i)

    pl.subplot(num_row, num_col, 4 * num_col + i + 1)
    pl.hist(estimation['mp_var_vec'][:, i], bins=40)
    pl.plot(var_mp[i] * np.ones((1, 2)).T, [0, pl.gca().get_ylim()[1]], color=mp_col, linestyle='-', linewidth=2)
    if n_bin < 5:
        pl.plot(sc_true_var_vec[i] * np.ones((1, 2)).T, [0, pl.gca().get_ylim()[1]], color=sc_true_col, linestyle='-', linewidth=2)
        pl.plot(sc_var_vec[i] * np.ones((1, 2)).T, [0, pl.gca().get_ylim()[1]], color=sc_obs_col, linestyle='-', linewidth=2)
    if i == 0:
        pl.ylabel('MP var')

pl.show()
