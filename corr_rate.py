import pickle
import numpy as np
import numpy.random as rnd
from pystan import StanModel
import matplotlib.pyplot as pl

recompile = False
rnd.seed(11)

n_trial = 100
n_bin = 1
base_rate = 10
threshold = 1.9
exponent = 1.1

mu_mp = [2, 2]
mp_corr = 0.4
corrmat = np.array([[1, mp_corr], [mp_corr, 1]])
C_mp = np.diag([2, 2])
# C_mp[0, 1] = mp_corr
# C_mp[1, 0] = C_mp[0, 1]
C_mp = np.sqrt(C_mp).dot(corrmat.dot(np.sqrt(C_mp)))
U = rnd.multivariate_normal(mu_mp, C_mp, (n_trial, n_bin))
rate = np.zeros((n_trial, n_bin, 2))
for t in range(n_trial):
    for b in range(n_bin):
        for c in range(2):
            if U[t, b, c] > threshold:
                rate[t, b, c] = base_rate * (U[t, b, c] - threshold) ** exponent
spike_count = np.floor(np.sum(rate, axis=1))
sc_mean_vec = np.mean(spike_count, axis=0)
sc_var_vec = np.var(spike_count, axis=0)
sc_corr_mat = np.corrcoef(spike_count.T)
print(sc_mean_vec)
print(sc_var_vec)
print(sc_corr_mat)

n_samples = 100
stdnorm_samples = rnd.normal(size=(n_samples, len(mu_mp)))
# pickle.dump(stdnorm_samples, open('stdns.pkl', 'wb'))
# stdnorm_samples = pickle.load(open('stdns.pkl', 'rb'))

corr_dat = {
    'n_trial': n_trial,
    'n_bin': n_bin,
    'n_unit': len(mu_mp),
    'sc_corr_mat': sc_corr_mat,
    'sc_mean_vec': sc_mean_vec,
    'sc_var_vec': sc_var_vec,
    'mp_mean_prior_mean': 1,
    'mp_mean_prior_var': 1,
    'mp_var_prior_shape': 1,
    'mp_var_prior_scale': 1,
    'mp_corr_prior_conc': 1,
    'exponent_prior_mean': exponent,
    'base_rate_prior_mean': base_rate,
    'threshold_prior_mean': threshold,
    'n_samples': n_samples,
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

pl.subplot(421)
pl.scatter(U[:, 0, 0], U[:, 0, 1])

pl.subplot(422)
pl.scatter(spike_count[:, 0], spike_count[:, 1])

pl.subplot(423)
pl.hist(estimation['mp_corr_mat'][:, 0, 1], bins=40)
pl.plot(mp_corr * np.ones((1, 2)).T, [0, pl.gca().get_ylim()[1]], color='r', linestyle='-', linewidth=2)
pl.plot(sc_corr_mat[0, 1] * np.ones((1, 2)).T, [0, pl.gca().get_ylim()[1]], color='g', linestyle='-', linewidth=2)
pl.xlim([-1, 1])

pl.subplot(425)
pl.hist(estimation['mp_mean_vec'][:, 0], bins=40)
pl.plot(mu_mp[0] * np.ones((1, 2)).T, [0, pl.gca().get_ylim()[1]], color='r', linestyle='-', linewidth=2)
pl.plot(sc_mean_vec[0] * np.ones((1, 2)).T, [0, pl.gca().get_ylim()[1]], color='g', linestyle='-', linewidth=2)

pl.subplot(426)
pl.hist(estimation['mp_mean_vec'][:, 1], bins=40)
pl.plot(mu_mp[1] * np.ones((1, 2)).T, [0, pl.gca().get_ylim()[1]], color='r', linestyle='-', linewidth=2)
pl.plot(sc_mean_vec[1] * np.ones((1, 2)).T, [0, pl.gca().get_ylim()[1]], color='g', linestyle='-', linewidth=2)

pl.subplot(427)
pl.hist(estimation['mp_var_vec'][:, 0], bins=40)
pl.plot(C_mp[0, 0] * np.ones((1, 2)).T, [0, pl.gca().get_ylim()[1]], color='r', linestyle='-', linewidth=2)
pl.plot(sc_var_vec[0] * np.ones((1, 2)).T, [0, pl.gca().get_ylim()[1]], color='g', linestyle='-', linewidth=2)

pl.subplot(428)
pl.hist(estimation['mp_var_vec'][:, 1], bins=40)
pl.plot(C_mp[1, 1] * np.ones((1, 2)).T, [0, pl.gca().get_ylim()[1]], color='r', linestyle='-', linewidth=2)
pl.plot(sc_var_vec[1] * np.ones((1, 2)).T, [0, pl.gca().get_ylim()[1]], color='g', linestyle='-', linewidth=2)

pl.show()
