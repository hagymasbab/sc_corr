import pickle
import numpy as np
import numpy.random as rnd
from pystan import StanModel
import matplotlib.pyplot as pl

recompile = True

n_trial = 50
n_bin = 10
base_rate = 10
threshold = 1.9
exponent = 1.1

mu_mp = [1, 1]
mp_corr = 0.4
C_mp = np.identity(2)  # variance is unit int this test
C_mp[0, 1] = mp_corr
C_mp[1, 0] = C_mp[0, 1]
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

corr_dat = {
    'n_trial': n_trial,
    'n_bin': n_bin,
    'n_unit': len(mu_mp),
    'sc_corr_mat': sc_corr_mat,
    'sc_mean_vec': sc_mean_vec,
    'sc_var_vec': sc_var_vec,
    'mp_mean_prior_mean': 1,
    'mp_mean_prior_var': 0.1,
    'mp_var_prior_shape': 1,
    'mp_var_prior_scale': 1,
    'mp_corr_prior_conc': 1
}

if recompile:
    sm = StanModel(file='corr_rate.stan')
    with open('corr_rate.pkl', 'wb') as f:
        pickle.dump(sm, f)
else:
    sm = pickle.load(open('corr_rate.pkl', 'rb'))

fit = sm.sampling(data=corr_dat, iter=2000, chains=2)
estimation = fit.extract(permuted=True)

# pl.subplot(221)
# pl.hist(estimation['mp_corr_mat'], bins=40)
# pl.plot(mp_corr * np.ones((1, 2)), [0, pl.gca().get_ylim()[1]], color='r', linestyle='-', linewidth=2)
