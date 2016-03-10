import pickle
import numpy as np
import numpy.random as rnd
from pystan import StanModel
import matplotlib.pyplot as pl

recompile = True

base_rate = 10
threshold = 1.9
exponent = 1.1
mp_corr = 0.4
n_trial = 50
n_bins = 10

mu_mp = [1, 1]
C_mp = np.identity(2)
C_mp[0, 1] = mp_corr
C_mp[1, 0] = C_mp[0, 1]
U = rnd.multivariate_normal(mu_mp, C_mp, (n_trial, n_bins))
rate = np.zeros((n_trial, n_bins, 2))
for t in range(n_trial):
    for b in range(n_bins):
        for c in range(2):
            if U[t, b, c] > threshold:
                rate[t, b, c] = base_rate * (U[t, b, c] - threshold) ** exponent
spike_count = np.floor(np.sum(rate, axis=1))
sc_mean_vec = np.mean(spike_count, axis=0)
print(sc_mean_vec)
sc_corr = np.corrcoef(spike_count.T)[0, 1]
print(sc_corr)

corr_dat = {
    'n_trial': n_trial,
    'n_bins': n_bins,
    'sc_corr': sc_corr,
    'sc_mean_vec': sc_mean_vec
}

if recompile:
    sm = StanModel(file='corr_rate.stan')
    with open('corr_rate.pkl', 'wb') as f:
        pickle.dump(sm, f)
else:
    sm = pickle.load(open('corr_rate.pkl', 'rb'))

fit = sm.sampling(data=corr_dat, iter=2000, chains=2)
estimation = fit.extract(permuted=True)

pl.subplot(221)
pl.hist(estimation['mp_corr'], bins=40)
pl.plot(mp_corr * np.ones((1, 2)), [0, pl.gca().get_ylim()[1]], color='r', linestyle='-', linewidth=2)

pl.subplot(222)
pl.hist(estimation['base_rate'], bins=40)

pl.subplot(223)
pl.hist(estimation['threshold'], bins=40)

pl.subplot(224)
pl.hist(estimation['exponent'], bins=40)
