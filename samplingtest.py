import pickle
from pystan import StanModel
import matplotlib.pyplot as pl

# x = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
x = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
# x = [1, 1, 1]

cf_dat = {
    'N': len(x),
    'x': x,
    'prior_width': 0.2
}

recompile = False

if recompile:
    sm = StanModel(file='coinflip.stan')
    with open('coinflip.pkl', 'wb') as f:
        pickle.dump(sm, f)
else:
    sm = pickle.load(open('coinflip.pkl', 'rb'))

# fit = sm.sampling(data=cf_dat, iter=2000, chains=2)
fit = sm.sampling(data=cf_dat, iter=20, chains=1, seed='random', init=[{'beta': 0.5}], warmup=10)
estimation = fit.extract(permuted=True)

print(estimation['beta'])

pl.hist(estimation['beta'], bins=40)
pl.xlim([0, 1])
pl.show()

cmm = cMM('corr_rate.pkl')