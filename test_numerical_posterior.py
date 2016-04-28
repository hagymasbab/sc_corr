from corrmeasure import correlationMeasurementModel as cMM, corr_error_pdf
import numpy as np
import matplotlib.pyplot as pl
import numpy.random as rn
from scipy import interpolate
from scipy.optimize import curve_fit
from scipy.stats import beta
import csnltools as ct


np.seterr(all='raise')
# rn.seed(4)
cmm = cMM('corr_rate.pkl')
n_trial = 100

mu_mp = 1 * np.ones(2)
var_mp = 1 * np.ones(2)
true_corr = -0.3
corr_mp = np.array([[1, true_corr], [true_corr, 1]])
nn_corrmat, nn_mean, nn_var = cmm.generate(corr_mp, mu_mp, var_mp, n_trial*1000, 1)
nonoise_corr = nn_corrmat[0, 1]
print(nonoise_corr)
sc_corrmat, sc_mean, sc_var = cmm.generate(corr_mp, mu_mp, var_mp, n_trial, 1)
corr_sc = sc_corrmat[0, 1]

x_dense = np.linspace(-1, 1, 200)
corr_distr = np.zeros(len(x_dense))
for i in range(len(x_dense)):
    corr_distr[i] = corr_error_pdf(x_dense[i], nonoise_corr, n_trial)
# likelihood = cmm.evaluate_pairwise_corr_likelihood(n_trial, corr_sc, true_corr, mu_mp, var_mp, 100)
# print corr_sc, likelihood

res = 30
x = np.linspace(-1, 1, res)
post = cmm.pairwise_corr_numerical_posterior(n_trial, corr_sc, 100, 100, res)

# tck = interpolate.splrep(x, post, s=0)
# y_spline = interpolate.splev(x_spline, tck, der=0)

# betafit_params, param_cov = curve_fit(betacurve, x, post, (2, 1))
# print betafit_params
# for i in range(len(x_spline)):
#     beta_vals[i] = betacurve(x_spline[i], betafit_params[0], betafit_params[1])

pl.plot(x, post)
pl.plot(x_dense, corr_distr * (max(post) / max(corr_distr)), color='grey')
pl.plot(np.ones(2) * true_corr, pl.ylim(), color='black', linewidth=2)
pl.plot(np.ones(2) * nonoise_corr, pl.ylim(), color='grey')
pl.plot(np.ones(2) * corr_sc, pl.ylim(), color='red')
pl.plot(np.ones(2) * ct.pdf_mean(x, post), pl.ylim(), color='green', linestyle='--')
pl.plot(np.ones(2) * ct.pdf_map(x, post), pl.ylim(), color='orange', linestyle='--')
pl.show()
