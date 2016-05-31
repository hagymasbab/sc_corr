from pystan import StanModel
import pickle
from corrmeasure import correlationMeasurementModel as cMM
import csnltools as ct
import numpy as np


# sm = StanModel(file='corrmeasure.stan', verbose=False)
# with open('corrmeasure.pkl', 'wb') as f:
#     pickle.dump(sm, f)

cmm = cMM('corrmeasure.pkl')
sc_corr, sc_mean, sc_var = cmm.quick_sample(1)
samp_init = {'mp_corr_chol': np.linalg.cholesky(ct.corrmat_2by2(cmm.quick_corr)), 'mp_mean_vec': np.ones(2) * cmm.quick_mean, 'mp_var_vec': np.ones(2) * cmm.quick_var}
# mps_corr, mps_mean, mps_var = cmm.infer(sc_corr, sc_mean, sc_var, 100, 1, 100, 1000, n_chains=1, init=[samp_init])
post = cmm.pairwise_corr_numerical_posterior(100, sc_corr[0, 0, 1], sc_mean, sc_var, 1000, 500, 60)
# cmm.plot_inference(sc_corr, sc_mean, sc_var, mps_corr, mps_mean, mps_var, ct.corrmat_2by2(cmm.quick_corr), np.ones(2) * cmm.quick_mean, np.ones(2) * cmm.quick_var, 100)
cmm.plot_numerical_inference(sc_corr, post, true_corr=ct.corrmat_2by2(cmm.quick_corr), true_mean=np.ones(2) * cmm.quick_mean, true_var=np.ones(2) * cmm.quick_var, n_trial=100)
