import numpy as np
# import numpy.random as rnd
from corrmeasure import correlationMeasurementModel as cMM
import pickle
import matplotlib.pyplot as pl
from csnltools import histogramMode


recalc = True
n_reest = 5

n_unit = 2
n_pair = n_unit * (n_unit - 1) / 2
n_bin = 1
n_trial = 100

mu_mp = 2 * np.ones(n_unit)
var_mp = 1 * np.ones(n_unit)
true_corr = 0.5
corr_mp = np.array([[1, true_corr], [true_corr, 1]])

sampNums = [100, 1000]
chainNums = [1, 2]
genSampNums = [100, 200]

samples = np.empty((len(sampNums), len(chainNums), len(genSampNums), n_reest, np.max(sampNums)))
samples[:] = np.NAN

if recalc:
    cmm = cMM('corr_rate.pkl')
    cmm.mp_corr_prior_conc = 2

    for sn in range(len(sampNums)):
        n_samp = sampNums[sn]
        for cn in range(len(chainNums)):
            n_chain = chainNums[cn]
            n_step_per_chain = np.floor(2 * n_samp / n_chain)
            n_samp = n_step_per_chain * n_chain / 2
            for gsn in range(len(genSampNums)):
                n_samp_gen = genSampNums[gsn]
                sc_corr, sc_mean, sc_var = cmm.generate(corr_mp, mu_mp, var_mp, n_trial, n_bin)
                for re in range(n_reest):
                    mps_corr, mps_mean, mps_var = cmm.infer(sc_corr, sc_mean, sc_var, n_trial, n_bin, n_samp_gen, n_step_per_chain, n_chain)
                    samples[sn, cn, gsn, re, 0:n_samp] = mps_corr[:, 0, 1]

    pickle.dump(samples, open('mapstab.pkl', 'wb'))
else:
    samples = pickle.load(open('mapstab.pkl', 'rb'))


maps = np.zeros((len(sampNums), len(chainNums), len(genSampNums), n_reest))
map_variances = np.zeros((len(sampNums), len(chainNums), len(genSampNums)))
for sn in range(len(sampNums)):
    for cn in range(len(chainNums)):
        pl.subplot(len(sampNums), len(chainNums), sn * len(chainNums) + cn + 1)
        pl.ylim([-1, 1])
        for gsn in range(len(genSampNums)):
            for re in range(n_reest):
                # TODO handle nans
                actsamp = samples[sn, cn, gsn, re, :]
                actsamp = actsamp[actsamp is not np.NAN]
                maps[sn, cn, gsn, re] = histogramMode(actsamp, 20)
            pl.scatter((gsn+1)*np.ones(n_reest), maps[sn, cn, gsn, :])
pl.show()
