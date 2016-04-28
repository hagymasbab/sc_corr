import numpy as np
import numpy.random as rnd
from corrmeasure import correlationMeasurementModel as cMM
import pickle
import matplotlib.pyplot as pl
# from csnltools import histogramMode


recalc = False
n_reest = 18
n_averaging = 6

n_unit = 2
n_pair = n_unit * (n_unit - 1) / 2
n_bin = 1
n_trial = 100

mu_mp = 2 * np.ones(n_unit)
var_mp = 1 * np.ones(n_unit)
true_corr = 0.5
corr_mp = np.array([[1, true_corr], [true_corr, 1]])

samp_init = {'mp_corr_chol': np.linalg.cholesky(corr_mp), 'mp_mean_vec': mu_mp, 'mp_var_vec': var_mp}

sampNums = [500, 2000]
chainNums = [5]
genSampNums = [100, 200, 300]
thinning = 1

samples = np.empty((len(sampNums), len(chainNums), len(genSampNums), n_reest, np.max(sampNums)))
samples[:] = np.NAN

if recalc:
    cmm = cMM('corr_rate.pkl')
    cmm.mp_corr_prior_conc = 2

    sc_corr, sc_mean, sc_var = cmm.generate(corr_mp, mu_mp, var_mp, n_trial, n_bin)
    obs_corr = sc_corr[0, 1]
    for sn in range(len(sampNums)):
        n_samp = sampNums[sn]
        for cn in range(len(chainNums)):
            n_chain = chainNums[cn]
            n_step_per_chain = np.floor(2 * n_samp / n_chain)
            n_samp = n_step_per_chain * n_chain / 2

            initvals = []
            for i in range(n_chain):
                initvals.append(samp_init)
            print(initvals)

            for gsn in range(len(genSampNums)):
                n_samp_gen = genSampNums[gsn]
                stdnorm_samples = rnd.normal(size=(n_samp_gen, n_unit))
                for re in range(n_reest):
                    mps_corr, mps_mean, mps_var = cmm.infer(sc_corr, sc_mean, sc_var, n_trial, n_bin, stdnorm_samples, n_step_per_chain*thinning, n_chain, thin=thinning)
                    # mps_corr, mps_mean, mps_var = cmm.infer(sc_corr, sc_mean, sc_var, n_trial, n_bin, n_samp_gen, n_step_per_chain, n_chain, init=initvals)
                    samples[sn, cn, gsn, re, 0:n_samp] = mps_corr[:, 0, 1]

    pickle.dump((samples, obs_corr), open('mapstab.pkl', 'wb'))
else:
    loaded = pickle.load(open('mapstab.pkl', 'rb'))
    samples = loaded[0]
    obs_corr = loaded[1]


n_estimates = n_reest / n_averaging
maps_avg = np.zeros((len(sampNums), len(chainNums), len(genSampNums), n_estimates))
maps = np.zeros((len(sampNums), len(chainNums), len(genSampNums), n_reest))
map_variances = np.zeros((len(sampNums), len(chainNums), len(genSampNums)))
for sn in range(len(sampNums)):
    for cn in range(len(chainNums)):
        pl.subplot(len(sampNums), len(chainNums), sn * len(chainNums) + cn + 1)
        if cn == 0:
            pl.ylabel("Sample num %d" % sampNums[sn])
        if sn == 0:
            pl.title("Chain num %d" % chainNums[cn])
        elif sn == len(sampNums) - 1:
            pl.xlabel("Transform samps")
        pl.ylim([-1, 1])
        for gsn in range(len(genSampNums)):
            for re in range(n_reest):
                actsamp = samples[sn, cn, gsn, re, :]
                actsamp = actsamp[actsamp is not np.NAN]
                # maps[sn, cn, gsn, re] = histogramMode(actsamp, 20)
                maps[sn, cn, gsn, re] = np.mean(actsamp)
            for ae in range(n_estimates):
                actsamp = samples[sn, cn, gsn, ae * n_averaging:(ae + 1) * n_averaging, :]
                for a in range(n_averaging):
                    actsamp[a, :] = actsamp[a, actsamp[a, :] is not np.NAN]
                maps_avg[sn, cn, gsn, ae] = np.mean(np.mean(actsamp, axis=1))
            pl.scatter(genSampNums[gsn] * np.ones(n_reest), maps[sn, cn, gsn, :])
            pl.scatter(genSampNums[gsn] * np.ones(n_estimates), maps_avg[sn, cn, gsn, :], color='red')
            print maps_avg[sn, cn, :, :]
        pl.plot(pl.xlim(), true_corr * np.ones(2), color="black", linestyle='--', linewidth=1)
        pl.plot(pl.xlim(), obs_corr * np.ones(2), color="red", linestyle='-', linewidth=1)
pl.show()
