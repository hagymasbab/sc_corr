import numpy as np
from corrmeasure import correlationMeasurementModel as cMM
import matplotlib.pyplot as pl
import numpy.random as rnd

seed = 4
rnd.seed(seed)
corr = -0.4
means = [2, 3]
trialNums = [150, 25]
cmm = cMM('corr_rate.pkl')
n_bin = 1

for m in range(len(means)):
    act_mean = means[m]
    for t in range(len(trialNums)):
        n_trial = trialNums[t]
        sc_corr, sc_mean, sc_var = cmm.generate(np.array([[1, corr], [corr, 1]]), act_mean * np.ones(2), np.ones(2), n_trial, n_bin, seed)
        mps_corr, mps_mean, mps_var = cmm.infer(sc_corr, sc_mean, sc_var, n_trial, n_bin, 200, 2000, 2, seed)

        pl.subplot(len(trialNums), len(means), t * len(means) + m + 1)
        pl.hist(mps_corr[:, 0, 1], bins=40, normed=True)
        pl.plot(corr * np.ones((1, 2)).T, [0, pl.gca().get_ylim()[1]], color='red', linestyle='-', linewidth=2)
        pl.plot(sc_corr[0, 1] * np.ones((1, 2)).T, [0, pl.gca().get_ylim()[1]], color='green', linestyle='-', linewidth=2)
        pl.xlim([-1, 1])
        pl.xlabel('mean=%.2f, n_trials=%d' % (act_mean, n_trial))
pl.show()
