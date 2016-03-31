import numpy as np
from corrmeasure import correlationMeasurementModel as cMM


def histogramMode(data, resolution):
    pass

cmm = cMM('corr_rate.pkl')

n_unit = 2
n_bin = 1
n_trial = 100

n_corrs = 5
n_means = 3
n_reest = 5

est_vs_true = np.zeros((n_means, n_corrs, n_reest, 2))

for mp_mean in np.linspace(0, 3, n_means):
    for mp_corr in np.linspace(-1, 1, n_corrs):
        mu_mp = mp_mean * np.ones(n_unit)
        var_mp = 1 * np.ones(n_unit)
        corr_mp = [[1, mp_corr], [mp_corr, 1]]

        for i in range(n_reest):
            sc_synth = cmm.generate(mu_mp, var_mp, corr_mp, n_trial, n_bin)
            mp_samples = cmm.infer(sc_synth[0], sc_synth[1], sc_synth[2], n_trial, n_bin, 100, 2000, 2)
            MAP_corr = histogramMode(mp_samples(0)[:, 0, 1])
            est_vs_true[]
