import numpy as np
from corrmeasure import correlationMeasurementModel as cMM
from csnltools import histogramMode, heatmapPlot
import pickle
import matplotlib.pyplot as plt

recalc = True

n_unit = 3
n_pair = n_unit * (n_unit - 1) / 2
n_bin = 1
n_trial = 100
n_samp = 500

n_corrs = 5
n_means = 2
n_reest = 10

mean_vals = np.linspace(1, 3, n_means)
corr_vals = np.linspace(-0.9, 0.9, n_corrs)
var_mp = 1 * np.ones(n_unit)

if recalc:
    cmm = cMM('corr_rate.pkl')

    est_vs_true = np.zeros((n_means, n_corrs, n_corrs, n_reest, 2))

    for m in range(n_means):
        mu_mp = mean_vals[m] * np.ones(n_unit)
        for c1 in range(n_corrs):
            for c2 in range(n_corrs):
                f = open('biastest_progress.txt', 'a')
                f.write('Mean %d/%d corr1 %d/%d corr2 %d/%d\n' % (n_means, m+1, n_corrs, c1+1, n_corrs, c2+1))
                f.close()
                corr_mp = np.identity(n_unit)
                corr_mp[0, 1] = corr_vals[c1]
                corr_mp[0, 2] = corr_vals[c2]
                corr_mp[1, 2] = corr_vals[c1] * corr_vals[c2]
                corr_mp = corr_mp + corr_mp.T - np.identity(n_unit)

                for i in range(n_reest):
                    sc_corr, sc_mean, sc_var = cmm.generate(corr_mp, mu_mp, var_mp, n_trial, n_bin)
                    mps_corr, mps_mean, mps_var = cmm.infer(sc_corr, sc_mean, sc_var, n_trial, n_bin, 100, n_samp, 2)
                    MAP_corr1 = histogramMode(mps_corr[:, 0, 1], 40)
                    MAP_corr2 = histogramMode(mps_corr[:, 0, 2], 40)
                    est_vs_true[m, c1, c2, i, 0] = corr_vals[c1] - MAP_corr1
                    est_vs_true[m, c1, c2, i, 1] = corr_vals[c2] - MAP_corr2

    pickle.dump(est_vs_true, open('biastest2D.pkl', 'wb'))
else:
    est_vs_true = pickle.load(open('biastest2D.pkl', 'rb'))

for m in range(n_means):
    for p in range(2):
        ax = plt.subplot(2, n_means, p*n_means+m+1)
        avg_error = np.mean(est_vs_true[m, :, :, :, :], axis=2)
        heatmapPlot(ax, avg_error[:, :, p], corr_vals, corr_vals, "average MAP corr of pair 1-%d" % (p+2))
        plt.xlabel('MP corr, pair 1-2')
        plt.ylabel('MP corr, pair 1-3')
        if p == 0:
            plt.title("MP mean=%.2f" % mean_vals[m])
plt.show()
