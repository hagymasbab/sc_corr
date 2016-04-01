import numpy as np
from corrmeasure import correlationMeasurementModel as cMM
import pickle
import matplotlib.pyplot as plt


def histogramMode(data, resolution):
    binVals, binEdges = np.histogram(data, bins=resolution)
    maxIdx = binVals.argmax()
    return (binEdges[maxIdx] + binEdges[maxIdx+1]) / 2

recalc = True

n_unit = 2
n_bin = 1
n_trial = 100

n_corrs = 10
n_means = 3
n_reest = 5

mean_vals = np.linspace(0.5, 3, n_means)
print(mean_vals)

corr_vals = np.linspace(-0.9, 0.9, n_corrs)

if recalc:
    cmm = cMM('corr_rate.pkl')

    est_vs_true = np.zeros((n_means, n_corrs, n_reest, 3))

    for m in range(n_means):
        for c in range(n_corrs):
            mu_mp = mean_vals[m] * np.ones(n_unit)
            var_mp = 1 * np.ones(n_unit)
            corr_mp = np.array([[1, corr_vals[c]], [corr_vals[c], 1]])

            for i in range(n_reest):
                print(i)
                sc_corr, sc_mean, sc_var = cmm.generate(corr_mp, mu_mp, var_mp, n_trial, n_bin)
                mps_corr, mps_mean, mps_var = cmm.infer(sc_corr, sc_mean, sc_var, n_trial, n_bin, 100, 1000, 1)
                MAP_corr = histogramMode(mps_corr[:, 0, 1], 20)
                est_vs_true[m, c, i, :] = [corr_vals[c], MAP_corr, sc_corr[0, 1]]

    pickle.dump(est_vs_true, open('biastest.pkl', 'wb'))
else:
    est_vs_true = pickle.load(open('biastest.pkl', 'rb'))

for m in range(n_means):
    plt.subplot(1, n_means, m+1)
    true_corr = np.reshape(est_vs_true[m, :, :, 0], (n_corrs * n_reest))
    min_obs = np.zeros(n_corrs)
    max_obs = np.zeros(n_corrs)
    true_for_obs = np.zeros(n_corrs)
    for c in range(n_corrs):
        min_obs[c] = np.min(est_vs_true[m, c, :, 2])
        max_obs[c] = np.max(est_vs_true[m, c, :, 2])
        true_for_obs[c] = est_vs_true[m, c, 0, 0]
    # obs_corr = np.reshape(est_vs_true[m, :, :, 2], (n_corrs * n_reest))
    est_corr = np.reshape(est_vs_true[m, :, :, 1], (n_corrs * n_reest))
    plt.fill_between(true_for_obs, min_obs, max_obs, facecolor='grey', alpha=0.5)
    plt.scatter(true_corr, est_corr)
    # plt.scatter(true_corr, obs_corr, color='black', alpha=0.5)
    plt.xlim([-1, 1])
    plt.ylim([-1, 1])
    plt.plot([-1, 1], [-1, 1], color="black", linestyle='--', linewidth=1)
    plt.title("mean=%.2f" % mean_vals[m])
    if m == 0:
        plt.ylabel("posterior MAP of MP corr")
    if m == np.floor(n_means / 2) - 1:
        plt.xlabel("true MP corr")
plt.show()
