import numpy as np
import numpy.random as rnd
from corrmeasure import correlationMeasurementModel as cMM
import pickle
import matplotlib.pyplot as plt


def histogramMode(data, resolution):
    binVals, binEdges = np.histogram(data, bins=resolution)
    maxIdx = binVals.argmax()
    return (binEdges[maxIdx] + binEdges[maxIdx+1]) / 2

recalc = False

n_unit = 2
n_pair = n_unit * (n_unit - 1) / 2
n_bin = 1
n_trial = 100
n_samp = 500

n_corrs = 10
n_means = 3
n_reest = 5

mean_vals = np.linspace(0.5, 3, n_means)
print(mean_vals)

corr_vals = np.linspace(-0.9, 0.9, n_corrs)

if recalc:
    cmm = cMM('corr_rate.pkl')

    est_vs_true = np.zeros((n_means, n_corrs, n_reest, 3))
    samples = np.zeros((n_means, n_corrs, n_reest, n_samp))

    for m in range(n_means):
        for c in range(n_corrs):
            mu_mp = mean_vals[m] * np.ones(n_unit)
            var_mp = 1 * np.ones(n_unit)
            corr_mp = np.array([[1, corr_vals[c]], [corr_vals[c], 1]])

            for i in range(n_reest):
                sc_corr, sc_mean, sc_var = cmm.generate(corr_mp, mu_mp, var_mp, n_trial, n_bin)
                mps_corr, mps_mean, mps_var = cmm.infer(sc_corr, sc_mean, sc_var, n_trial, n_bin, 100, n_samp, 2)
                samples[m, c, i, :] = mps_corr[:, 0, 1]
                MAP_corr = histogramMode(mps_corr[:, 0, 1], 20)
                est_vs_true[m, c, i, :] = [corr_vals[c], MAP_corr, sc_corr[0, 1]]

    pickle.dump((est_vs_true, samples), open('biastest.pkl', 'wb'))
else:
    loaded_data = pickle.load(open('biastest.pkl', 'rb'))
    est_vs_true = loaded_data[0]
    samples = loaded_data[1]


central_reest = np.floor(n_reest / 2)
for m in range(n_means):
    plt.subplot(1, n_means, m+1)
    true_corr = est_vs_true[m, :, :, 0]
    for c in range(n_corrs):
        for r in range(n_reest):
            true_corr[c, r] += (r - central_reest) * 0.02
    true_corr = np.reshape(true_corr, (n_corrs * n_reest))
    mean_est = np.zeros(n_corrs * n_reest)
    low_std = np.zeros(n_corrs * n_reest)
    high_std = np.zeros(n_corrs * n_reest)
    std = np.zeros(n_corrs * n_reest)
    min_obs = np.zeros(n_corrs)
    max_obs = np.zeros(n_corrs)
    true_for_obs = np.zeros(n_corrs)
    for c in range(n_corrs):
        min_obs[c] = np.min(est_vs_true[m, c, :, 2])
        max_obs[c] = np.max(est_vs_true[m, c, :, 2])
        true_for_obs[c] = est_vs_true[m, c, 0, 0]
        for r in range(n_reest):
            act_mean = np.mean(samples[m, c, r, :])
            mean_est[c * n_reest + r] = act_mean
            act_std = np.std(samples[m, c, r, :])
            act_map = est_vs_true[m, c, r, 1]
            low_std[c * n_reest + r] = act_map - (act_mean - act_std)
            high_std[c * n_reest + r] = (act_mean + act_std) - act_map
            std[c * n_reest + r] = act_std

    # obs_corr = np.reshape(est_vs_true[m, :, :, 2], (n_corrs * n_reest))
    est_corr = np.reshape(est_vs_true[m, :, :, 1], (n_corrs * n_reest))

    plt.fill_between(true_for_obs, min_obs, max_obs, facecolor='grey', alpha=0.5)
    # plt.scatter(true_corr, est_corr)
    plt.errorbar(true_corr, est_corr, yerr=[low_std, high_std], fmt='o', linewidth=1.5, markersize=7)
    # plt.errorbar(true_corr, mean_est, yerr=std, fmt='o')
    # plt.scatter(true_corr, obs_corr, color='black', alpha=0.5)
    plt.xlim([-1, 1])
    plt.ylim([-1, 1])
    plt.plot([-1, 1], [-1, 1], color="black", linestyle='--', linewidth=1)
    plt.title("MP mean=%.2f" % mean_vals[m])
    if m == 0:
        plt.ylabel("posterior MAP of MP corr")
    if m == np.floor(n_means / 2):
        plt.xlabel("true MP corr")
plt.show()
