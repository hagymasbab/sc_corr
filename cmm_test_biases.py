import numpy as np
import numpy.random as rnd
from corrmeasure import correlationMeasurementModel as cMM
import pickle
import matplotlib.pyplot as plt
from csnltools import histogramMode


recalc = True
plotContrast = False

n_unit = 2
n_pair = n_unit * (n_unit - 1) / 2
n_bin = 1
n_trial = 100
n_samp = 1000
n_chain = 2
n_step_per_chain = np.floor(2 * n_samp / n_chain)
n_samp = n_step_per_chain * n_chain / 2
n_noiselessTrial = 10000
n_samp_gen = 500

n_corrs = 2
n_means = 3
n_reest = 3

mean_vals = np.linspace(1, 3, n_means)
print(mean_vals)

corr_vals = np.linspace(-0.8, 0.8, n_corrs)
print(corr_vals)

if recalc:
    cmm = cMM('corr_rate.pkl')
    cmm.mp_corr_prior_conc = 2

    est_vs_true = np.zeros((n_means, n_corrs, n_reest, 4))
    samples = np.zeros((n_means, n_corrs, n_reest, n_samp))

    for m in range(n_means):
        for c in range(n_corrs):
            mu_mp = mean_vals[m] * np.ones(n_unit)
            var_mp = 1 * np.ones(n_unit)
            corr_mp = np.array([[1, corr_vals[c]], [corr_vals[c], 1]])

            sc_corr_noiseless, temp_mean, temp_var = cmm.generate(corr_mp, mu_mp, var_mp, n_noiselessTrial, n_bin)

            for i in range(n_reest):
                seed = (m+1)*(c+1)*(i+1)
                sc_corr, sc_mean, sc_var = cmm.generate(corr_mp, mu_mp, var_mp, n_trial, n_bin, seed)
                rnd.seed()
                seed = rnd.randint(10000)
                mps_corr, mps_mean, mps_var = cmm.infer(sc_corr, sc_mean, sc_var, n_trial, n_bin, n_samp_gen, n_step_per_chain, n_chain, seed)
                samples[m, c, i, :] = mps_corr[:, 0, 1]
                MAP_corr = histogramMode(mps_corr[:, 0, 1], 20)
                est_vs_true[m, c, i, :] = [corr_vals[c], MAP_corr, sc_corr[0, 1], sc_corr_noiseless[0, 1]]

    pickle.dump((est_vs_true, samples), open('biastest.pkl', 'wb'))
else:
    loaded_data = pickle.load(open('biastest.pkl', 'rb'))
    est_vs_true = loaded_data[0]
    samples = loaded_data[1]


central_reest = np.floor(n_reest / 2)
if plotContrast:
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
else:
    for c in range(n_corrs):
        plt.subplot(1, n_corrs, c+1)
        true_mean = np.zeros(n_means * n_reest)
        all_map = np.zeros(n_means * n_reest)
        low_std = np.zeros(n_means * n_reest)
        high_std = np.zeros(n_means * n_reest)
        means_for_noiseless = np.zeros(n_means * 2)
        noiseless_corr = np.zeros(n_means * 2)
        for m in range(n_means):
            means_for_noiseless[m*2] = mean_vals[m] - (central_reest+1) * 0.05
            means_for_noiseless[m*2+1] = mean_vals[m] + (n_reest - central_reest) * 0.05
            noiseless_corr[m*2] = est_vs_true[m, c, 1, 3]
            noiseless_corr[m*2+1] = est_vs_true[m, c, 1, 3]
            reest_obs = est_vs_true[m, c, :, 2]
            reest_mean = np.zeros(n_reest)
            sortindices = np.argsort(reest_obs)
            for r in range(n_reest):
                r_idx = sortindices[r]
                true_mean[m * n_reest + r] = mean_vals[m] + (r - central_reest) * 0.05
                reest_mean[r] = true_mean[m * n_reest + r]
                act_mean = np.mean(samples[m, c, r_idx, :])
                act_std = np.std(samples[m, c, r_idx, :])
                act_map = est_vs_true[m, c, r_idx, 1]
                low_std[m * n_reest + r] = act_map - (act_mean - act_std)
                high_std[m * n_reest + r] = (act_mean + act_std) - act_map
                all_map[m * n_reest + r] = act_map
            srtd_obs = np.sort(reest_obs)
            invcrv = - (srtd_obs - corr_vals[c]) + corr_vals[c]
            plt.plot(reest_mean, srtd_obs, color='red', linewidth=2)
            plt.fill_between(reest_mean, np.minimum(srtd_obs, invcrv), np.maximum(srtd_obs, invcrv), facecolor='grey', alpha=0.3)

        plt.plot(means_for_noiseless, noiseless_corr, color='green', linewidth=2)
        plt.errorbar(true_mean, all_map, yerr=[low_std, high_std], fmt='o', linewidth=1.5, markersize=7)
        plt.plot(plt.xlim(), corr_vals[c] * np.ones(2), color="black", linestyle='--', linewidth=1)
        plt.title("MP corr=%.2f" % corr_vals[c])
        if m == 0:
            plt.ylabel("posterior MAP of MP corr")
        if m == np.floor(n_means / 2):
            plt.xlabel("MP mean")
plt.show()
