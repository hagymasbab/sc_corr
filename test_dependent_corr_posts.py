import numpy as np
import scipy.stats as st
import matplotlib.pyplot as plt
import csnltools as ct
import numpy.random as rnd

n_obs = 30
coupling_12 = 0.9
true_02 = 0.5
true_01 = 0.5
true_23 = 0.5
true_13 = 0.5
true_03 = 0.5
real_corr = np.array([[1, true_01, true_02, true_03], [true_01, 1, coupling_12, true_13], [true_02, coupling_12, 1, true_23], [true_03, true_13, true_23, 1]])
# observation = np.array([[1, 2, 3, 4]])
observation = rnd.multivariate_normal(np.zeros(4), real_corr, (n_obs))

res = 20
corr_vals = np.linspace(-0.9, 0.9, res)

like_landscape_indep = np.ones((res, res))
like_landscape_coupled = np.ones((res, res))
for i in range(res):
    for j in range(res):
        C_indep_1 = np.array([[1, corr_vals[i]], [corr_vals[i], 1]])
        # print C_indep_1
        C_indep_2 = np.array([[1, corr_vals[j]], [corr_vals[j], 1]])
        C_coupled_1 = np.array([[1, corr_vals[i], corr_vals[j]], [corr_vals[i], 1, coupling_12], [corr_vals[j], coupling_12, 1]])
        C_coupled_2 = 1
        for o in range(n_obs):
            # likelihood for the independent model
            # rint observation[o, 0:2]
            like_landscape_indep[i, j] *= st.multivariate_normal.pdf(observation[o, 0:2], np.zeros(2), C_indep_1)
            like_landscape_indep[i, j] *= st.multivariate_normal.pdf(observation[o, 2:4], np.zeros(2), C_indep_2)
            # likelihood for the coupled model
            # print observation[o, 0:3]
            # print np.zeros(3)
            # print C_coupled_1
            if np.all(np.linalg.eigvals(C_coupled_1) > 0):
                like_landscape_coupled[i, j] *= st.multivariate_normal.pdf(observation[o, 0:3], np.zeros(3), C_coupled_1)
                like_landscape_coupled[i, j] *= st.norm.pdf(observation[o, 3], 0, C_coupled_2)
            else:
                like_landscape_coupled[i, j] *= 0

# print like_landscape_indep
total_sum_indep = np.sum(np.sum(like_landscape_indep))
total_sum_coupled = np.sum(np.sum(like_landscape_coupled))
ent_indep = st.entropy(like_landscape_indep.flatten())
ent_coupled = st.entropy(like_landscape_coupled.flatten())
print ent_indep
print ent_coupled

ax1 = plt.subplot(2, 2, 1)
ct.heatmapPlot(ax1, like_landscape_indep, corr_vals, corr_vals, 'likelihood')
ax1 = plt.subplot(2, 2, 2)
ct.heatmapPlot(ax1, like_landscape_coupled, corr_vals, corr_vals, 'likelihood')
# corr_vals_dense = np.linspace(-0.9, 0.9, 200)
# plt.plot(res * corr_vals_dense, coupling_12 * corr_vals_dense, linewidth=2, color='red')
plt.subplot(2, 2, 3)
plt.plot(corr_vals, np.sum(like_landscape_indep, axis=1) / total_sum_indep, color='blue', linewidth=3)
plt.plot(corr_vals, np.sum(like_landscape_coupled, axis=1) / total_sum_coupled, color='red', linewidth=2)
plt.subplot(2, 2, 4)
plt.plot(corr_vals, np.sum(like_landscape_indep, axis=0) / total_sum_indep, color='blue', linewidth=3)
plt.plot(corr_vals, np.sum(like_landscape_coupled, axis=0) / total_sum_coupled, color='red', linewidth=2)
plt.show()
