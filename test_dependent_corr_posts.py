import numpy as np
import scipy.stats as st
import matplotlib.pyplot as plt
import csnltools as ct

observation = np.array([[1, 2, 3, 4]])
coupling = 0.2

res = 100
corr_vals = np.linspace(-0.9, 0.9, res)

like_landscape_indep = np.ones((res, res))
like_landscape_coupled = np.ones((res, res))
for i in range(res):
    for j in range(res):
        C_indep_1 = np.array([[1, corr_vals[i]], [corr_vals[i], 1]])
        # print C_indep_1
        C_indep_2 = np.array([[1, corr_vals[j]], [corr_vals[j], 1]])
        C_coupled_1 = np.array([[1, corr_vals[i], corr_vals[j]], [corr_vals[i], 1, coupling], [corr_vals[j], coupling, 1]])
        C_coupled_2 = 1
        for o in range(observation.shape[0]):
            # likelihood for the independent model           
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

print like_landscape_indep

ax1 = plt.subplot(1, 2, 1)
ct.heatmapPlot(ax1, like_landscape_indep, corr_vals, corr_vals, 'likelihood')
ax1 = plt.subplot(1, 2, 2)
ct.heatmapPlot(ax1, like_landscape_coupled, corr_vals, corr_vals, 'likelihood')
plt.show()
