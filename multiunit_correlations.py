import numpy as np
import numpy.random as rn
import matplotlib.pyplot as pl
import scipy.stats as st


def corrvals2mat(corrvals, dim):
    cm = np.zeros((dim, dim))
    cm[np.triu_indices(dim, 1)] = corrvals
    cm += cm.T + np.identity(dim)
    return cm


def eval_mixture(x, coeffs, means, stds):
    y = np.zeros(len(x))
    for i in range(len(x)):
        for k in range(len(coeffs)):
            y[i] += coeffs[k] * st.norm.pdf(x[i], means[k], stds[k])
    return y


def samp_mixture(n_samp, coeffs, means, stds):
    samps = np.zeros(n_samp)
    for i in range(n_samp):
        act_comp = np.nonzero(rn.multinomial(1, coeffs))[0]
        samps[i] = rn.normal(means[act_comp], stds[act_comp])
    return samps


firstMixtureCoeff = [1, 1, 1]
mixtureMean = [[0.1, -0.5], [0.1, -0.5], [0.01, 0.1]]
mixtureStd = [[0.01, 0.01], [0.01, 0.01], [0.001, 0.001]]
colors = ["red", "green", "blue"]

x = np.linspace(-1, 1, 1000)

pl.subplot(1, 2, 1)
for ii in range(3):
    y = eval_mixture(x, np.array([firstMixtureCoeff[ii], 1-firstMixtureCoeff[ii]]), mixtureMean[ii], mixtureStd[ii])
    pl.plot(x, y)

unitNums = [2, 3, 5, 8, 10, 20, 30, 50]
# unitNums = [2, 3]
n_corr_group = unitNums[-1] * (unitNums[-1] - 1) / 2
n_corr_inter = unitNums[-1] ** 2
group_corrs = np.zeros((2, n_corr_group))
inter_corrs = np.zeros(n_corr_inter)
for ii in range(2):
    group_corrs[ii, :] = samp_mixture(n_corr_group, np.array([firstMixtureCoeff[ii], 1-firstMixtureCoeff[ii]]), mixtureMean[ii], mixtureStd[ii])
inter_corrs = samp_mixture(n_corr_inter, np.array([firstMixtureCoeff[2], 1-firstMixtureCoeff[2]]), mixtureMean[2], mixtureStd[2])
# TODO set values between -1 and 1

n_reest = 10

allCorrs = np.zeros((n_reest, len(unitNums)))
for r in range(n_reest):
    summedCorrs = []
    for un in unitNums:
        act_cnum = un * (un - 1) / 2
        c1 = corrvals2mat(group_corrs[0, 0:act_cnum], un)
        c2 = corrvals2mat(group_corrs[1, 0:act_cnum], un)
        cint = np.reshape(inter_corrs[0:un ** 2], (un, un))
        c = np.zeros((2*un, 2*un))
        c[0:un, 0:un] = np.triu(c1)
        c[un:, un:] = np.triu(c2)
        c[0:un, un:] = cint
        c += c.T - np.identity(2*un)
        # TODO heatmap plot of c
        samples = rn.multivariate_normal(np.zeros(2*un), c, 10000)
        group1 = np.sum(samples[:, 0:un], axis=1)
        group2 = np.sum(samples[:, un:], axis=1)
        summedCorrs.append(np.corrcoef(group1, group2)[0, 1])
    allCorrs[r, :] = summedCorrs

pl.subplot(1, 2, 2)
pl.errorbar(unitNums, np.mean(allCorrs, axis=0), yerr=st.sem(allCorrs, axis=0))

pl.show()
