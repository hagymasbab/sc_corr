import numpy as np
import matplotlib.pyplot as pl
from scipy.stats import norm


def eval_mixture(x, coeffs, means, stds):
    y = np.zeros(len(x))
    for i in range(len(x)):
        for k in range(len(coeffs)):
            y[i] += coeffs[k] * norm.pdf(x[i], means[k], stds[k])
    return y

firstMixtureCoeff = [0.1, 0.3, 0.7]
mixtureMean = [[-0.5, 0.5], [-0.1, 0.8], [-0.8, 0.1]]
mixtureStd = [[0.1, 0.1], [0.1, 0.1], [0.1, 0.1]]
colors = ["red", "green", "blue"]

x = np.linspace(-1, 1, 100)

for ii in range(3):
    y = eval_mixture(x, np.array([firstMixtureCoeff[ii], 1-firstMixtureCoeff[ii]]), mixtureMean[ii], mixtureStd[ii])
    pl.plot(x, y)
pl.show()
