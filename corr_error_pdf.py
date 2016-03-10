import numpy as np
from numpy import zeros, floor, exp, corrcoef
from scipy import pi, sqrt
from scipy.special import hyp2f1, gamma
import matplotlib.pyplot as plt
import numpy.random as rnd


def corr_error_pdf(trueR, r, n):

    num_const = (n - 2) * gamma(n - 1)
    num_rho = (1 - trueR ** 2) ** ((n - 1) / 2)
    num_r = (1 - r ** 2) ** ((n - 4) / 2)
    den_const = sqrt(2 * pi) * gamma(n - 0.5)
    den_rhor = (1 - trueR * r) ** (n - 1.5)
    num_hyp = hyp2f1(0.5, 0.5, (2 * n - 1) / 2, (trueR * r + 1) / 2)

    return (num_const * num_rho * num_r * num_hyp) / (den_const * den_rhor)


def lognormal_corr(C):
    var1_n = C[0, 0]
    var2_n = C[1, 1]
    sd1_n = sqrt(var1_n)
    sd2_n = sqrt(var2_n)
    cov_n = C[0, 1]
    rho_n = cov_n / (sd1_n * sd2_n)

    return (exp(rho_n * sd1_n * sd2_n) - 1) / sqrt((exp(var1_n) - 1) * (exp(var2_n) - 1))


lognorm = True
nSamples = 10000
sampleSize = 100
covN = 0.9
C = np.array([[1, covN], [covN, 1]])
if lognorm:
    trueR = lognormal_corr(C)
else:
    trueR = C[0, 1] / sqrt(C[0, 0] * C[1, 1])
print(trueR)

corrVals = zeros((nSamples, 1))
for i in range(nSamples):
    actSamp = rnd.multivariate_normal([0, 0], C, sampleSize)
    if lognorm:
        actSamp = exp(actSamp)
    corrVals[i] = corrcoef(actSamp.T)[1, 0]

plt.hist(corrVals, 50, normed=1)

stepsize = 0.01
stepnum = int(floor(2 / stepsize))
pdf = zeros((stepnum, 1))
x = zeros((stepnum, 1))
for i in range(stepnum):
    x[i] = -1 + stepsize * i
    pdf[i] = corr_error_pdf(trueR, x[i], sampleSize)

plt.plot(x, pdf, color='g', linewidth=3)
plt.plot([trueR, trueR], [0, plt.gca().get_ylim()[1]], color='r', linestyle='-', linewidth=2)
plt.show()
