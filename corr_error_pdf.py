import numpy as np
from numpy import zeros, floor, exp, corrcoef
from scipy import pi, sqrt
from scipy.io import loadmat
from scipy.special import hyp2f1, gamma
from scipy.stats import beta
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


def corr_error_mean(trueR, N):
    return trueR - (trueR * (1 - trueR ** 2)) / (2 * (N - 1))


def corr_error_var(trueR, N):
    # first-order approximation
    return (1 + 11 * trueR ** 2 / (2 * (N - 1))) * ((1 - trueR ** 2) ** 2) / (N - 1)


def beta_params_from_moments(mu, sigma):
    alpha = (((1 - mu) / sigma) - (1 / mu)) * mu ** 2
    beta = alpha * ((1 / mu) - 1)
    return [alpha, beta]


def lognormal_corr(C):
    var1_n = C[0, 0]
    var2_n = C[1, 1]
    sd1_n = sqrt(var1_n)
    sd2_n = sqrt(var2_n)
    cov_n = C[0, 1]
    rho_n = cov_n / (sd1_n * sd2_n)

    return (exp(rho_n * sd1_n * sd2_n) - 1) / sqrt((exp(var1_n) - 1) * (exp(var2_n) - 1))


nSamples = 1000
covN = 0.6
C = np.array([[1, covN], [covN, 1]])
trueR_N = C[0, 1] / sqrt(C[0, 0] * C[1, 1])

stepsize = 0.01
stepnum = int(floor(2 / stepsize))
pdf = zeros((stepnum, 1))
bpdf = zeros((stepnum, 1))
x = zeros((stepnum, 1))

# lsc = loadmat('/Users/karaj/csnl/majom/data/SC_nat_atoc100a01_bin10.mat')
# lsc = loadmat('/home/banmi/csnl/majom/data/SC_nat_atoc100a01_bin10.mat')
lsc = loadmat('../majom/data/SC_nat_atoc100a01_bin10.mat')
sc = np.sum(lsc['spikeCount'][5:7, :, 1:51], axis=2)
trueR_N = corrcoef(sc)[0, 1]

sampleSizes = [20, 50, 100]
nSampSize = len(sampleSizes)

for ns in range(nSampSize):
    sampleSize = sampleSizes[ns]
    corrVals_N = zeros((nSamples, 1))
    corrVals_lN = zeros((nSamples, 1))
    for i in range(nSamples):
        actSamp = sc[:, np.random.choice(sc.shape[1], sampleSize, replace=False)]
        corrVals_N[i] = corrcoef(actSamp)[1, 0]

    plt.subplot(100 + nSampSize * 10 + ns + 1)
    plt.hist(corrVals_N, 50, normed=1)

    corr_mean = corr_error_mean(trueR_N, sampleSize)
    corr_var = corr_error_var(trueR_N, sampleSize)
    print([corr_mean, corr_var])
    betaparams = beta_params_from_moments((corr_mean + 1) / 2, corr_var / 4)

    for i in range(stepnum):
        x[i] = -1 + stepsize * i
        pdf[i] = corr_error_pdf(trueR_N, x[i], sampleSize)
        bpdf[i] = beta.pdf((x[i] + 1) / 2, betaparams[0], betaparams[1]) / 2

    plt.plot(x, pdf, color='r', linewidth=3)
    plt.plot(x, bpdf, color='y', linewidth=3)
    # plt.plot([trueR_N, trueR_N], [0, plt.gca().get_ylim()[1]], color='r', linestyle='-', linewidth=2)
    plt.title('Sample size = %d' % sampleSize)

plt.show()
