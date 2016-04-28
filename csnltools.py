import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as st


def histogramMode(data, resolution):
    binVals, binEdges = np.histogram(data, bins=resolution)
    maxIdx = binVals.argmax()
    return (binEdges[maxIdx] + binEdges[maxIdx + 1]) / 2


def heatmapPlot(ax, data, xtics, ytics, cbarlabel):
    heatmap = ax.pcolor(data, vmin=-1, vmax=1)
    # put the major ticks at the middle of each cell
    ax.set_xticks(np.arange(data.shape[0]) + 0.5, minor=False)
    ax.set_yticks(np.arange(data.shape[1]) + 0.5, minor=False)

    ax.set_xticklabels([str(x) for x in xtics], minor=False)
    ax.set_yticklabels([str(y) for y in ytics], minor=False)

    cbar = plt.colorbar(heatmap)
    cbar.set_label(cbarlabel)


def beta_params_from_moments(mu, sigma_square):
    if mu == 0:
        return 0, 0
    else:
        alpha = (((1 - mu) / sigma_square) - (1 / mu)) * mu ** 2
        beta = alpha * ((1 / mu) - 1)
        return alpha, beta


def covariance_matrix(var_vec, corr_mat):
    std_mat = np.sqrt(np.diag(var_vec))
    return std_mat.dot(corr_mat.dot(std_mat))


def correlation(x):
    if all(ix == 0 for ix in x[0, :]) or all(iy == 0 for iy in x[1, :]):
        return 0.0
    else:
        corr = np.corrcoef(x)[0, 1]
        if np.isnan(corr):
            return 0.0
        else:
            return corr


def pdf_mean(x, pdf):
    return np.sum(x * pdf)


def pdf_map(x, pdf):
    return x[np.argmax(pdf)]


def pdf_std(x, pdf):
    mu = pdf_mean(x, pdf)
    var = np.sum((x - mu) ** 2 * pdf)
    return np.sqrt(var)


def beta_over_correlations(x, shape, rate):
    return st.beta.pdf((x + 1) / 2, shape, rate) / 2
