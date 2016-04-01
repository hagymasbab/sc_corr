import numpy as np
import matplotlib.pyplot as plt


def histogramMode(data, resolution):
    binVals, binEdges = np.histogram(data, bins=resolution)
    maxIdx = binVals.argmax()
    return (binEdges[maxIdx] + binEdges[maxIdx+1]) / 2


def heatmapPlot(ax, data, xtics, ytics):
    heatmap = ax.pcolor(data, vmin=-1, vmax=1)
    # put the major ticks at the middle of each cell
    ax.set_xticks(np.arange(data.shape[0])+0.5, minor=False)
    ax.set_yticks(np.arange(data.shape[1])+0.5, minor=False)

    ax.set_xticklabels([str(x) for x in xtics], minor=False)
    ax.set_yticklabels([str(y) for y in ytics], minor=False)

    plt.colorbar(heatmap)
