from corrmeasure import correlationMeasurementModel as cMM
import numpy as np
import matplotlib.pyplot as pl

np.seterr(all='raise')
cmm = cMM('corr_rate.pkl')

res = 100
x = np.linspace(-1, 1, res)
post = cmm.pairwise_corr_numerical_posterior(100, -0.5, 100, 100, res)
print post
pl.plot(x, post)
pl.show()
