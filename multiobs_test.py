from pystan import StanModel
import pickle
from corrmeasure import correlationMeasurementModel as cMM


# sm = StanModel(file='corrmeasure.stan', verbose=False)
# with open('corrmeasure.pkl', 'wb') as f:
#     pickle.dump(sm, f)

cmm = cMM('corrmeasure.pkl')
sc_corr, sc_mean, sc_var = cmm.quick_sample()
mps_corr, mps_mean, mps_var = cmm.infer(sc_corr, sc_mean, sc_var, 100, 1, 100, 1000)
print(mps_corr)