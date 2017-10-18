

import numpy as np
import statsmodels.api as sm

data = np.genfromtxt("../input/out_n.csv", delimiter=',')

X = data[:-1, :].T
Y = data[-1, :].T

print X.shape, Y.shape

res = sm.OLS(Y, X).fit()

print res.summary()