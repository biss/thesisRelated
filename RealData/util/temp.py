
'''
import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
from cvxopt import matrix, solvers

def newModel(A, b, l):
        m, n = A.size
        lam = 50

        #with l-1 and l-2 regularizer

        #I = matrix(np.diag([1,1,1,1,1,1]))
        #print 'shape: ', (A.T*A).size, (A.T*A + I).size
        #Q = matrix([[A.T*A + I, -A.T*A - I],[-A.T*A - I, A.T*A + I]])

        Q = matrix(A.T*A)

        ones = matrix(1.0 * l, (n,1))
        aux = matrix(A.T*b)
        c = ones - aux
        N = n
        I = matrix(0.0, (N,N))
        I[::N+1] = 1.0
        G = matrix(-I)
        h = matrix(n*[0.0])
        return solvers.qp(Q, c, G, h)

data = pd.read_csv('kuiper_1.csv', sep=',')
number, features = data.shape
print data.columns.values
selfdata = np.array(data).T
X = matrix(selfdata[:features - 1, :].T)
y = matrix(selfdata[-1, :])

results = smf.ols('Price ~ Mileage + Cylinder + Liter + Doors + Cruise + Sound + Leather', data=data).fit()

print results.params
l = .5
sol = newModel(X, y, l)
res = np.array(sol['x'])

print res

'''

import numpy as np
b = np.array([[5,3,6,7,2],[10,11,3,4,9],[5,7,4,3,1]])
a = np.array([[1,0,1,0,1],[0,1,0,1,0]])
c = [[], []]

for i in range(5):
    for j in range(2):
        if a[j,i] == 1:
            c[j].append(i)
        #if a[j,i] == 0:
            #c[j].append(i)

print c

c_1 = np.take(b, c[0], axis = 1)
c_2 = np.take(b, c[1], axis = 1)

print type(c_1)
print c_2