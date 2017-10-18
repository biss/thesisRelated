__author__ = 'biswajeet'


import numpy as np
from l1regls import l1regls
from cvxopt import matrix, solvers
import pandas as pd # conventional alias
from sklearn.datasets import load_boston
import statsmodels.formula.api as smf

dataset = load_boston()
columns=dataset.feature_names
print columns
df = pd.DataFrame(dataset.data, columns=dataset.feature_names)
df['target'] = np.log(dataset.target)

instance_count, attr_count = df.shape

sLength = len(df['CRIM'])
df['intercept'] = pd.Series(np.ones(sLength), index=df.index)

regressor = df.drop(['target'], axis = 1)
target = np.log(df[['target']])

A = matrix(regressor.as_matrix())
b = matrix(target.as_matrix())

print A, b

'''
m, n = 200, 5
A, b = normal(m,n), normal(m,1)
A[:, 0] = 1
x = l1regls(A,b)
'''

def newModel(A, b):
    m, n = A.size
    lam = .5
    Q = matrix([[A.T*A, -A.T*A],[-A.T*A, A.T*A]])
    ones = matrix(1.0*lam, (2*n,1))
    aux = matrix([-A.T*b, A.T*b])
    c = ones + aux
    I = matrix(0.0, (2*n,2*n))
    I[::n+1] = 1.0
    G = matrix(-I)
    h = matrix(n*[0.0] + n*[0.0])
    return solvers.qp(Q, c, G, h)

print 'simple: '
sol = newModel(A, b)
new = np.array(sol['x'])
res = []
print 'new: ',new
for i in range(14):
    print 'hello: ',new[i,:], new[i+14, :]
    res.append(new[i,:] - new[i+14, :])
print(np.array(res))
print 'hallelujah'

#print 'l1regls: '
#print(l1regls(A, b))

print columns
lm = smf.ols(formula='target ~ CRIM + ZN + INDUS + CHAS + NOX + RM + AGE + DIS + RAD + TAX + PTRATIO '
                     '+ B + LSTAT', data=df).fit()
print lm.params