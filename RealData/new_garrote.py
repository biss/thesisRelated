
__author__ = 'biswajeet'

''' garrote estimate on usedcar data. All the features are normalised
    code change: the estimate containers were changed to numpy array'''

import numpy as np
import math, csv, pandas as pd
from numpy import linalg as LA
import copy
import statsmodels.formula.api as smf
from cvxopt import matrix, solvers

class SwitchingRegression:

    #total number of data points
    number = 0
    features = 0
    c = 2

    obj = 0
    abs_v = np.zeros(shape = (1,2))

    #each column of data is a data point,[x1, x2, x3, x4, y].T is a column
    data = np.zeros(shape = (features, number), dtype=float)

    '''initialized parameters of the two regimes. For eg in (y = b*x1 + c*x2 + d*x3 + e*x4)
       --> 1st component is a, 2nd is b, 3rd is c... i.e. [a, b, c, d, e]'''
    betas = np.zeros(shape = (c, features), dtype=float)
    regime_param = np.zeros(shape = (c, features), dtype=float)
    belittle = np.zeros(shape = (c, features), dtype=float)

    #membership matrix initialised uniformly
    U = np.random.uniform(0.0, 1.0, [c, number])

    m = 1.2
    epsilon = 0.0000005
    E = np.random.uniform(0.0, 1.0, [c, number])
    delta = 0
    training_iter = 0

    #supress print messages
    solvers.options['show_progress'] = False

    ''' estimates the parameter of the model using the new cost function
        and quadratic programming'''
    def newModel(self, A, b, l):
        m, n = A.size
        lam = 50

        #with l-1 and l-2 regularizer
        '''
        I = matrix(np.diag([1,1,1,1,1,1]))
        #print 'shape: ', (A.T*A).size, (A.T*A + I).size
        Q = matrix([[A.T*A + I, -A.T*A - I],[-A.T*A - I, A.T*A + I]])
        '''

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

    def load_data(self):

        #mean centered and unit std dev data
        data = pd.read_csv('input/kuiper_1.csv', sep=',')
        self.number, self.features = data.shape
        print self.number, self.features
        #print data

        '''self.regime_param = [np.zeros(shape=(1,self.features - 1)), np.zeros(shape=(1,self.features - 1))]'''
        self.regime_param = np.zeros(shape = (self.c, self.features - 1), dtype=float)
        self.betas = np.zeros(shape = (self.c, self.features - 1), dtype=float)
        self.belittle = np.zeros(shape = (self.c, self.features - 1), dtype=float)
        self.data = np.zeros(shape = (self.features, self.number), dtype=float)

        '''results = smf.ols('Price ~ Mileage + Cylinder + Liter + Doors + Cruise + Sound + Leather', data=data).fit()

        for i in range(self.c):
            self.betas[i, :] = np.array(results.params[1:])'''
        self.betas[0, :] = np.array([[-0.12032018, -0.65328469, 0.26691265, -0.17563946, 0.28700011, 0.00153014, 0.12933018]])
        self.betas[1, :] = np.array([[ -0.163352959, 1.26497273, -0.253439970, -0.114382012, -0.000856110065,
                                       0.0788672084, 0.0901723782]])

        self.data = np.array(data).T
        #print self.data

    def init(self, cluster):
        self.c = cluster
        self.load_data()
        self.U = np.random.uniform(0.0, 1.0, [self.c, self.number])
        self.E = np.random.uniform(0.0, 1.0, [self.c, self.number])

    def update_belittle(self, l):
        #1
        print
        print 'Iternation :', self.training_iter

        X = self.data[1:, :].T
        Y = self.data[0, :].reshape(self.number, 1)

        #print 'Y', Y.shape

        root_U = np.power(self.U, self.m/2)

        obj = 0
        abs_val = np.zeros(shape=(1,2))

        ''' use the  '''
        for i in range(self.c):
            print 'Model =', i

            D_sqr_U = np.diag(root_U[i, :])
            D_beta = np.diag(list(self.betas[i, :]))

            #print 'beta diag ', D_beta
            #print 'beta diag ', list(self.betas[i, :])

            ''' construct the new A and y matrix. y = diag(sqrt(U));
                A = diag(sqrt(U)) (X) diag(sqrt(beta))'''

            #print 'imp ', D_sqr_U.shape, X.shape, D_beta.shape

            A = matrix(D_sqr_U.dot(X).dot(D_beta))
            y = matrix(np.dot(D_sqr_U, Y))

            sol = self.newModel(A, y, l)
            res = np.array(sol['x'])

            self.belittle[i] = res.T

            print 'c: ',i,"; ",self.belittle[i]
            print 'beta: ',i,"; ",self.betas[i]

            self.regime_param[i] = self.belittle[i] * self.betas[i]

            print 'regime_param: ',i,"; ",self.regime_param[i]

            print 'l-1 norm value of ',i+1, 'th model: ', np.sum(np.absolute(self.regime_param[i]))

            #print 'hello ', A.size, self.regime_param[i].shape, y.size

            abs_val[:, i] = np.sum(np.absolute(self.regime_param[i]))
            obj_value = np.dot(A, self.regime_param[i].reshape(self.features - 1, 1)) - y
            obj += np.dot(obj_value.T, obj_value)

            print 'cumulative objective function value: ', obj

        self.obj = obj
        self.abs_v = abs_val
        #print abs_val


    def train(self, l):

        U_old = copy.copy(self.U)
        self.update_belittle(l)
        #1  print 'U before updation:', self.U

        self.update_membership()
        #1  print 'U after updation:', self.U
        error = LA.norm(U_old - self.U)
        #print '--------------->',error
        print 'Em:',self.Em()

        self.training_iter = 0
        while error > self.epsilon and self.training_iter < 1000:
            #1  print 'U before updation:', self.U
            U_old = copy.copy(self.U)
            #1  print "error: ", self.E
            #1  print "U: ", self.U

            self.update_belittle(l)
            #1  print 'after update param'

            self.update_membership()
            #1  print 'after update membership'

            #this error depends on the membership values
            error = LA.norm(U_old - self.U)
            print 'error------>', error
            print 'Em:', self.Em()

            self.training_iter += 1
        #1  print 'U after updation:', self.U
        print "error after completing training: ", error

    def update_membership(self):
        self.construct_error_matrix()

        #1  print 'error terms:', self.E

        for k in range(self.number):
            if all(self.E[:, k] > 0) == True:
                #1  print 'all error terms are non zero.'
                den = np.zeros(shape=(self.c, 1))
                for i in range(2):
                    den[i, 0] = float(1/self.calc_denominator(i, k))

                #1  print "updated value", den[:, 0]
                self.U[:, k] = den[:, 0]

            else:
                #1  print 'some error terms are zero.'
                for i in range(self.c):
                    if self.E[i, k] > 0:
                        self.U[i, k] = 0.0
                    else:
                        if sum (x > 0 for x in self.E[:, k]) > 0:
                            self.U[i, k] = float(1 / (sum (x > 0 for x in self.E[:, k])))

    def calc_denominator(self, i, k):
        value = 0.0
        for j in range(self.c):
            value = value + math.pow(self.E[i, k]/self.E[j, k], 1/(self.m - 1))
        return value

    def construct_error_matrix(self):
        for i in range(self.c):
            for j in range(self.number):

                self.E[i, j] = self.calculate_error(i+1, self.data[-1, j], self.data[:-1, j])

    def calculate_error(self, regime, dep, indep):

        if regime == 1:
            #print "indep.shape, self.regime_param[regime - 1].shape1: ",indep.shape, self.regime_param[regime - 1].shape

            return np.square(dep - (np.dot(self.regime_param[regime-1], indep.reshape(self.features - 1)) + self.delta))
        if regime == 2:
            #print "indep.shape, self.regime_param[regime - 1].shape: ",indep.shape, self.regime_param[regime - 1].shape

            return np.square(dep - (np.dot(self.regime_param[regime-1], indep.reshape(self.features - 1))  + self.delta))
        else:
            print "invalid regime"
            return 0

    def Em(self):
        error = 0
        for k in range(self.number):
            for i in range(self.c):
                error += pow(self.U[i, k], self.m) * self.E[i, k]
        return error

if __name__ == '__main__':

    clusters = 2
    '''
    sr = SwitchingRegression()
    sr.load_data()
    sr.init(clusters)
    print sr.features, sr.number
    print sr.data[:, 1]
    print sr.regime_param
    '''
    obj = []
    abs_va = []
    res = {}
    no_of_iter = {}
    lmd = [0.5, 0.8]
    for element in lmd:

        sr = SwitchingRegression()
        sr.init(clusters)
        #print sr.regime_param[0].shape
        sr.train(element)

        #specifying limits
        #print 'betas for ',element,' are: ', sr.regime_param
        #print sr.training_iter
        no_of_iter[element] = sr.training_iter
        res[element] = sr.regime_param

        res[element] = sr.regime_param
        obj.append(sr.obj)
        abs_va.append(sr.abs_v)

    print res

    with open('output/result_garrote.csv', 'wb') as file:
        writer = csv.writer(file, delimiter='\t')
        writer.writerow(res.keys())
        for row in zip(*res.values()):
            for row1 in zip(*row):
                writer.writerow(list(row1))
            writer.writerow('-')
        writer.writerow(obj)
        writer.writerow(abs_va)

    '''sr = SwitchingRegression()

    sr.load_data()

    sr.train()

    #specifying limits
    print 'betas are: ', sr.regime_param
    print sr.training_iter'''