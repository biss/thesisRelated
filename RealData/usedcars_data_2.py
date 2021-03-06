
__author__ = 'biswajeet'

''' fitting FCR model with l1 regularizer on used car data. All the features are normalised.
    writes file with a comparable view'''

import numpy as np
import math, csv, pandas as pd
from numpy import linalg as LA
import copy
from cvxopt import matrix, solvers
from sklearn.metrics import mean_squared_error as mse

class SwitchingRegression:

    #total number of data points
    number = 0
    obj = 0
    abs_v = np.zeros(shape = (1,2))
    features = 0
    c = 2
    error = 0

    #each column of data is a data point,[x1, x2, x3, x4, y].T is a column
    data = np.zeros(shape = (features, number), dtype=float)

    '''initialized parameters of the two regimes. For eg in (y = a*x1 + b*x2 + c*x3 + d*x4 + e*x_5 + f*x_6)
       --> 1st component is a, 2nd is b, 3rd is c... i.e. [a, b, c, d, e, f]'''
    regime_param = []

    #membership matrix initialised uniformly
    U = np.random.uniform(0.0, 1.0, [c, number])
    rms = 0
    m = 2
    epsilon = 0.0000005
    E = np.random.uniform(0.0, 1.0, [c, number])
    delta = 0
    training_iter = 0
    #supress print messages
    solvers.options['show_progress'] = False

    def newModel(self, A, b, l):
        m, n = A.size
        lam = 50

        #with l-1 and l-2 regularizer
        '''
        I = matrix(np.diag([1,1,1,1,1,1]))
        #print 'shape: ', (A.T*A).size, (A.T*A + I).size
        Q = matrix([[A.T*A + I, -A.T*A - I],[-A.T*A - I, A.T*A + I]])
        '''

        Q = matrix([[A.T*A, -A.T*A],[-A.T*A, A.T*A]])

        ones = matrix(1.0*l, (2*n,1))
        aux = matrix([-A.T*b, A.T*b])
        c = ones + aux
        N = 2*n
        I = matrix(0.0, (N,N))
        I[::N+1] = 1.0
        G = matrix(-I)
        h = matrix(n*[0.0] + n*[0.0])
        return solvers.qp(Q, c, G, h)

    def load_data(self):
        data = pd.read_csv('input/kuiper_ip.csv', sep=',')
        self.number, self.features = data.shape
        #print self.number, self.features
        data = data[['Mileage', 'Cylinder', 'Liter', 'Cruise', 'Sound', 'Leather', 'Price']]
        self.data = np.array(data).T

    def init(self, cluster):
        self.regime_param = [[0.0 for i in range(self.features-1)], [0.0 for i in range(self.features-1)]]
        self.c = cluster
        self.U = np.random.uniform(0.0, 1.0, [self.c, self.number])
        self.E = np.random.uniform(0.0, 1.0, [self.c, self.number])

    def crispify(self):
        return np.round(self.U, 0)

    def set_from_array(self, array):
        c = [[], []]
        cl, n = array.shape
        for i in range(n):
            for j in range(cl):
                if array[j, i] == 1:
                    c[j].append(i)

        return c[0], c[1]

    def update_param(self, l):
        #1
        print
        print 'in param_update:'

        X = self.data[:self.features - 1, :].T
        Y = self.data[-1, :]

        root_U = np.sqrt(self.U)

        ''' use the  '''
        obj = 0
        abs_val = np.zeros(shape=(1,2))
        for i in range(self.c):

            D = np.diag(root_U[i, :])

            ''' construct the new A and y matrix'''
            A = matrix(np.dot(D, X))
            y = matrix(np.dot(D, Y))

            sol = self.newModel(A, y, l)
            new = np.array(sol['x'])
            res = []

            for j in range(self.features - 1):
                #print 'hello: ',new[j,:], new[j+7, :]
                res.append(new[j,:] - new[j + self.features - 1, :])
                #print res
            self.regime_param[i] = np.array(res)

            print 'l-1 norm value of ',i+1, 'th model: ', sum(np.absolute(self.regime_param[i]))
            abs_val[:, i] = sum(np.absolute(self.regime_param[i]))
            obj_value = np.dot(A, self.regime_param[i]) - y
            obj += np.dot(obj_value.T, obj_value)

            print 'objective function value: ', obj

        self.obj = obj
        self.abs_v = abs_val
        print abs_val


    def train(self, l):

        U_old = copy.copy(self.U)
        self.update_param(l)
        #1  print 'U before updation:', self.U

        self.update_membership()
        #1  print 'U after updation:', self.U
        error = LA.norm(U_old - self.U)
        #print '--------------->',error
        print 'Em:',self.Em()

        self.training_iter = 0
        while error > self.epsilon:
            #1  print 'U before updation:', self.U
            U_old = copy.copy(self.U)
            #1  print "error: ", self.E
            #1  print "U: ", self.U

            self.update_param(l)
            #1  print 'after update param'

            self.update_membership()
            #1  print 'after update membership'

            #this error depends on the membership values
            error = LA.norm(U_old - self.U)
            print 'error------>', error
            print 'Em:', self.Em()

            self.training_iter += 1
        #1  print 'U after updation:', self.U
        self.error = error
        print "error after completing training: ", error

        '''RMSE calculation'''
        crisp_U = self.crispify()

        ''' get the indices of the samples which belong to
            cluster 1 and cluster 2 individually depending
            on the crisp membership matrix'''
        clus_1, clus_2 = self.set_from_array(crisp_U)
        print clus_1, clus_2
        data_1 = np.take(self.data, clus_1, axis=1)
        data_2 = np.take(self.data, clus_2, axis=1)
        X_1 = data_1[1:, :].T
        Y_1 = data_1[0, :].reshape(data_1.shape[1], 1)
        X_2 = data_2[1:, :].T
        Y_2 = data_2[0, :].reshape(data_2.shape[1], 1)

        #rmse = np.zeros(shape=(1, 2))
        rms_1 = math.sqrt(mse(np.dot(X_1, self.regime_param[0]), Y_1) +\
            mse(np.dot(X_2, self.regime_param[0]), Y_2))
        rms_2 = math.sqrt(mse(np.dot(X_2, self.regime_param[0]), Y_2)+ \
            mse(np.dot(X_1, self.regime_param[0]), Y_1))
        if (rms_1 > rms_2):
            self.rms = rms_2
        else:
            self.rms = rms_1
        self.U = crisp_U

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
            #print "indep.shape, self.regime_param[regime - 1].shape: ",indep.shape, self.regime_param
            #1  print "variables one: ", dep, indep
            #1  print "calculated value E :", math.pow(dep - (np.dot(self.regime1_param.T, indep) + self.delta), 2)

            return math.pow(dep - (np.dot(self.regime_param[regime-1].T, indep) + self.delta), 2)
        if regime == 2:
            #print "indep.shape, self.regime_param[regime - 1].shape: ",indep.shape, self.regime_param[regime - 1].shape
            #1  print "variables two: ", dep, indep
            #1  print "calculated value E :", math.pow(dep - (np.dot(self.regime2_param.T, indep) + self.delta), 2)
            return math.pow(dep - (np.dot(self.regime_param[regime-1].T, indep)  + self.delta), 2)
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
    obj = []
    abs_va = []
    res = {}
    no_of_iter = {}
    lmd = [.2]

    #lmd = [0.1*i for i in range(10)]
    #lmd = lmd + [1.0*i for i in range(2,10)]

    for element in lmd:

        sr = SwitchingRegression()
        sr.load_data()
        sr.init(clusters)
        #print sr.regime_param[0].shape
        sr.train(element)

        #specifying limits
        #print 'betas for ',element,' are: ', sr.regime_param
        #print sr.training_iter
        no_of_iter[element] = sr.training_iter
        res[element] = [sr.regime_param, np.sqrt(sr.obj/sr.number), sr.obj, sr.abs_v]
        obj.append(sr.obj)
        abs_va.append(sr.abs_v)
    print res


    #with open('result_0.1_10_1.csv', 'wb') as file:
    with open('output/junk.csv', 'wb') as file:
        writer = csv.writer(file, delimiter='\t')
        writer.writerow(res.keys())
        writer.writerow('-')
        for row in zip(*res.values()):
            for row1 in zip(*row):
                for row2 in zip(*row1):
                    writer.writerow(list(row2))
                writer.writerow('-')
        #writer.writerow(list(obj))
        #writer.writerow(list(abs_va))
    print obj

    '''sr = SwitchingRegression()

    sr.load_data()

    sr.train()

    #specifying limits
    print 'betas are: ', sr.regime_param
    print sr.training_iter'''