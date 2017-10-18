
__author__ = 'biswajeet'
__date__ = '24-03-2015'

''' garrotte with single scaling factors
    garrote estimate on usedcar data. All the features are normalised
    code change: the estimater holders were changed to numpy array.
    Writes file with a comparision view. The betas were initialised with
    the lasso estimates'''

import numpy as np
import math, csv, pandas as pd
from numpy import linalg as LA
import copy
from cvxopt import matrix, solvers
from sklearn.metrics import mean_squared_error as mse
from scipy import stats

''' class implementing all the necessary variables and methods'''
class SwitchingRegression_g:

    def __init__(self, cluster=2):

        self.c = cluster

        ''' variable to store the value of the objective function after each iteration.
        This objective value is the sum of objective value of all the c models. '''
        self.obj = 0

        '''rms error'''
        self.rms = 0

        ''' each column of data is a features-dimensional data sample.
            Example: [x1, x2, x3, x4, y].T is a column.
            So, the data matrix consists of 'number' number of columns'''
        # mean centered and unit std dev data
        data = np.genfromtxt('input/kuiper.csv', dtype=float, delimiter=',', skip_header=1)

        #data = data / data.max(axis=0)
        data = stats.zscore(data, axis=0)
        self.number, self.features = data.shape
        print self.number, self.features
        self.data = data.T
        print self.data

        ''' stores the sum of absolute values of the models'''
        self.abs_v = np.zeros(shape = (1,self.c))

        self.betas = np.zeros(shape = (self.c, self.features - 1), dtype=float)
        self.regime_param = np.zeros(shape = (self.c, self.features - 1), dtype=float)
        self.belittle = np.zeros(shape = (1, self.features - 1), dtype=float)

        '''membership matrix initialised uniformly'''
        self.U = np.random.uniform(0.0, 1.0, [self.c, self.number])

        self.m = 2

        ''' for terminating condition'''
        self.epsilon = 0.0000005

        '''error matrix initialise'''
        self.E = np.random.uniform(0.0, 1.0, [self.c, self.number])

        ''' for invertibility issue.'''
        self.delta = 0

        '''stores the number of training iterations required to
            reach the precision of epsilon'''
        self.training_iter = 0

        self.err = 0

        self.betas[0, :] = np.array([[-0.079, 0.336, 0.110, -0.013, 0.002, 0.012]])
        self.betas[1, :] = np.array([[-0.173, 0.488, -0.115, 0.278, 0.010, 0.004]])

        #supress print messages
        solvers.options['show_progress'] = False

    ''' estimates the parameter of the model using the new cost function
        and quadratic programming'''

    def newModel(self, A, b, l):
        A_1 = matrix(A[0])
        A_2 = matrix(A[1])

        b_1 = matrix(b[0])
        b_2 = matrix(b[1])
        m, n = A_1.size

        print 'sizes ', A_1.size, b_1.size

        if l == 0.0:
            I = matrix(0.0, (n, n))
            I[::n + 1] = 2.0
            Q = I * (A_1.T * A_1 + A_2.T * A_2)
            # print 'Q :', Q

            I[::n + 1] = 2.0
            p = I * A_1.T * b_1 + I * A_2.T * b_2
            # I[::col + 1] = 1.0
            # G = matrix(-I)
            # h = matrix(col * [0.0])
            return solvers.qp(Q, -p)

        else:
            I = matrix(0.0, (n, n))
            I[::n + 1] = 2.0
            Q = I * (A_1.T * A_1 + A_2.T * A_2)

            ones = matrix(1.0 * l, (n, 1))
            aux = I * A_1.T * b_1 + I * A_2.T * b_2
            c = ones - aux
            N = n
            I = matrix(0.0, (N, N))
            I[::N + 1] = 1.0
            G = matrix(-I)
            h = matrix(n * [0.0])
            return solvers.qp(Q, c, G, h)

    ''' load the usedcars dataset which is already mean centered and unit std dev,
        initialise the parameters of the model. The beta parameter is initialised
        with the lasso estimates'''

    def clusters(self):
        return np.round(self.U, 0)

    def set_from_array(self, array):
        c = [[], []]
        cl, n = array.shape
        for i in range(n):
            for j in range(cl):
                if array[j, i] == 1:
                    c[j].append(i)

        return c[0], c[1]

    def update_belittle(self, l):

        print

        X = self.data[:-1, :].T
        print 'hello :', self.data[:-1, :].shape, self.data[-1, :].shape
        Y = self.data[-1, :].reshape(self.number, 1)

        # print 'Y', Y.shape

        root_U = np.power(self.U, self.m / 2)

        A = np.array(
            [np.zeros(shape=(self.number, self.features - 1)), np.zeros(shape=(self.number, self.features - 1))])
        # A_2 = matrix(np.zeros(shape=(self.number, self.features - 1)))
        y = np.array([np.zeros(shape=(self.number, 1)), np.zeros(shape=(self.number, 1))])

        obj = 0
        abs_val = np.zeros(shape=(1, 2))

        ''' use the '''
        for i in range(self.c):
            D_sqr_U = np.diag(root_U[i, :])
            D_beta = np.diag(list(self.betas[i, :]))

            # print 'beta diag ', D_beta
            # print 'beta diag ', list(self.betas[i, :])

            ''' construct the new A and y matrix. y = diag(sqrt(U));
                A = diag(sqrt(U)) (X) diag(sqrt(beta))'''

            # print 'imp ', D_sqr_U.shape, X.shape, D_beta.shape

            A[i] = D_sqr_U.dot(X).dot(D_beta)
            y[i] = np.dot(D_sqr_U, Y)

        sol = self.newModel(A, y, l)
        res = np.array(sol['x'])

        self.belittle[0] = res.T

        for j in range(self.c):
            # print 'c: ', self.belittle[0]
            # print 'beta: ', j,"; ",self.betas[j]

            self.regime_param[j] = self.belittle[0] * self.betas[j]

            print 'regime_param: ', j, "; ", self.regime_param[j]

            # print 'hello ', A.size, self.regime_param[i].shape, y.size

            abs_val[:, j] = np.sum(np.absolute(self.regime_param[j]))
            obj_value = np.dot(A[j], self.regime_param[j].reshape(self.features - 1, 1)) - y[j]
            # print 'obj :::', obj_value.shape
            obj += np.dot(obj_value.T, obj_value)

            # print 'cumulative objective function value: ', obj

        self.obj = obj
        self.abs_v = abs_val
        # print abs_val

    def train(self, l):

        U_old = copy.copy(self.U)
        self.update_belittle(l)
        # 1  print 'U before updation:', self.U

        self.update_membership()
        # 1  print 'U after updation:', self.U
        error = LA.norm(U_old - self.U)
        # print '--------------->',error
        print 'Em:', self.Em()

        self.training_iter = 0
        while error > self.epsilon and self.training_iter < 1000:
            # 1  print 'U before updation:', self.U
            U_old = copy.copy(self.U)
            # 1  print "error: ", self.E
            # 1  print "U: ", self.U

            self.update_belittle(l)
            # 1  print 'after update param'

            self.update_membership()
            # 1  print 'after update membership'

            # this error depends on the membership values
            error = LA.norm(U_old - self.U)
            print 'error------>', error
            print 'Em:', self.Em()

            self.training_iter += 1
        # 1  print 'U after updation:', self.U
        # self.obj = np.sqrt(self.obj/self.number)
        print "error after completing training: ", error

        '''RMSE calculation'''
        new_U = self.clusters()

        clus_1, clus_2 = self.set_from_array(new_U)
        data_1 = np.take(self.data, clus_1, axis=1)
        data_2 = np.take(self.data, clus_2, axis=1)
        X_1 = data_1[:-1, :].T
        Y_1 = data_1[-1, :].reshape(data_1.shape[1], 1)
        X_2 = data_2[:-1, :].T
        Y_2 = data_2[-1, :].reshape(data_2.shape[1], 1)

        rms_1 = math.sqrt(mse(np.dot(X_1, self.regime_param[0]), Y_1) + mse(np.dot(X_2, self.regime_param[1]), Y_2))
        rms_2 = math.sqrt(mse(np.dot(X_2, self.regime_param[1]), Y_2) + mse(np.dot(X_1, self.regime_param[0]), Y_1))
        if (rms_1 > rms_2):
            self.rms = rms_2
        else:
            self.rms = rms_1

    def update_membership(self):
        self.construct_error_matrix()

        # 1  print 'error terms:', self.E

        for k in range(self.number):
            if all(self.E[:, k] > 0) == True:
                # 1  print 'all error terms are non zero.'
                den = np.zeros(shape=(self.c, 1))
                for i in range(2):
                    den[i, 0] = float(1 / self.calc_denominator(i, k))

                # 1  print "updated value", den[:, 0]
                self.U[:, k] = den[:, 0]

            else:
                # 1  print 'some error terms are zero.'
                for i in range(self.c):
                    if self.E[i, k] > 0:
                        self.U[i, k] = 0.0
                    else:
                        if sum(x > 0 for x in self.E[:, k]) > 0:
                            self.U[i, k] = float(1 / (sum(x > 0 for x in self.E[:, k])))

    def calc_denominator(self, i, k):
        value = 0.0
        for j in range(self.c):
            value = value + math.pow(self.E[i, k] / self.E[j, k], 1 / (self.m - 1))
        return value

    def construct_error_matrix(self):
        for i in range(self.c):
            for j in range(self.number):
                self.E[i, j] = self.calculate_error(i + 1, self.data[-1, j], self.data[:-1, j])

    def calculate_error(self, regime, dep, indep):

        return np.square(dep - (np.dot(self.regime_param[regime - 1], indep.reshape(self.features - 1)) + self.delta))

    def fill_err(self):
        count1 = 0
        count2 = 0
        for i in range(self.number):
            if self.U[0, i] > self.U[1, i]:
                # print i,'in ', 0
                count1 += 1
                self.err += self.U[0, i] * self.E[0, i]
            else:
                # print i,'in ', 1
                count2 += 1
                self.err += self.U[1, i] * self.E[1, i]
        self.crisp_obj = self.err
        self.err = np.sqrt(self.err / self.number)

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

    lmd = [0.0, 0.01, .1, .2, .5, 1.0]
    c = {}
    rms= {}
    for element in lmd:

        sr = SwitchingRegression_g()

        sr.train(element)
        sr.fill_err()

        no_of_iter[element] = sr.training_iter
        p = np.matrix.sum(np.matrix(sr.E * sr.U))

        res[element] = [sr.regime_param, np.array([[sr.crisp_obj]]), np.array([[sr.err]]),
                        np.array([[np.matrix.sum(np.matrix(sr.E * sr.U))]]), sr.abs_v, sr.belittle]

        obj.append(sr.obj)
        abs_va.append(sr.abs_v)
        c[element] = sr.belittle
        #print sr.belittle
        rms[element] = sr.err

    print res
    print rms

    with open('output/fcrm.garrotte.usedcars.single_c.csv', 'wb') as file:
        writer = csv.writer(file, delimiter='\t')
        writer.writerow(res.keys())
        writer.writerow('-')
        for row in zip(*res.values()):
            for row1 in zip(*row):
                for row2 in zip(*row1):
                    writer.writerow(list(row2))
                writer.writerow('-')
