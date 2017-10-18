
__author__ = 'biswajeet'
__date__ = '24-03-2015'

''' -------------DATA: USEDCARS ------------
    garrote estimate on unnormalized synthetic data. All the features
    are normalised code change: the estimater holders were changed to
    numpy array. Writes file with a comparision view. The betas were
    initialised with the lasso estimates.

    CHANGES:
    1. RMS error is calculated without the membership values.
    The membership values are first made crisp and the rms value is
    calculated clusterwise.
    2. Separate c's are used in each model'''

import numpy as np
import math, csv, pandas as pd
from numpy import linalg as LA
import copy
from cvxopt import matrix, solvers
from sklearn.metrics import mean_squared_error as mse

''' class implementing all the necessary variables and methods'''
class SwitchingRegression_g:

    def __init__(self, cluster=2):

        self.c = cluster

        self.crisp_obj = 0

        # mean centered and unit std dev data
        self.data = np.genfromtxt('input/out_n.csv', dtype=float, delimiter=',')
        self.features, self.number = self.data.shape
        #print 'first ', self.features, self.number

        ''' variable to store the value of the objective function after each iteration.
        This objective value is the sum of objective value of all the c models. '''
        self.obj = 0

        '''rms error'''
        self.rms = 0

        ''' stores the sum of absolute values of the models'''
        self.abs_v = np.zeros(shape=(1, self.c))

        ''' each column of data is a features-dimensional data sample.
            Example: [x1, x2, x3, x4, y].T is a column.
            So, the data matrix consists of 'number' number of columns'''

        self.betas = np.zeros(shape=(self.c, self.features - 1), dtype=float)
        self.regime_param = np.zeros(shape=(self.c, self.features - 1), dtype=float)
        self.belittle = np.zeros(shape=(self.c, self.features - 1), dtype=float)

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

        # supress print messages
        solvers.options['show_progress'] = False

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

    ''' estimates the parameter of the model using the new cost function
        and quadratic programming'''
    def newModel(self, A, b, l):
        row, col = A.size
        #print row, col

        if l == 0.0:
            I = matrix(0.0, (col, col))
            I[::col + 1] = 2.0
            Q = matrix(I * A.T * A)
            p = matrix(-I * A.T * b)
            return solvers.qp(Q, p)

        else:
            I = matrix(0.0, (col, col))
            I[::col + 1] = 2.0
            Q = matrix(I * A.T * A)

            ones = matrix(1.0 * l, (col, 1))
            aux = matrix(-I * A.T * b)
            p = ones + aux
            I[::col + 1] = 1.0
            G = matrix(-I)
            h = matrix(col * [0.0])
            return solvers.qp(Q, p, G, h)

    def clusters(self):
        return np.round(self.U, 0)

    ''' returns a list of list. first sublist is the list of samples number
        which are crispified to first model. second sublist has similar
        meaning'''
    def set_from_array(self, array):
        c = [[], []]
        cl, n = array.shape
        for i in range(n):
            for j in range(cl):
                if array[j, i] == 1:
                    c[j].append(i)
        return c[0], c[1]

    ''' updates the value of the scaling parameters, objective value of the model
        absolute value of the model. the scaling parameters are estimated using
        quadratic programming.'''

    def update_belittle(self, l):

        #print 'Iternation no.:', self.training_iter

        X = self.data[:-1, :].T
        #print 'hello :',self.data[:-1, :].shape, self.data[-1, :].shape, self.number
        Y = self.data[-1, :].reshape(self.number, 1)

        root_U = np.power(self.U, self.m/2)
        abs_val = np.zeros(shape=(1,2))

        for i in range(self.c):

            D_sqr_U = np.diag(list(root_U[i, :]))
            D_beta = np.diag(list(self.betas[i, :]))

            A = matrix(D_sqr_U.dot(X).dot(D_beta))
            #print 'A ', A
            y = matrix(np.dot(D_sqr_U, Y))

            sol = self.newModel(A, y, l)
            res = np.array(sol['x'])

            self.belittle[i] = res.T

        for j in range(self.c):

            self.regime_param[j] = self.belittle[j] * self.betas[j]
            #print 'l-1 norm value of ',j+1, 'th model: ', np.sum(np.absolute(self.regime_param[j]))
            abs_val[:, j] = np.sum(np.absolute(self.regime_param[j]))

        self.abs_v = abs_val

    def update_param(self):
        # 1  print 'in param_update:'

        X = self.data[:-1, :].T
        Y = self.data[-1, :].reshape(1, self.number)

        for i in range(self.c):
            D = np.zeros(shape=(self.number, self.number))
            for k in range(self.number):
                D[k, k] = copy.copy(pow(self.U[i, k], self.m))

            a = (X.T).dot(D).dot(X)
            b = np.linalg.pinv(a)
            #print 'shape ', self.regime_param[i].shape
            self.regime_param[i] = np.dot(b, ((X.T).dot(D).dot(Y.T))).reshape(self.features-1,)


    def train(self, l):

        U_old = copy.copy(self.U)
        self.update_param()

        self.update_membership()
        error = LA.norm(U_old - self.U)
        print 'Em:',self.Em()

        self.training_iter = 0
        while error > self.epsilon and self.training_iter < 1000:

            U_old = copy.copy(self.U)

            self.update_param()

            self.update_membership()

            #this error depends on the membership values
            error = LA.norm(U_old - self.U)
            print 'error------>', error
            print 'Em:', self.Em()

            self.training_iter += 1

        print "parameters after completing training: ", self.regime_param
        #print 'obj ',np.matrix.sum(np.matrix(self.E))

        self.betas = self.regime_param
        self.regime_param = np.zeros(shape=(self.c, self.features - 1), dtype=float)

        self.update_belittle(l)
        self.update_membership()
        error = LA.norm(U_old - self.U)
        print 'error------>', error
        print 'Em:', self.Em()

        X = self.data[:-1, :].T
        Y = self.data[-1, :].reshape(self.number, 1)
        obj = 0.0

        for j in range(self.c):

            obj_value = np.dot(X, self.regime_param[j].reshape(self.features - 1, 1)) - Y
            #print 'obj :::', obj_value.shape
            obj += np.dot(obj_value.T, obj_value)

        print 'final objective function value: ', float(obj)
        self.obj = obj


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

        return np.square(dep - (np.dot(self.regime_param[regime - 1], indep.reshape(self.features - 1)) + self.delta))

    def Em(self):
        error = 0
        for k in range(self.number):
            for i in range(self.c):
                error += pow(self.U[i, k], self.m) * self.E[i, k]
        return error

if __name__ == '__main__':

    clusters = 2

    obj = {}
    abs_va = []
    res = {}
    no_of_iter = {}

    lmd = [0.0, 0.1, .5, 1, 0.01]
    c = {}
    rms = {}
    for element in lmd:

        sr = SwitchingRegression_g(clusters)
        #print sr.regime_param[0].shape
        sr.train(element)
        sr.fill_err()

        #specifying limits
        #print 'betas for ',element,' are: ', sr.regime_param
        #print sr.training_iter
        no_of_iter[element] = sr.training_iter
        res[element] = [sr.regime_param, sr.belittle, np.array([[sr.err]]), np.array([[np.matrix.sum(np.matrix(sr.E*sr.U))]]), np.array([[sr.crisp_obj]])]

        #res[element] = sr.regime_param
        obj[element]=sr.obj
        abs_va.append(sr.abs_v)
        c[element] = sr.belittle
        rms[element] = sr.err

    print res
    print rms

    with open('output/fcrm.improved_garrotte.synthetic.multiple_c.csv', 'wb') as file:
        writer = csv.writer(file, delimiter='\t')
        writer.writerow(res.keys())
        writer.writerow('-')
        for row in zip(*res.values()):
            for row1 in zip(*row):
                for row2 in zip(*row1):
                    writer.writerow(list(row2))
                writer.writerow('-')