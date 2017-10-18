
__author__ = 'biswajeet'
__date__ = '24-03-2015'

''' -------------DATA: USEDCARS ------------
    garrote estimate on normalized synthetic data. All the features
    are normalised code change: the estimater holders were changed to
    numpy array. Writes file with a comparision view. The betas were
    initialised with the lasso estimates.

    CHANGES:
    1. RMS error is calculated without the membership values.
    The membership values are first made crisp and the rms value is
    calculated clusterwise.
    2. Separate c's are used in each model'''

import numpy as np
import math, csv
from numpy import linalg as LA
import copy
from cvxopt import matrix, solvers
from scipy import stats
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import confusion_matrix as cm

''' class implementing all the necessary variables and methods'''
class SwitchingRegression:

    def __init__(self, cluster=2):

        self.c = cluster
        self.crisp_obj = 0

        ''' variable to store the value of the objective function after each iteration.
        This objective value is the sum of objective value of all the c models. '''
        self.obj = 0

        '''rms error'''
        self.rms = 0

        ''' stores the sum of absolute values of the models'''
        self.abs_v = np.zeros(shape = (1, self.c))

        self.m = 2

        ''' for terminating condition'''
        self.epsilon = 0.000005

        ''' for invertibility issue.'''
        self.delta = 0

        '''stores the number of training iterations required to
            reach the precision of epsilon'''
        self.training_iter = 0

        ''' each column of data is a features-dimensional data sample.
            Example: [x1, x2, x3, x4, y].T is a column.
            So, the data matrix consists of 'number' number of columns'''
        data = np.genfromtxt('input/kuiper.csv', dtype=float, delimiter=',', skip_header=1)

        #data = data / data.max(axis=0)
        data = stats.zscore(data, axis=0)
        self.number, self.features = data.shape
        print self.number, self.features
        self.data = data
        #print self.data

        '''initialize parameters of the two models(regimes).
            For eg in (y = b*x1 + c*x2 + d*x3 + e*x4)
           --> 1st component is a, 2nd is b, 3rd is c... i.e. [a, b, c, d, e]'''
        self.betas = np.zeros(shape = (self.c, self.features - 1), dtype=float)
        self.regime_param = np.zeros(shape = (self.c, self.features - 1), dtype=float)
        self.belittle = np.zeros(shape = (self.c, self.features - 1), dtype=float)

        ''' membership matrix initialised uniformly '''
        self.U = np.random.uniform(0.0, 1.0, [self.c, self.number])

        '''error matrix initialise'''
        self.E = np.random.uniform(0.0, 1.0, [self.c, self.number])

        # supress print messages
        solvers.options['show_progress'] = False

        self.err = 0

    ''' estimates the parameter of the model using the new cost function
        and quadratic programming'''

    def newModel(self, A, b, l):
        row, col = A.size
        #print row, col

        # with l-1 and l-2 regularizer
        '''
        I = matrix(np.diag([1,1,1,1,1,1]))
        #print 'shape: ', (A.T*A).size, (A.T*A + I).size
        Q = matrix([[A.T*A + I, -A.T*A - I],[-A.T*A - I, A.T*A + I]])
        '''

        if l == 0.0:
            I = matrix(0.0, (col, col))
            I[::col + 1] = 2.0
            Q = matrix(I * A.T * A)
            p = matrix(-I * A.T * b)

            return solvers.qp(Q, p)

        else:
            I = matrix(0.0, (col, col))
            I[::col + 1] = 2.0
            Q = matrix(I*A.T*A)
            ones = matrix(1.0 * l, (col, 1))
            aux = matrix(-I*A.T * b)
            p = ones + aux
            I[::col + 1] = 1.0
            G = matrix(-I)
            h = matrix(col * [0.0])
            return solvers.qp(Q, p, G, h)

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

    def update_param(self):
        # 1  print 'in param_update:'

        X = self.data[:, :-1]
        Y = self.data[:, -1].reshape(1, self.number)

        for i in range(self.c):
            D = np.zeros(shape=(self.number, self.number))
            for k in range(self.number-1):
                D[k, k] = copy.copy(pow(self.U[i, k], self.m))

            a = (X.T).dot(D).dot(X)

            # avoid the singular case
            b = np.linalg.pinv(a)
            self.betas[i] = np.dot(b, ((X.T).dot(D).dot(Y.T))).reshape(self.features - 1, )

    def update_belittle(self, l):

        X = self.data[:, :-1]
        print 'hello :',self.data[:, :-1].shape, self.data[:, -1].shape
        Y = self.data[:, -1].reshape(self.number, 1)

        root_U = np.power(self.U, self.m/2)

        obj = 0
        abs_val = np.zeros(shape=(1, self.c))

        for i in range(self.c):

            D_sqr_U = np.diag(root_U[i, :])
            D_beta = np.diag(list(self.betas[i, :]))
            print D_sqr_U.shape, X.shape, D_beta.shape
            A = matrix(D_sqr_U.dot(X).dot(D_beta))
            y = matrix(np.dot(D_sqr_U, Y))

            sol = self.newModel(A, y, l)
            res = np.array(sol['x'])

            self.belittle[i] = res.T

        for j in range(self.c):

            self.regime_param[j] = self.belittle[j] * self.betas[j]

            abs_val[:, j] = np.sum(np.absolute(self.regime_param[j]))
            obj_value = np.dot(X, self.regime_param[j].reshape(self.features - 1, 1)) - Y
            obj += np.dot(obj_value.T, obj_value)

        self.obj = obj
        self.abs_v = abs_val

    def fill_err(self):

        tmp = 0
        for i in range(self.number):
            if self.U[0, i] > self.U[1, i]:
                tmp += self.U[0, i] * self.E[0, i]
            else:
                tmp += self.U[1, i] * self.E[1, i]
        self.crisp_obj = tmp
        self.err = np.sqrt(tmp / self.number)

    def train(self, l):

        U_old = copy.copy(self.U)
        self.update_param()

        self.update_membership_()

        error = LA.norm(U_old - self.U)

        print 'Em:', self.Em()

        self.training_iter = 1
        while error > self.epsilon:

            U_old = copy.copy(self.U)

            self.update_param()

            self.update_membership_()

            # this error depends on the membership values
            error = LA.norm(U_old - self.U)
            print 'error------>', error
            print 'Em:', self.Em()

            self.training_iter += 1

        print 'inside the garrotte'
        self.update_belittle(l)

        self.update_membership()

        error = LA.norm(U_old - self.U)
        #print '--------------->',error
        #print 'Em:',self.Em()

        print "error after completing training: ", error


    def update_membership_(self):

        self.construct_error_matrix_()

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

    def update_membership(self):

        self.construct_error_matrix()

        for k in range(self.number):
            if all(self.E[:, k] > 0) == True:
                #1  print 'all error terms are non zero.'
                den = np.zeros(shape=(self.c, 1))
                for i in range(2):
                    den[i, 0] = float(1/self.calc_denominator(i, k))

                self.U[:, k] = den[:, 0]

            else:
                for i in range(self.c):
                    if self.E[i, k] > 0:
                        self.U[i, k] = 0.0
                    else:
                        if sum (x > 0 for x in self.E[:, k]) > 0:
                            self.U[i, k] = float(1 / (sum (x > 0 for x in self.E[:, k])))

    def calc_denominator(self, i, k):
        value = 0.0
        for j in range(self.c):
            value += math.pow(self.E[i, k]/self.E[j, k], 1/(self.m - 1))
        return value

    def construct_error_matrix(self):
        for i in range(self.c):
            for j in range(self.number):
                self.E[i, j] = self.calculate_error(i+1, self.data[j, -1], self.data[j, :-1])

    def construct_error_matrix_(self):
        for i in range(self.c):
            for j in range(self.number):
                self.E[i, j] = self.calculate_error_(i + 1, self.data[j, -1], self.data[j, :-1])

    def calculate_error(self, regime, dep, indep):
        return np.square(dep - (np.dot(self.regime_param[regime - 1], indep.reshape(self.features - 1)) + self.delta))

    def calculate_error_(self, regime, dep, indep):
        return np.square(dep - (np.dot(self.betas[regime - 1], indep.reshape(self.features - 1)) + self.delta))

    def Em(self):
        error = 0
        for k in range(self.number):
            for i in range(self.c):
                error += pow(self.U[i, k], self.m) * self.E[i, k]
        return error


if __name__ == '__main__':

    clusters = 2

    abs_va = []
    res = {}
    no_of_iter = {}

    lmd = [0.0, 0.01, .1, .2, .5, 1.0]
    rms = {}

    for element in lmd:

        sr = SwitchingRegression()
        sr.train(element)
        sr.fill_err()

        no_of_iter[element] = sr.training_iter

        abs_va.append(sr.abs_v)

        sr.fill_err()
        rms[element] = np.matrix.sum(np.matrix(sr.E * sr.U))

        res[element] = [sr.regime_param, np.array([[sr.crisp_obj]]), np.array([[sr.err]]),
                        np.array([[np.matrix.sum(np.matrix(sr.E * sr.U))]]), sr.abs_v, sr.belittle]

        #actual = list(list(sr.U[:,i]).index(max(list(sr.U[:,i]))) for i in range(sr.number))
        #print 'actual labels ',actual

        '''
        for i in range(9):

            sr = SwitchingRegression_ms()

            sr.train(element)
            sr.fill_err()

            predicted = list(list(sr.U[:, i]).index(max(list(sr.U[:, i]))) for i in range(sr.number))

            confusion_matrix = cm(actual, predicted)
            np.set_printoptions(precision=2)
            print 'Confusion matrix, without normalization for lambda = ', element, ' and round no ', i
            print(confusion_matrix)
        '''

    print res

    with open('output/fcrm.improved_garrotte.usedcars.multiple_c.csv', 'wb') as file:
        writer = csv.writer(file, delimiter='\t')
        writer.writerow(res.keys())
        writer.writerow('-')
        for row in zip(*res.values()):
            for row1 in zip(*row):
                for row2 in zip(*row1):
                    writer.writerow(list(row2))
                writer.writerow('-')
    print rms