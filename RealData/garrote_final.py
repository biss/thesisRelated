
__author__ = 'biswajeet'
__date__ = '28-03-2015'

''' garrote estimate on usedcar data. All the features are normalised
    code change: the estimater holders were changed to numpy array.
    Writes file with a comparision view. The betas were initialised with
    the lasso estimates'''

import numpy as np
import math, csv, pandas as pd
from numpy import linalg as LA
import copy
from cvxopt import matrix, solvers

''' class implementing all the necessary variables and methods'''
class SwitchingRegression_g:

    # number of samples
    number = 1
    # number of features
    features = 1
    # number of models
    c = 2

    ''' variable to store the value of the objective function after each iteration.
    This objective value is the sum of objective value of all the c models. '''
    obj = 0

    ''' stores the sum of absolute values of the models'''
    abs_v = np.zeros(shape = (1,c))

    ''' each column of data is a features-dimensional data sample.
        Example: [x1, x2, x3, x4, y].T is a column.
        So, the data matrix consists of 'number' number of columns'''
    data = np.zeros(shape = (features, number), dtype=float)

    '''initialize parameters of the two models(regimes).
        For eg in (y = b*x1 + c*x2 + d*x3 + e*x4)
       --> 1st component is a, 2nd is b, 3rd is c... i.e. [a, b, c, d, e]'''
    betas = np.zeros(shape = (c, features - 1), dtype=float)
    regime_param = np.zeros(shape = (c, features - 1), dtype=float)
    belittle = np.zeros(shape = (1, features - 1), dtype=float)

    '''membership matrix initialised uniformly'''
    U = np.random.uniform(0.0, 1.0, [c, number])

    m = 1.2

    ''' for terminating condition'''
    epsilon = 0.0000005

    '''error matrix initialise'''
    E = np.random.uniform(0.0, 1.0, [c, number])

    ''' for invertibility issue.'''
    delta = 0

    '''stores the number of training iterations required to
        reach the precision of epsilon'''
    training_iter = 0

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

        Q = matrix(A_1.T*A_1 + A_2.T*A_2)

        ones = matrix(1.0 * l, (n,1))
        aux = matrix(A_1.T*b_1) + matrix(A_2.T*b_2)
        c = ones - aux
        N = n
        I = matrix(0.0, (N,N))
        I[::N+1] = 1.0
        G = matrix(-I)
        h = matrix(n*[0.0])
        return solvers.qp(Q, c, G, h)

    def load_data(self):

        # data is mean centered and unit std dev
        data = pd.read_csv('input/kuiper_ip.csv', sep=',')

        self.number, self.features = data.shape
        print self.number, self.features
        #print data

        ''' reinitializing the parameters of the model '''
        self.regime_param = np.zeros(shape = (self.c, self.features - 1), dtype=float)
        self.betas = np.zeros(shape = (self.c, self.features - 1), dtype=float)
        self.belittle = np.zeros(shape = (1, self.features - 1), dtype=float)

        self.data = np.zeros(shape = (self.features, self.number), dtype=float)

        ''' set the betas of the model:  as the lasso estimates '''
        self.betas[0, :] = np.array([[-0.11322999, -0.44083284, -0.14810016, 0.20871709, -0.02476554, 0.06751583]])
        self.betas[1, :] = np.array([[ -0.16078763, 1.21275641, -0.25432137, -0.00653286, 0.04788197, 0.07565833]])

        '''load data '''
        self.data = np.array(data).T
        print self.data

    def init(self, cluster):
        self.c = cluster
        self.load_data()
        self.U = np.random.uniform(0.0, 1.0, [self.c, self.number])
        self.E = np.random.uniform(0.0, 1.0, [self.c, self.number])

    def update_belittle(self, l):
        #1
        #print
        #print 'Iternation :', self.training_iter

        X = self.data[1:, :].T
        Y = self.data[0, :].reshape(self.number, 1)

        #print 'Y', Y.shape

        root_U = np.power(self.U, self.m/2)

        A = np.array([np.zeros(shape=(self.number, self.features - 1)), np.zeros(shape=(self.number, self.features - 1))])
        #A_2 = matrix(np.zeros(shape=(self.number, self.features - 1)))
        y = np.array([np.zeros(shape=(self.number, 1)), np.zeros(shape=(self.number, 1))])

        obj = 0
        abs_val = np.zeros(shape=(1,2))

        ''' use the  '''
        for i in range(self.c):

            D_sqr_U = np.diag(root_U[i, :])
            D_beta = np.diag(list(self.betas[i, :]))

            #print 'beta diag ', D_beta
            #print 'beta diag ', list(self.betas[i, :])

            ''' construct the new A and y matrix. y = diag(sqrt(U));
                A = diag(sqrt(U)) (X) diag(sqrt(beta))'''

            #print 'imp ', D_sqr_U.shape, X.shape, D_beta.shape

            A[i] = D_sqr_U.dot(X).dot(D_beta)
            y[i] = np.dot(D_sqr_U, Y)

        sol = self.newModel(A, y, l)
        res = np.array(sol['x'])

        self.belittle[0] = res.T

        for j in range(self.c):
            #print 'Model :', j
            #print 'c: ', self.belittle[0]
            #print 'beta: ', j,"; ",self.betas[j]

            self.regime_param[j] = self.belittle[0] * self.betas[j]

            #print 'regime_param: ',j,"; ",self.regime_param[j]

            #print 'l-1 norm value of ',j+1, 'th model: ', np.sum(np.absolute(self.regime_param[j]))

            #print 'hello ', A.size, self.regime_param[i].shape, y.size

            abs_val[:, j] = np.sum(np.absolute(self.regime_param[j]))
            obj_value = np.dot(A[j], self.regime_param[j].reshape(self.features - 1, 1)) - y[j]
            #print 'obj :::', obj_value.shape
            obj += np.dot(obj_value.T, obj_value)

            #print 'cumulative objective function value: ', obj

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
        #self.obj = np.sqrt(self.obj/self.number)
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

    #lmd = [0.1*i for i in range(10)]
    #lmd = lmd + [1.0*i for i in range(2,10)]
    lmd = [.5]
    c = {}
    for element in lmd:

        sr = SwitchingRegression_g()
        sr.init(clusters)
        #print sr.regime_param[0].shape
        sr.train(element)

        #specifying limits
        #print 'betas for ',element,' are: ', sr.regime_param
        #print sr.training_iter
        no_of_iter[element] = sr.training_iter
        res[element] = [sr.regime_param, np.sqrt(sr.obj/sr.number), sr.obj, sr.abs_v, sr.belittle]

        #res[element] = sr.regime_param
        obj.append(sr.obj)
        abs_va.append(sr.abs_v)
        c[element] = sr.belittle
        print 'belittle',sr.belittle

    print res

    with open('output/result_garrote_1.0.csv', 'wb') as file:
        writer = csv.writer(file, delimiter='\t')
        writer.writerow(res.keys())
        writer.writerow('-')
        for row in zip(*res.values()):
            for row1 in zip(*row):
                for row2 in zip(*row1):
                    writer.writerow(list(row2))
                writer.writerow('-')
    #print obj

    '''
            for row1 in zip(*row):
                writer.writerow(list(row1))
            writer.writerow('-')
        writer.writerow(obj)
        writer.writerow(abs_va)

        for row in zip(*c.values()):
            for row1 in zip(*row):
                writer.writerow(list(row1))

    sr = SwitchingRegression()

    sr.load_data()

    sr.train()

    #specifying limits
    print 'betas are: ', sr.regime_param
    print sr.training_iter'''