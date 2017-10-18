from sklearn.cross_validation import KFold
import numpy as np
from scipy import stats

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
class SwitchingRegression_ms:

    def __init__(self, data, cluster=2):

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

        self.data = data.T
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

        '''
        results = smf.ols('Price ~ Mileage + Cylinder + Liter + Cruise + Sound + Leather', data=data).fit()
        for i in range(self.c):
            self.betas[i, :] = np.array(results.params[1:])
        '''
        self.betas[0, :] = np.array([[-0.079, 0.336, 0.110, -0.013, 0.002, 0.012]])
        self.betas[1, :] = np.array([[-0.173, 0.488, -0.115, 0.278, 0.010, 0.004]])

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
            # print 'Q :', Q

            p = matrix(-I * A.T * b)
            # I[::col + 1] = 1.0
            # G = matrix(-I)
            # h = matrix(col * [0.0])
            return solvers.qp(Q, p)

        else:
            I = matrix(0.0, (col, col))
            I[::col + 1] = 2.0
            #print I
            Q = matrix(I*A.T*A)
            #print Q
            #print LA.det(Q)

            ones = matrix(1.0 * l, (col, 1))
            aux = matrix(-I*A.T * b)
            p = ones + aux
            I[::col + 1] = 1.0
            G = matrix(-I)
            h = matrix(col * [0.0])
            return solvers.qp(Q, p, G, h)

    ''' load the usedcars dataset which is already mean centered and unit std dev,
        initialise the parameters of the model. The beta parameter is initialised
        with the lasso estimates
    '''

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

    ''' '''
    def update_belittle(self, l):

        #print 'Iternation no.:', self.training_iter

        X = self.data[:-1, :].T
        #print 'hello :',self.data[:-1, :].shape, self.data[-1, :].shape
        Y = self.data[-1, :].reshape(self.number, 1)

        root_U = np.power(self.U, self.m/2)

        obj = 0
        abs_val = np.zeros(shape=(1, self.c))

        ''' use the  '''
        for i in range(self.c):

            D_sqr_U = np.diag(root_U[i, :])
            D_beta = np.diag(list(self.betas[i, :]))

            ''' construct the new A and y matrix. y = diag(sqrt(U));
                A = diag(sqrt(U)) (X) diag(sqrt(beta))'''
            A = matrix(D_sqr_U.dot(X).dot(D_beta))
            y = matrix(np.dot(D_sqr_U, Y))

            sol = self.newModel(A, y, l)
            res = np.array(sol['x'])

            self.belittle[i] = res.T

        for j in range(self.c):
            #print 'Model :', j
            #print 'c: ', self.belittle[0]
            #print 'beta: ', j,"; ",self.betas[j]

            self.regime_param[j] = self.belittle[j] * self.betas[j]

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

    def fill_err(self):
        tmp = 0
        for i in range(self.number):
            if self.U[0, i] > self.U[1, i]:
                tmp += self.U[0, i]*self.E[0, i]
            else:
                tmp += self.U[1, i]*self.E[1, i]
        self.crisp_obj = tmp
        self.err = np.sqrt(tmp / self.number)
        # print 'count ',count1, count2
        #print 'count ',count1, count2

    def train(self, l):

        U_old = copy.copy(self.U)
        self.update_belittle(l)
        #1  print 'U before updation:', self.U

        self.update_membership()
        #1  print 'U after updation:', self.U
        error = LA.norm(U_old - self.U)
        #print '--------------->',error
        #print 'Em:',self.Em()

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

            #print 'error------>', error
            #print 'Em:', self.Em()

            self.training_iter += 1
        # 1  print 'U after updation:', self.U
        # self.obj = np.sqrt(self.obj/self.number)

        #print "error after completing training: ", error

    def update_membership(self):

        self.construct_error_matrix()

        #1  print 'error terms:', self.E

        for k in range(self.number):
            if all(self.E[:, k] > 0) == True:
                #1  print 'all error terms are non zero.'
                den = np.zeros(shape=(self.c, 1), dtype = float)
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
        error = 0.0
        for k in range(self.number):
            for i in range(self.c):
                error += pow(self.U[i, k], self.m) * self.E[i, k]
        return error



if __name__ == '__main__':

    clusters = 2

    data = np.genfromtxt('input/kuiper.csv', dtype=float, delimiter=',', skip_header=1)
    data = stats.zscore(data, axis=0)

    X = np.delete(data, -1, 1)
    y = data[:, -1]
    kf = KFold(804, n_folds=5)
    print len(kf)
    print kf

    for train_index, test_index in kf:
        # print("TRAIN:", train_index, "TEST:", test_index)
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        train = np.c_[X_train, y_train]
        test = np.c_[X_test, y_test]

        print train.shape
        print test.shape

        lmd = [i for i in range(0.0, 1.0, 0.01)]
        beta = {}
        res= {}

        for element in lmd:

            sr = SwitchingRegression_ms(X_train)

            sr.train(element)

            # res[element] = sr.regime_param

            #rms[element] = np.matrix.sum(np.matrix(sr.E * sr.U))
            print element,': ', np.matrix.sum(np.matrix(sr.E * sr.U))
            sr.fill_err()

            res[element] = [sr.regime_param, np.array([[sr.crisp_obj]]), np.array([[sr.err]]), np.array([[np.matrix.sum(np.matrix(sr.E*sr.U))]]), sr.abs_v, sr.belittle]

