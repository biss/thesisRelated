
__author__ = 'biswajeet'

''' version 1.1: fcrm with L-1 norm
'''

import copy
import csv
import math
import numpy as np
from numpy import genfromtxt
from numpy import linalg as LA
from scipy import stats

from cvxopt import matrix, solvers


class SwitchingRegression:
    def __init__(self):
        # total number of data points
        self.data = genfromtxt('input/kuiper.csv', delimiter=',', skip_header=1)

        #self.data = self.data / self.data.max(axis=0)
        self.data = stats.zscore(self.data, axis=0)

        self.number, self.dim = self.data.shape
        # initialized parameters of the two regimes, in (y = a + b*x1+ c*x2 + d*x3 + e*x4
        # --> 1st component is a, 2nd is b, 3rd is c... ie [a, b, c, d, e]
        self.regime_param = np.array([[0.0, 0, 0, 0, 0, 0], [0.0, 0, 0, 0, 0, 0]])
        #self.regime2_param = [0.0, 0, 0, 0, 0, 0]

        # membership matrix initialised uniformly

        # U = genfromtxt('input/u.csv', delimiter=',')
        # U = np.array([[1,0],[1,0],[1,0],[0,1],[0,1],[0,1]]).T
        self.m = 2
        self.c = 2
        self.abs_v = np.zeros(shape=(1, self.c))
        self.epsilon = 0.000005
        self.E = np.random.uniform(0, 1.0, [self.c, self.number])
        self.delta = 0
        self.training_iter = 0
        self.U = np.random.uniform(0.0, 1.0, [self.c, self.number])
        print 'data:', self.data.shape
        self.err = 0
        self.crisp_obj = 0
        #supress print messages
        solvers.options['show_progress'] = False

    def newModel(self, A, b, lm):
        m, n = A.size

        if lm == 0.0:
            I = matrix(0.0, (n, n))
            I[::n + 1] = 2.0
            Q = matrix(I*A.T * A)
            # print 'Q :', Q

            p = matrix(-I * A.T * b)
            # I[::col + 1] = 1.0
            # G = matrix(-I)
            # h = matrix(col * [0.0])
            return solvers.qp(Q, p)
        else:
            #only l-1 regularizer
            Q = matrix([[A.T*A, -A.T*A],[-A.T*A, A.T*A]])

            ones = matrix(1.0*lm, (2*n,1))
            aux = matrix([-A.T*b, A.T*b])
            c = ones + aux
            N = 2*n
            I = matrix(0.0, (N,N))
            I[::N+1] = 1.0
            G = matrix(-I)
            h = matrix(n*[0.0] + n*[0.0])
            return solvers.qp(Q, c, G, h)

    def update_param(self, lmd):
        #1
        #print
        #print 'in param_update:'

        X = self.data[:, :-1]
        Y = self.data[:, -1]

        abs_val = np.zeros(shape=(1, self.c))

        root_U = np.power(self.U, self.m/2)
        obj = []
        for i in range(self.c):

            D = np.diag(root_U[i, :])

            ''' construct the new A and y matrix'''
            A = matrix(np.dot(D, X))
            y = matrix(np.dot(D, Y))
            if lmd == 0.0:
                sol = self.newModel(A, y, lmd)
                res = np.array(sol['x'])
            else:
                sol = self.newModel(A, y, lmd)
                new = np.array(sol['x'])
                res = []
                for j in range(self.dim-1):
                    #print 'hello: ',new[j,:], new[j+6, :]
                    res.append(new[j,:] - new[j + self.dim -1 , :])
            print self.regime_param[i].shape,
            self.regime_param[i] = np.array(res).reshape(6,)

            '''print 'l-1 norm value of ',i+1, 'th model: ', sum(np.absolute(self.regime_param[i]))'''

            obj_value = np.dot(A, self.regime_param[i].reshape(self.dim-1, 1)) - y
            obj.append(np.dot(obj_value.T, obj_value))

        print 'objective function value: ', obj[0] + obj[1]

        for j in range(self.c):
            abs_val[:, j] = np.sum(np.absolute(self.regime_param[j]))

        self.abs_v = abs_val


    def train(self, lmd):

        print 'starting with regime params: ', self.regime_param
        print 'initial U: ', self.U
        U_old = copy.copy(self.U)
        self.update_param(lmd)
        #1  print 'U before updation:', self.U

        self.update_membership()
        #1  print 'U after updation:', self.U
        error = LA.norm(U_old - self.U)
        #print '--------------->',error

        '''print 'Em:',self.Em()'''

        self.training_iter = 1
        while error > self.epsilon:
            #1  print 'U before updation:', self.U
            U_old = copy.copy(self.U)
            #1  print "error: ", self.E
            #1  print "U: ", self.U

            self.update_param(lmd)
            #1  print 'after update param'

            self.update_membership()
            #1  print 'after update membership'

            #this error depends on the membership values
            error = LA.norm(U_old - self.U)

            '''print 'error------>', error
            #print 'Em:', self.Em()'''

            self.training_iter += 1
        #1  print 'U after updation:', self.U

        '''print "frobenius norm after completing training: ", error'''
        print 'ending with regime params: ', self.regime_param
        print 'final U: ', self.U

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
        for i in range(2):
            for j in range(self.number):

                self.E[i, j] = self.calculate_error(i, self.data[j, -1], self.data[j, :-1])

    def calculate_error(self, regime, dep, indep):

        return math.pow(dep - (np.dot(self.regime_param[regime].reshape(1, self.dim - 1), indep.reshape(self.dim - 1, 1)) + self.delta),2)

    def Em(self):
        error = 0
        for k in range(self.number):
            for i in range(self.c):
                error += pow(self.U[i, k], self.m) * self.E[i, k]
        return error

    def fill_err(self):
        tmp = 0
        for i in range(self.number):
            if self.U[0, i] > self.U[1, i]:
                tmp += self.U[0, i] * self.E[0, i]
            else:
                tmp += self.U[1, i] * self.E[1, i]
        self.crisp_obj = tmp
        self.err = np.sqrt(tmp / self.number)

if __name__ == '__main__':

    res = {}
    no_of_iter = {}
    rms = {}
    lmd=[0.0, 1.0, .1, .2, .5, 10]
    for element in lmd:
        print 'for element :', element
        sr = SwitchingRegression()

        sr.train(element)
        sr.fill_err()
        no_of_iter[element] = sr.abs_v
        res[element] = sr.regime_param
        rms[element] = [sr.err, sr.crisp_obj, np.matrix.sum(np.matrix(sr.E*sr.U))]
    print res
    print rms
    print no_of_iter

    with open('output/fcrm_with_l1.usedcars.normalized.csv', 'wb') as file:
        writer = csv.writer(file, delimiter='\t')
        writer.writerow(res.keys())
        writer.writerow('-')
        for row in zip(*res.values()):
            for row2 in zip(*row):
                writer.writerow(list(row2))
            writer.writerow('-')