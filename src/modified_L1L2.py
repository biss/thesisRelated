
__author__ = 'biswajeet'

''' version 1.1: introducing L-1, L2 norm'''

import copy
import csv
import math
import numpy as np
from numpy import genfromtxt
from numpy import linalg as LA

from cvxopt import matrix, solvers


class SwitchingRegression:

    #total number of data points
    number = 200

    #each column of data is a data point,[x1, x2, x3, x4, y].T is a column
    data = np.zeros(shape = (7, number), dtype=float)
    #data = np.array([[1,2],[2,2.5],[5,4], [2,1],[3,4],[4,7]]).T

    #initialized parameters of the two regimes, in (y = a + b*x1+ c*x2 + d*x3 + e*x4
    # --> 1st component is a, 2nd is b, 3rd is c... ie [a, b, c, d, e]
    regime_param = [[0.0, 0, 0, 0, 0, 0], [0.0, 0, 0, 0, 0, 0]]
    #regime2_param = [0.0, 0, 0, 0, 0, 0]

    #membership matrix initialised uniformly
    U = np.random.uniform(0.0, 1.0, [2,number])
    #U = np.array([[1,0],[1,0],[1,0],[0,1],[0,1],[0,1]]).T
    m = 1.2
    c = 2
    epsilon = 0.0000005
    E = np.random.uniform(0, 1.0, [2,number])
    delta = 0
    training_iter = 0
    #supress print messages
    solvers.options['show_progress'] = False

    def newModel(self, A, b, l):
        m, n = A.size

        #with l-1 and l-2 regularizer

        I = matrix(np.diag([1,1,1,1,1,1]))
        #print 'shape: ', (A.T*A).size, (A.T*A + I).size
        Q = matrix([[A.T*A + I, -A.T*A - I],[-A.T*A - I, A.T*A + I]])

        #Q = matrix([[A.T*A, -A.T*A],[-A.T*A, A.T*A]])

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
        self.data = genfromtxt('input/out_m.csv', delimiter=',')
        #print 'data:', self.data

    def update_param(self, l):
        #1
        print
        print 'in param_update:'

        X = self.data[:6, :].T
        Y = self.data[-1, :]

        root_U = np.sqrt(self.U)

        ''' use the  '''
        for i in range(self.c):

            D = np.diag(root_U[i, :])

            ''' construct the new A and y matrix'''
            A = matrix(np.dot(D, X))
            y = matrix(np.dot(D, Y))

            sol = self.newModel(A, y, l)
            new = np.array(sol['x'])
            res = []
            for j in range(6):
                #print 'hello: ',new[j,:], new[j+6, :]
                res.append(new[j,:] - new[j+6, :])
            self.regime_param[i] = np.array(res)

            print 'l-1 norm value of ',i+1, 'th model: ', sum(np.absolute(self.regime_param[i]))

            obj_value = np.dot(A, self.regime_param[i]) - y
            obj = np.dot(obj_value.T, obj_value)

            print 'objective function value: ', obj


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
        for i in range(2):
            for j in range(self.number):

                self.E[i, j] = self.calculate_error(i+1, self.data[-1, j], self.data[:-1, j])

    def calculate_error(self, regime, dep, indep):

        if regime == 1:
            #print indep.shape, self.regime_param[regime - 1].shape
            #1  print "variables one: ", dep, indep
            #1  print "calculated value E :", math.pow(dep - (np.dot(self.regime1_param.T, indep) + self.delta), 2)

            return math.pow(dep - (np.dot(self.regime_param[regime-1].T, indep) + self.delta), 2)
        if regime == 2:
            #print indep.shape, self.regime_param[regime - 1].shape
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

    res = {}
    no_of_iter = {}
    lmd = [5]
    for element in lmd:
        print 'for element = ', element

        sr = SwitchingRegression()
        sr.load_data()
        sr.train(element)

        #specifying limits
        #print 'betas for ',element,' are: ', sr.regime_param
        #print sr.training_iter
        no_of_iter[element] = sr.training_iter
        res[element] = sr.regime_param
    print res

    with open('output/fcrm.l1_l2.synthetic.multicolinear.unnorm.csv', 'wb') as file:
        writer = csv.writer(file, delimiter='\t')
        writer.writerow(res.keys())
        for row in zip(*res.values()):
            for row1 in zip(*row):
                writer.writerow(list(row1))
            writer.writerow('-')

    '''sr = SwitchingRegression()

    sr.load_data()

    sr.train()

    #specifying limits
    print 'betas are: ', sr.regime_param
    print sr.training_iter'''