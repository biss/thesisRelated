__author__ = 'biswajeet'

''' version 1.1: multi-colinearity. FCRM with dependent features.
    {x5 = 2.2*x1+ random noise}
'''

import numpy as np
import math
from numpy import linalg as LA
import copy
from numpy import genfromtxt

class SwitchingRegression:

    #total number of data points
    number = 200

    #each column of data is a data point,[x1, x2, x3, x4, y].T is a column
    data = np.zeros(shape = (6, number), dtype=float)
    #data = np.array([[1,2],[2,2.5],[5,4], [2,1],[3,4],[4,7]]).T

    #initialized parameters of the two regimes, in (y = a + b*x1+ c*x2 + d*x3 + e*x4
    # --> 1st component is a, 2nd is b, 3rd is c... ie [a, b, c, d, e]
    regime1_param = [0.0, 0, 0, 0, 0]
    regime2_param = [0.0, 0, 0, 0, 0]

    #membership matrix initialised uniformly
    U = np.random.uniform(0.0, 1.0, [2,number])
    #U = np.array([[1,0],[1,0],[1,0],[0,1],[0,1],[0,1]]).T
    m = 2
    c = 2
    epsilon = 0.0000005
    E = np.random.uniform(0, 1.0, [2,number])
    delta = 0
    training_iter = 0

    def load_data(self):
        self.data = genfromtxt('input/out_original.csv', delimiter=',')
        print 'data:', self.data

    def update_param(self):
        #1  print 'in param_update:'

        X = self.data[:5, :].T
        Y = np.zeros(shape=(1, self.number))
        print 'shape:', Y[0,:].shape, self.data[-1, :].shape
        Y[0, :] = np.array(self.data[-1, :])

        #1  print Y

        for i in range(self.c):
            D = np.zeros(shape = (self.number, self.number))
            for k in range(self.number):
                D[k, k] = copy.copy(pow(self.U[i, k], self.m))

            #1  print 'X and D shape: ',X.shape, D.shape
            if i == 0:

                a = (X.T).dot(D).dot(X)

                #avoid the singular case
                b = np.linalg.pinv(a)

                self.regime1_param = np.dot(b,((X.T).dot(D).dot(Y.T)))
                #1  print i, ": ", np.dot(b,((X.T).dot(D).dot(Y.T)))
            else:
                a = (X.T).dot(D).dot(X)
                b = np.linalg.pinv(a)

                self.regime2_param = np.dot(b,((X.T).dot(D).dot(Y.T)))
                #1  print i,": " , np.dot(b,((X.T).dot(D).dot(Y.T)))

    def train(self):

        U_old = copy.copy(self.U)
        self.update_param()
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

            self.update_param()
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
            #1  print "variables one: ", dep, indep
            #1  print "calculated value E :", math.pow(dep - (np.dot(self.regime1_param.T, indep) + self.delta), 2)

            return math.pow(dep - (np.dot(self.regime1_param.T, indep) + self.delta), 2)
        if regime == 2:
            #1  print "variables two: ", dep, indep
            #1  print "calculated value E :", math.pow(dep - (np.dot(self.regime2_param.T, indep) + self.delta), 2)
            return math.pow(dep - (np.dot(self.regime2_param.T, indep) + self.delta), 2)
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

    sr = SwitchingRegression()

    sr.load_data()

    sr.train()

    #specifying limits
    print 'betas are: ', sr.regime1_param, sr.regime2_param
    print sr.training_iter