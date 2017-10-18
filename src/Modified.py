__author__ = 'biswajeet'
__date__ = 'Last Edited: 16th Apr 2016'

''' DESCRIPTION: normal FCRM on un-normalized synthetic data
    DATA FILE used: out_original
'''


import numpy as np
import math
from numpy import linalg as LA
import copy
from numpy import genfromtxt

class SwitchingRegression:


    def __init__(self):
        #total number of data points
        self.number = 200

        self.obj_crisp = 0

        #initialized parameters of the two regimes, in (y = a + b*x1+ c*x2 + d*x3 + e*x4
        # --> 1st component is a, 2nd is b, 3rd is c... ie [a, b, c, d, e]
        self.regime1_param = [0.0, 0, 0, 0, 0, 0]
        self.regime2_param = [0.0, 0, 0, 0, 0, 0]

        #membership matrix initialised uniformly
        self.U = np.random.uniform(0.0, 1.0, [2, self.number])
        #self.U = genfromtxt('input/u.csv', delimiter=',')
        #U = np.array([[1,0],[1,0],[1,0],[0,1],[0,1],[0,1]]).T
        self.m = 2
        self.c = 2
        self.epsilon = 0.0000005
        self.E = np.random.uniform(0, 1.0, [2, self.number])
        self.delta = 0
        self.training_iter = 0
        self.err = 0

    def fill_err(self):
        temp = 0
        for i in range(self.number):
            if self.U[0, i] > self.U[1, i]:
                temp += self.U[0, i]*self.E[0, i]
            else:
                temp += self.U[0, i]*self.E[1, i]
        self.obj_crisp = temp
        self.err = np.sqrt(temp / self.number)

    def load_data(self):
        self.data = genfromtxt('input/out_m.csv', delimiter=',')

        #print 'data:', self.data

    def update_param(self):
        #1  print 'in param_update:'

        X = self.data[:-1, :].T
        Y = self.data[-1, :].reshape(1, self.number)

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

        self.training_iter = 1
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
            return math.pow(dep - (np.dot(self.regime2_param.T, indep)  + self.delta), 2)
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
    sr.fill_err()

    #specifying limits
    print 'betas are: ', sr.regime1_param, sr.regime2_param
    print 'iterations: ' ,sr.training_iter
    print 'RMSE: ',sr.err
    print 'fcrm objective value crisp: ', sr.obj_crisp
    print 'fcrm objective value: ', np.matrix.sum(np.matrix(sr.E*sr.U))