__author__ = 'biswajeet'
__date__ = 'Last Edited: 16th Apr 2016'

''' DESCRIPTION: normal FCRM on un-normalized synthetic data
    DATA FILE used: out_original
'''


import numpy as np
import math, csv
from numpy import linalg as LA
import copy
from numpy import genfromtxt

class SwitchingRegression:

    def __init__(self, path, u):
        # total number of data points
        self.data = genfromtxt(path, delimiter=',')

        #self.data = self.data / self.data.max(axis=0)
        self.crisp_obj = 0

        self.dim, self.number = self.data.shape
        # initialized parameters of the two regimes, in (y = a + b*x1+ c*x2 + d*x3 + e*x4
        # --> 1st component is a, 2nd is b, 3rd is c... ie [a, b, c, d, e]
        self.regime_param = np.array([[0.0 for i in range(self.dim - 1)], [0.0 for i in range(self.dim - 1)]])
        #self.regime1_param = [0.0, 0, 0, 0, 0, 0]

        #self.regime2_param = [0.0, 0, 0, 0, 0, 0]

        # membership matrix initialised uniformly

        self.m = 2
        self.c = 2
        self.epsilon = 0.0000005
        self.E = np.random.uniform(0, 1.0, [self.c, self.number])
        self.delta = 0
        self.training_iter = 0
        self.U = u
        self.err = 0

    def update_param(self):
        #1  print 'in param_update:'

        X = self.data[:-1, :].T
        Y = self.data[-1, :].reshape(1, self.number)

        #1  print Y

        for i in range(self.c):
            D = np.zeros(shape = (self.number, self.number))
            for k in range(self.number):
                D[k, k] = copy.copy(pow(self.U[i, k], self.m))

            #print 'X and D shape: ',X.shape, D.shape
            '''
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
            '''
            a = (X.T).dot(D).dot(X)

            # avoid the singular case
            b = np.linalg.pinv(a)
            self.regime_param[i] = np.dot(b, ((X.T).dot(D).dot(Y.T))).reshape(self.dim-1,)

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
            if all(self.E[:, k] > 0):
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

                self.E[i, j] = self.calculate_error(i, self.data[-1, j], self.data[:-1, j])

    def calculate_error(self, regime, dep, indep):

        return math.pow(dep - (np.dot(self.regime_param[regime].reshape(1, self.dim-1), indep.reshape(self.dim-1,1))  + self.delta), 2)

    def Em(self):
        error = 0
        for k in range(self.number):
            for i in range(self.c):
                error += pow(self.U[i, k], self.m) * self.E[i, k]
        return error

    def fill_err(self):
        d = 0
        for i in range(self.number):
            if self.U[0, i] > self.U[1, i]:
                self.err += self.E[0, i]*self.U[0, i]
                d += self.E[0, i]
            else:
                self.err += self.E[1, i]*self.U[1, i]
                d += self.E[1, i]
        self.crisp_obj = d
        self.err = np.sqrt(self.err / self.number)

if __name__ == '__main__':
    beta = {}
    for i in range(1):
        u = np.random.uniform(0, 1.0, [2, 200])
        sr = SwitchingRegression('input/out_n.csv', u)

        #sr.load_data()

        sr.train()
        sr.fill_err()

        #specifying limits
        print sr.U
        print 'betas are: ', sr.regime_param
        print sr.crisp_obj, sr.err, np.matrix.sum(np.matrix(sr.E*sr.U))
        beta[i] = sr.regime_param
    #print sr.regime_param

    with open('result_output/ordinary_fcrm.csv', 'wb') as file:
        writer = csv.writer(file, delimiter='\t')
        writer.writerow(beta.keys())
        writer.writerow('-')
        for row in zip(*beta.values()):
            for row2 in zip(*row):
                writer.writerow(list(row2))
            writer.writerow('-')