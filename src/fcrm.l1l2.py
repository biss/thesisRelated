
__author__ = 'biswajeet'

''' version 1.1: fcrm with L-1 norm
'''

import copy
import csv
import math
import numpy as np
from numpy import genfromtxt
from numpy import linalg as LA

from cvxopt import matrix, solvers


class SwitchingRegression:


    def __init__(self):
        #total number of data points
        self.number = 200
        self.err = 0
        self.crisp_obj = 0

        self.regime_param = [[0.0, 0, 0, 0], [0.0, 0, 0, 0]]
        self.U = genfromtxt('input/u.csv', delimiter=',')
        self.m = 2
        self.c = 2
        self.epsilon = 0.0000005
        self.E = np.zeros(shape=(2, self.number))
        self.delta = 0
        self.training_iter = 0

        self.data = genfromtxt('input/out_n.csv', delimiter=',')
        self.dim, self.number = self.data.shape
        # print 'data:', self.data

        #supress print messages
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

    def newModel(self, A, b, lm, g):
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
            # l-1,l-2 penalty
            I = matrix(np.diag([2*g for i in range(n)]))
            Q = matrix([[2*A.T * A + I, -2*A.T * A - I], [-2*A.T * A - I, 2*A.T * A + I]])

            ones = matrix(1.0*lm, (2*n,1))
            aux = matrix([-2*A.T*b, 2*A.T*b])
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

        X = self.data[:-1, :].T
        Y = self.data[-1, :]

        root_U = np.power(self.U, self.m/2)
        obj = []
        for i in range(self.c):

            D = np.diag(root_U[i, :])

            ''' construct the new A and y matrix'''
            A = matrix(np.dot(D, X))
            y = matrix(np.dot(D, Y))
            if lmd == 0.0:
                sol = self.newModel(A, y, lmd, lmd)
                res = np.array(sol['x'])
            else:
                sol = self.newModel(A, y, lmd, lmd)
                new = np.array(sol['x'])
                res = []
                for j in range(self.dim-1):
                    #print 'hello: ',new[j,:], new[j+6, :]
                    res.append(new[j,:] - new[j+self.dim-1, :])
            self.regime_param[i] = np.array(res).reshape(self.dim-1, 1)

            '''print 'l-1 norm value of ',i+1, 'th model: ', sum(np.absolute(self.regime_param[i]))'''

            obj_value = np.dot(A, self.regime_param[i]) - y
            obj.append(np.dot(obj_value.T, obj_value))

        print 'objective function value: ', obj[0] + obj[1]


    def train(self, lmd):

        #print 'starting with regime params: ', self.regime_param
        #print 'initial U: ', self.U
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
        #print 'ending with regime params: ', self.regime_param
        #print 'final U: ', self.U

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
            #print 'check alginment :', self.regime_param[regime-1].shape, indep.shape
            return math.pow(dep - (np.dot(self.regime_param[regime-1].T, indep) + self.delta), 2)
        if regime == 2:
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
    rms = {}
    no_of_iter = {}

    lmd=[0.0, 1.0,0.5,0.1,0.01]
    for element in lmd:
        print 'for element :', element
        sr = SwitchingRegression()

        sr.train(element)
        sr.fill_err()

        no_of_iter[element] = sr.training_iter

        res[element] = np.array(sr.regime_param)
        rms[element] = [sr.err, sr.crisp_obj, np.matrix.sum(np.matrix(sr.E*sr.U))]
    #print res
    print rms

    with open('output/fcrm_with_l1l2.synthetic.norm.csv', 'wb') as file:
        writer = csv.writer(file, delimiter='\t')
        writer.writerow(res.keys())
        for row in zip(*res.values()):
            for row1 in zip(*row):
                for row2 in zip(*row1):
                    writer.writerow(list(row2))
            writer.writerow('-')