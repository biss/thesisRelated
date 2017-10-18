
__author__ = 'biswajeet'

''' version 1.1: fcrm with L-1 norm on multicolinear data
    data file used: out_m
'''

import copy
import csv
import math
import numpy as np
from numpy import genfromtxt
from numpy import linalg as LA

from cvxopt import matrix, solvers


class SwitchingRegression:

    def __init__(self, cluster=2):
        #total number of data points
        self.c = cluster

        self.crisp_obj = 0

        #each column of data is a data point,[x1, x2, x3, x4, y].T is a column
        self.data = genfromtxt('input/out_m_n.csv', delimiter=',')
        self.dim, self.number = self.data.shape
        #self.data = self.data / self.data.max(axis=0)

        #membership matrix initialised uniformly
        self.U = np.random.uniform(0.0, 1.0, [self.c, self.number])
        self.regime_param = np.zeros(shape=(self.c, self.dim-1))
        #U = np.array([[1,0],[1,0],[1,0],[0,1],[0,1],[0,1]]).T
        self.m = 2
        self.epsilon = 0.0000005
        self.E = np.random.uniform(0, 1.0, [self.c, self.number])
        self.delta = 0
        self.training_iter = 0
        self.err = 0
        #supress print messages
        solvers.options['show_progress'] = False

    def newModel(self, A, b, lm):
        m, n = A.size

        if lm == 0.0:
            I = matrix(0.0, (n, n))
            I[::n + 1] = 2.0
            Q = matrix(I * A.T * A)
            p = matrix(-I * A.T * b)
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

        X = self.data[:-1, :].T
        Y = self.data[-1, :].reshape(self.number, 1)

        root_U = np.power(self.U, self.m / 2)

        for i in range(self.c):

            D = np.diag(root_U[i, :])

            ''' construct the new A and y matrix'''
            A = matrix(np.dot(D, X))
            y = matrix(np.dot(D, Y))

            sol = self.newModel(A, y, lmd)
            new = np.array(sol['x'])
            print 'new ', new

            if lmd != 0:
                res = []
                for j in range(self.dim-1):
                    res.append(new[j,:] - new[j + self.dim - 1, :])
                print 'de ', self.regime_param[0].shape
                self.regime_param[i, :] = np.array(res).T
            else:
                self.regime_param[i, :] = new.T

            '''print 'l-1 norm value of ',i+1, 'th model: ', sum(np.absolute(self.regime_param[i]))'''

            obj_value = np.dot(A, self.regime_param[i]) - y
            print 'obj shape ', obj_value.shape
            obj = np.dot(obj_value.T, obj_value)

            print 'objective function value: ', obj

    def fill_err(self):
        tmp = 0
        for i in range(self.number):
            if self.U[0, i] > self.U[1, i]:
                tmp += self.U[0, i]*self.E[0, i]
            else:
                tmp += self.U[1, i]*self.E[1, i]
        self.crisp_obj = tmp
        self.err = np.sqrt(tmp / self.number)

    def train(self, lmd):

        U_old = copy.copy(self.U)
        self.update_param(lmd)

        self.update_membership()
        error = LA.norm(U_old - self.U)
        #print '--------------->',error

        '''print 'Em:',self.Em()'''

        self.training_iter = 1

        while error > self.epsilon:

            U_old = copy.copy(self.U)

            self.update_param(lmd)

            self.update_membership()

            error = LA.norm(U_old - self.U)

            self.training_iter += 1

    def update_membership(self):
        self.construct_error_matrix()

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
        return math.pow(dep - (np.dot(self.regime_param[regime-1].T, indep) + self.delta), 2)

    def Em(self):
        error = 0
        for k in range(self.number):
            for i in range(self.c):
                error += pow(self.U[i, k], self.m) * self.E[i, k]
        return error

if __name__ == '__main__':
    cluster = 2
    res = {}
    no_of_iter = {}
    rms = {}
    lmd = [0.0, 0.5,1.0,10.0, 0.01, 0.1, 5.0]

    for element in lmd:
        print 'for element :', element
        sr = SwitchingRegression(cluster)
        sr.train(element)
        sr.fill_err()

        obj = np.matrix.sum(np.matrix(sr.U*sr.E))
        no_of_iter[element] = sr.training_iter
        res[element] = [sr.regime_param, np.array([[obj]]), np.array([[sr.err]]), np.array([[sr.crisp_obj]])]
        rms[element] = sr.err
    print res
    print rms

    with open('output/result.fcrm_with_l1.synthetic.normalized.csv', 'wb') as file:
        writer = csv.writer(file, delimiter='\t')
        writer.writerow(res.keys())
        for row in zip(*res.values()):
            for row1 in zip(*row):
                for row2 in zip(*row1):
                    writer.writerow(list(row2))
                writer.writerow('-')