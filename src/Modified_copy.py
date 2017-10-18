__author__ = 'biswajeet'
__date__ = 'Last Edited: 16th Apr 2016'

''' DESCRIPTION: normal FCRM on un-normalized synthetic data
    DATA FILE used: out_original

    CHANGES:
    1. RMS error is calculated after crispifying the
       membership matrix
'''

import numpy as np
import math
from numpy import linalg as LA
import copy
from numpy import genfromtxt
from sklearn.metrics import mean_squared_error as mse

class SwitchingRegression:

    #total number of data points
    number = 200

    err = 0

    m = 2
    c = 2
    epsilon = 0.0000005
    E = np.random.uniform(0, 1.0, [2, number])
    delta = 0
    training_iter = 0

    # membership matrix initialised uniformly
    U = np.random.uniform(0.0, 1.0, [2, number])

    ''' variable to store the value of the objective function after each iteration.
        This objective value is the sum of objective value of all the c models. '''
    obj = 0

    '''rms error'''
    rms = 0

    ''' stores the sum of absolute values of the models'''
    abs_v = np.zeros(shape=(1, c))

    #each column of data is a data point,[x1, x2, x3, x4, y].T is a column
    data = np.zeros(shape = (6, number), dtype=float)
    #data = np.array([[1,2],[2,2.5],[5,4], [2,1],[3,4],[4,7]]).T

    #initialized parameters of the two regimes, in (y = a + b*x1+ c*x2 + d*x3 + e*x4
    # --> 1st component is a, 2nd is b, 3rd is c... ie [a, b, c, d, e]
    regime1_param = [0.0, 0, 0, 0, 0]
    regime2_param = [0.0, 0, 0, 0, 0]

    def load_data(self):
        self.data = genfromtxt('input/out_m_n.csv', delimiter=',')
        #print 'data:', self.data

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

    def fill_err(self):
        count1 = 0
        count2 = 0
        for i in range(self.number):
            if self.U[0, i] > self.U[1, i]:
                # print i,'in ', 0
                count1 += 1
                self.err += self.E[0, i]
            else:
                # print i,'in ', 1
                count2 += 1
                self.err += self.E[1, i]
        self.err = np.sqrt(self.err / self.number)

    def update_param(self):
        #1  print 'in param_update:'

        X = self.data[:-1, :].T
        Y = np.zeros(shape=(1, self.number))
        Y[0, :] = np.array(self.data[-1, :])

        abs_val = np.zeros(shape=(1, 2))

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

        for j in range(self.c):

            abs_val[:, 0] = np.sum(np.absolute(self.regime1_param))
            abs_val[:, 1] = np.sum(np.absolute(self.regime2_param))

        self.abs_v = abs_val

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


    '''------RMSE calculation------'''
    def rms_calculation(self):
        # crispify the membership matrix
        self.U = self.clusters()

        clus_1, clus_2 = self.set_from_array(self.U)
        #print clus_1, clus_2
        data_1 = np.take(self.data, clus_1, axis=1)
        data_2 = np.take(self.data, clus_2, axis=1)
        #print data_1.shape, data_2.shape
        X_1 = data_1[:-1, :].T
        Y_1 = data_1[-1, :].reshape(data_1.shape[1], 1)
        print X_1.shape, Y_1.shape
        X_2 = data_2[:-1, :].T
        Y_2 = data_2[-1, :].reshape(data_2.shape[1], 1)

        if Y_1.shape[0] == Y_2.shape[0]:
            rms_1 = math.sqrt(mse(np.dot(X_1, self.regime1_param), Y_1) + mse(np.dot(X_2, self.regime2_param), Y_2))
            rms_2 = math.sqrt(mse(np.dot(X_1, self.regime2_param), Y_2) + mse(np.dot(X_2, self.regime1_param), Y_1))
            print rms_1, rms_2

            if (rms_1 > rms_2):
                self.rms = rms_2
            else:
                self.rms = rms_1

        else:
            self.rms = math.sqrt(mse(np.dot(X_1, self.regime1_param), Y_1) + mse(np.dot(X_2, self.regime2_param), Y_2))
            print self.rms

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

    sr.rms_calculation()

    sr.fill_err()

    #specifying limits
    print 'betas are: ', sr.regime1_param, sr.regime2_param
    print sr.training_iter, sr.err