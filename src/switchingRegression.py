__author__ = 'biswajeet'

import numpy as np
import matplotlib.pyplot as plt
import math
from numpy import linalg as LA
import copy

class SwitchingRegression:

    number = 400
    #each column of data is a data point
    data = np.zeros(shape = (2, number), dtype=float)
    #data = np.array([[1,2],[2,2.5],[5,4], [2,1],[3,4],[4,7]]).T
    print data

    #initialized parameters of the two regimes, 1st component is intercept, 2nd: slope
    regime1_param = [0, 0]
    regime2_param = [0, 0]

    #membership matrix initialised uniformly
    U = np.random.uniform(0, 1.0, [2,number])
    #U = np.array([[1,0],[1,0],[1,0],[0,1],[0,1],[0,1]]).T
    m = 2
    c = 2
    epsilon = 0.00005
    E = np.random.uniform(0, 1.0, [2,number])
    delta = 0
    training_iter = 0

    #Data generation-----------------------------
    def generateData(self):

        D = np.zeros(shape = (2, self.number))

        for i in range(self.c):
            for k in range(self.number):
                D[i, k] = self.U[i, k]/sum(self.U[:, k])

        self.U = copy.copy(D)

        #first regime intercept and slope
        intercept_1 = 1
        slope_1 = 0

        #second regime intercept and slope
        intercept_2 = 1
        slope_2 = 1

        #generate the independent variable randomly
        x_axis_regime1 = np.random.uniform(-5.0, 5.0, self.number/2)
        x_axis_regime2 = np.random.uniform(-5.0, 5.0, self.number/2)

        #calculate the dependent variable without noise
        y_axis_regime1 = map(lambda x: intercept_1 + slope_1 * x , x_axis_regime1)

        y_axis_regime1 = y_axis_regime1 + np.random.normal(0, 0.25, self.number/2)

        '''print "unstaged data; ", x_axis_regime1, y_axis_regime1'''

        #adding noise to the dependent variable
        i = 0
        for x_item, y_item in zip(x_axis_regime1, y_axis_regime1):
            self.data[:, i] = [x_item, y_item]
            i += 1

        '''print 'first regime: ', self.data[:, : self.number/2]'''

        #second cluster generate
        y_axis_regime2 = map(lambda x: intercept_2 + slope_2 * x , x_axis_regime2)

        y_axis_regime2 = y_axis_regime2 + np.random.normal(0, 0.25, self.number/2)

        for x_item, y_item in zip(x_axis_regime2, y_axis_regime2):
            self.data[:, i] = [x_item, y_item]
            i += 1

    def plotPoints(self):
        '''regime1_data = np.zeros(shape = (2, self.number))
        regime2_data = np.zeros(shape = (2, self.number))'''
        regime1 = []
        regime2 = []
        for i in range(self.number):
            if self.U[0, i] > self.U[1, i]:
                #regime1_data[:, i] = copy.copy(self.data[:, i])
                regime1.append(copy.copy(self.data[:, i]))
            else:
                #regime2_data[:, i] = copy.copy(self.data[:, i])
                regime2.append(copy.copy(self.data[:, i]))
        regime1_data = np.array(regime1).T
        regime2_data = np.array(regime2).T
        color_val = list(self.U[0,:])
        plt.scatter(self.data[0,:], self.data[1,:], s=50, c=color_val)
        plt.gray()
        #print "Shapes: ",regime1_data[0, :].shape, regime1_data[1, :].shape
        #plt.scatter(regime1_data[0, :], regime1_data[1, :], color = 'red')
        #plt.scatter(regime2_data[0, :], regime2_data[1, :], color = 'blue')

    def plotPoints_1(self):
        '''regime1_data = np.zeros(shape = (2, self.number))
        regime2_data = np.zeros(shape = (2, self.number))'''
        regime1 = []
        regime2 = []
        for i in range(self.number):
            if self.U[0, i] > self.U[1, i]:
                # regime1_data[:, i] = copy.copy(self.data[:, i])
                regime1.append(copy.copy(self.data[:, i]))
            else:
                # regime2_data[:, i] = copy.copy(self.data[:, i])
                regime2.append(copy.copy(self.data[:, i]))
        regime1_data = np.array(regime1).T
        regime2_data = np.array(regime2).T
        print "Shapes: ", regime1_data[0, :].shape, regime1_data[1, :].shape
        plt.scatter(regime1_data[0, :], regime1_data[1, :], s=50, color='black')
        plt.scatter(regime2_data[0, :], regime2_data[1, :], s=50,color='black')

        #plt.show()

    def calculate_error(self, regime, dep, indep):

        if regime == 1:
            print "variables one: ", dep, indep
            return math.pow((dep - (self.regime1_param[0] + self.regime1_param[1]*indep + self.delta)), 2)
        if regime == 2:
            print "variables two: ", dep, indep
            return math.pow((dep - (self.regime2_param[0] + self.regime2_param[1]*indep + self.delta)), 2)
        else:
            print "invalid regime"
            return 0

    def update_param(self):

        #constructing the matrix of regressors with 1 as the first column
        X = np.ones(shape = (2, self.number))
        X[1, :] = self.data[0, :]
        Y = np.zeros(shape = (1, self.number))
        Y[0, :] = self.data[1, :]

        for i in range(self.c):
            D = np.zeros(shape = (self.number, self.number))
            for k in range(self.number):
                D[k, k] = copy.copy(pow(self.U[i, k], self.m))
            #print "X.T: ", X.T.shape, "D: ", D.shape, "X: ", X.shape, "Y: ", Y.shape
            if i == 0:

                a = X.dot(D).dot(X.T)
                #avoid the singular case
                b = np.linalg.pinv(a)
                self.regime1_param = np.dot(b,(X.dot(D).dot(Y.T)))
                print i, ": ", np.dot(b,(X.dot(D).dot(Y.T)))
            else:
                a = X.dot(D).dot(X.T)
                b = np.linalg.pinv(a)

                self.regime2_param = np.dot(b,(X.dot(D).dot(Y.T)))
                print i,": " , np.dot(b,(X.dot(D).dot(Y.T)))

    def train(self):
        error = .01

        count = 0
        while error > self.epsilon:
        #while count < 50:
            U_old = copy.copy(self.U)
            print "error: ", self.E
            print "U: ", self.U
            #print "U_old-before update: ", U_old

            self.update_param()
            print 'after update param'
            #self.construct_error_matrix()
            #print 'after error matrix construction'
            self.update_membership()
            print 'after update membership'

            #print "difference matrix: ", Err
            #print "U_new updated: ", self.U
            error = LA.norm(U_old - self.U)
            #print "Error: ", self.E
            #print "COEF: ", self.regime1_param, self.regime2_param

            self.training_iter += 1
            count += 1
        print "final error: ", error

    def update_membership(self):
        self.construct_error_matrix()

        for k in range(self.number):
            if all(self.E[:, k] > 0) == True:
                den = np.zeros(shape=(self.c, 1))
                for i in range(2):
                    den[i, 0] = float(1/self.calc_denominator(i, k))
                    #print "deno: ", den[i, 0]
                #print "updated value", den[:, 0]
                self.U[:, k] = den[:, 0]

            else:
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
                print 'current model: ', self.regime1_param, self.regime2_param, self.data[1, j], self.data[0, j]
                print "calculated value E",i,j," :", (self.data[1, j] - (self.regime1_param[0] + self.regime1_param[1]*
                                                      self.data[0, j] + self.delta))
                self.E[i, j] = self.calculate_error(i+1, self.data[1, j], self.data[0, j])


if __name__ == '__main__':

    sr = SwitchingRegression()

    sr.generateData()

    plt.axis([-5, 5, -5, 5])
    sr.plotPoints_1()
    plt.show()

    sr.train()

    #specifying limits
    plt.axis([-10,20,-40,70])
    x = range(-30, 50)
    y1 = map(lambda w: sr.regime1_param[0] + sr.regime1_param[1] * w, x)
    y2 = map(lambda w: sr.regime2_param[0] + sr.regime2_param[1] * w, x)
    print "final coef: ", sr.regime1_param, sr.regime2_param
    plt.plot(x, y1, color = 'red')
    plt.plot(x, y2, color = 'blue')
    sr.plotPoints()
    plt.show()
    print sr.training_iter

'''
    # in update_param:
        denominator_cluster1 = self.term1(1) * self.term5(1) - self.term3(1)*self.term3(1)
        self.regime1_param = [(self.term5(1)*self.term4(1) - self.term3(1)*self.term2(1))/denominator_cluster1,
                              (self.term1(1)*self.term2(1) - self.term3(1)*self.term4(1))/denominator_cluster1]

        denominator_cluster2 = self.term1(2) * self.term5(2) - self.term3(2)*self.term3(2)
        self.regime2_param = [(self.term5(2)*self.term4(2) - self.term3(2)*self.term2(2))/denominator_cluster2,
                              (self.term1(2)*self.term2(2) - self.term3(2)*self.term4(2))/denominator_cluster2]

    #uik square
    def term1(self, cluster):
        square = map(lambda x: x**2, self.U[cluster - 1, :])
        value = reduce(lambda x, y: x + y, square)
        #print "term1: ", value
        return value

    #uik square x*y
    def term2(self, cluster):
        square = map(lambda x: x**2, self.U[cluster - 1, :])
        value = 0
        for u, x, y in zip(square, self.data[0, :], self.data[1, :]):
            value = value + u*x*y
        #print "term2: ", value
        return value

    #uik square x
    def term3(self, cluster):
        square = map(lambda x: x**2, self.U[cluster - 1, :])
        value = 0
        for u, x in zip(square, self.data[0, :]):
            value = value + u*x
        #print "term3: ", value
        return value

    #uik square y
    def term4(self, cluster):
        square = map(lambda x: x**2, self.U[cluster - 1, :])
        value = 0
        for u, y in zip(square, self.data[1, :]):
            value = value + u*y
        #print "term4: ", value
        return value

    #uik *x square
    def term5(self, cluster):
        terms = map(lambda x: x**2, np.multiply(self.U[cluster - 1, :], self.data[0, :]))
        #print "term5: ", reduce(lambda x, y: x+y, terms)
        return reduce(lambda x, y: x+y, terms)
    '''