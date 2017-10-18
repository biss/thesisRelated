__author__ = 'biswajeet'

''' Last Edted: 16th Apr 2016
    Generate standardosed data.
    DATA FORMAT::: each column of data is a data point,[x1, x2, x3, x4, y].T is a column.
    Difference from generateDataOri ---> only normalized'''

import numpy as np
from numpy import genfromtxt
from scipy import stats

class SwitchingRegression:

    #total number of data points
    number = 200

    #each column of data is a data point,[1, x1, x2, x3, x4, y].T is a column,
    data = np.zeros(shape = (5, number), dtype=float)

    #Data generation-----------------------------
    def generateData(self):

        #first regime intercept and slope
        a_1 = 20
        b_1 = 5
        c_1 = 0
        d_1 = 2

        #second regime intercept and slope
        a_2 = 50
        b_2 = 10
        c_2 = -7
        d_2 = 0

        noise_g = np.random.normal(0, 1, self.number/2)

        #generate the independent variable randomly 200X4 matrix

        x_regime1 = np.array([[np.random.uniform(-5, 5) for i in range(self.number / 2)],
                              [np.random.uniform(-5, 5) for i in range(self.number / 2)],
                              [np.random.uniform(-5, 5) for i in range(self.number / 2)],
                              [np.random.uniform(-5, 5) for i in range(self.number / 2)]])

        #calculate the dependent variable without noise
        y_axis_regime1 = np.column_stack(np.array([a_1, b_1, c_1, d_1])).dot(x_regime1[:4, :])
        #1  print y_axis_regime1.shape

        #adding noise to the dependent variable
        y_axis_regime1 = y_axis_regime1 + noise_g

        #concantenating y values
        tmp_data = np.append(x_regime1, y_axis_regime1, axis=0)
        for i in range(self.number/2):
            self.data[:, i] = tmp_data[:, i]

        #generate second cluster points

        x_regime2 = np.array([[np.random.uniform(-5, 5) for i in range(self.number / 2)],
                              [np.random.uniform(-5, 5) for i in range(self.number / 2)],
                              [np.random.uniform(-5, 5) for i in range(self.number / 2)],
                              [np.random.uniform(-5, 5) for i in range(self.number / 2)]])

        y_axis_regime2 = np.column_stack(np.array([a_2, b_2, c_2, d_2])).dot(x_regime2[:4, :])

        #adding noise to the dependent variable
        y_axis_regime2 = y_axis_regime2 + np.random.normal(0, 1, self.number/2)

        tmp_data = np.append(x_regime2, y_axis_regime2, axis=0)
        for i in range(self.number/2, self.number, 1):
            self.data[:, i] = tmp_data[:, i-(self.number/2)]

        self.data = stats.zscore(self.data, axis=1)
        #print 'data:', self.data
        print 'shape:', self.data.T.shape

        np.savetxt("../input/out_n.csv", self.data, delimiter=",")

    def load_data(self):
        self.data = genfromtxt('input/out_n.csv', delimiter=',')
        print self.data

if __name__ == '__main__':
    sr = SwitchingRegression()

    sr.generateData()

    #sr.load_data()

