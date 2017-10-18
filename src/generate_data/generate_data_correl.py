__author__ = 'biswajeet'

''' Last Edited: 16th Apr 2016
    generate unnormalized data containing a multicolinear column.
    DATA FORMAT::: each column of data is a data point, [1, x1, x2, x3, x4, x5, y].T is a column
    x1 and x5 are linearly dependent x5 = 2.2*x1 + gaussian noise(0 mean, unit stdev).'''

import numpy as np
from numpy import genfromtxt
from scipy import stats

class SwitchingRegression:

    ''' total number of data points '''
    number = 200

    #each column of data is a data point,[1, x1, x2, x3, x4, y].T is a column,
    data = np.zeros(shape = (9, number), dtype=float)

    #Data generation-----------------------------
    def generateData(self):

        #first regime intercept and slope
        a_1 = 20
        b_1 = 5
        e_1 = 2

        #second regime intercept and slope
        a_2 = 50
        b_2 = 10
        c_2 = -7

        #noise terms to be added to the 3 dependent features
        noise_f6 = np.random.normal(0, 1, self.number/2)
        noise_f7 = np.random.normal(0, 1, self.number/2)
        noise_f8 = np.random.normal(0, 1, self.number/2)

        #feature1, 2 and 5
        feature_1 = [np.random.uniform(-5,5) for i in range(self.number/2)]
        feature_2 = [np.random.uniform(-5,5) for i in range(self.number/2)]
        feature_5 = [np.random.uniform(-5,5) for i in range(self.number/2)]

        feature_6 = list((x + y) for x,y in zip(list( x *2.2 for x in feature_1), list(noise_f6)))
        feature_7 = list((x + y) for x,y in zip(list( x *2.2 for x in feature_2), list(noise_f7)))
        feature_8 = list((x + y) for x,y in zip(list( x *2.2 for x in feature_5), list(noise_f8)))

        x_regime1 = np.array([feature_1,
                              feature_2,
                              [np.random.uniform(-5, 5) for i in range(self.number / 2)],
                              [np.random.uniform(-5, 5) for i in range(self.number / 2)],
                              feature_5,
                              feature_6,
                              feature_7,
                              feature_8,])

        #calculate the dependent variable without noise
        y_axis_regime1 = np.column_stack(np.array([a_1, b_1, 0, 0, e_1, 0, 0, 0])).dot(x_regime1[:, :])

        #adding noise to the dependent variable
        y_axis_regime1 = y_axis_regime1 + np.random.normal(0, 1, self.number/2)

        #concantenating y values
        tmp_data = np.append(x_regime1, y_axis_regime1, axis=0)
        for i in range(self.number/2):
            self.data[:, i] = tmp_data[:, i]

        ''' generate points for second regime
            noise terms to be added to the 3 dependent features'''
        _noise_f6 = np.random.normal(0, 1, self.number / 2)
        _noise_f7 = np.random.normal(0, 1, self.number / 2)
        _noise_f8 = np.random.normal(0, 1, self.number / 2)

        # feature1, 2 and 5
        _feature_1 = [np.random.uniform(-5, 5) for i in range(self.number / 2)]
        _feature_2 = [np.random.uniform(-5, 5) for i in range(self.number / 2)]
        _feature_3 = [np.random.uniform(-5, 5) for i in range(self.number / 2)]

        _feature_6 = list((x + y) for x, y in zip(list(x * 2.2 for x in _feature_1), list(_noise_f6)))
        _feature_7 = list((x + y) for x, y in zip(list(x * 2.2 for x in _feature_2), list(_noise_f7)))
        _feature_8 = list((x + y) for x, y in zip(list(x * 2.2 for x in _feature_3), list(_noise_f8)))

        x_regime2 = np.array([_feature_1,
                              _feature_2,
                              _feature_3,
                              [np.random.uniform(-5, 5) for i in range(self.number / 2)],
                              [np.random.uniform(-5, 5) for i in range(self.number / 2)],
                              _feature_6,
                              _feature_7,
                              _feature_8, ])

        # calculate the dependent variable without noise
        y_axis_regime2 = np.column_stack(np.array([a_2, b_2, c_2, 0, 0, 0, 0, 0])).dot(x_regime2[:, :])

        # adding noise to the dependent variable
        y_axis_regime2 = y_axis_regime2 + np.random.normal(0, 1, self.number / 2)

        # concantenating y values
        tmp_data_2 = np.append(x_regime2, y_axis_regime2, axis=0)
        for i in range(self.number / 2, self.number, 1):
            self.data[:, i] = tmp_data_2[:, i - (self.number / 2)]

        self.data = stats.zscore(self.data, axis=1)

        print 'shape:', self.data.T.shape

        np.savetxt("../input/eightD_n_m.csv", self.data, delimiter=",")

    def load_data(self):
        self.data = genfromtxt('input/eightD_n_m.csv', delimiter=',')
        print self.data

if __name__ == '__main__':
    sr = SwitchingRegression()

    sr.generateData()

    #sr.load_data()

