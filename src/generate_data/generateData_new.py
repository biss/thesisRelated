__author__='biswajit'

''' Last Edited: 16th Apr 2016
    generate switching regression data which switches
    after a certain value of independent variable'''

import numpy as np
from numpy import genfromtxt

class SwitchingRegression:

    ''' total number of data points '''
    number = 200

    #each column of data is a data point,[1, x1, x2, x3, x4, y].T is a column,
    data = np.zeros(shape = (2, number), dtype=float)

    #Data generation-----------------------------
    def generateData(self):

        # first regime intercept and slope
        intercept_1 = 1
        slope_1 = 0.5

        # second regime intercept and slope
        intercept_2 = -2
        slope_2 = 2

        # generate the independent variable randomly
        x_axis_regime1 = np.random.uniform(2.0, 20.0, self.number / 2)
        x_axis_regime2 = np.random.uniform(-10.0, 2.0, self.number / 2)

        # calculate the dependent variable without noise
        y_axis_regime1 = map(lambda x: intercept_1 + slope_1 * x, x_axis_regime1)

        y_axis_regime1 = y_axis_regime1 + np.random.normal(0, 0.7, self.number / 2)

        '''print "unstaged data; ", x_axis_regime1, y_axis_regime1'''

        # adding noise to the dependent variable
        i = 0
        for x_item, y_item in zip(x_axis_regime1, y_axis_regime1):
            self.data[:, i] = [x_item, y_item]
            i += 1

        '''print 'first regime: ', self.data[:, : self.number/2]'''

        # second cluster generate
        y_axis_regime2 = map(lambda x: intercept_2 + slope_2 * x, x_axis_regime2)

        y_axis_regime2 = y_axis_regime2 + np.random.normal(0, 0.9, self.number / 2)

        for x_item, y_item in zip(x_axis_regime2, y_axis_regime2):
            self.data[:, i] = [ x_item, y_item]
            i += 1
        print self.data
        np.savetxt("../input/out_new.csv", self.data, delimiter=",")

    def load_data(self):
        self.data = genfromtxt('input/out_new.csv', delimiter=',')
        print self.data

if __name__ == '__main__':
    sr = SwitchingRegression()

    sr.generateData()