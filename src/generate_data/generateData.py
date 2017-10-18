__author__ = 'biswajeet'

''' Last Edited: 16th Apr 2016
    generate unnormalized data containing a multicolinear column.
    DATA FORMAT::: each column of data is a data point, [1, x1, x2, x3, x4, x5, y].T is a column
    x1 and x5 are linearly dependent x5 = 2.2*x1 + gaussian noise(0 mean, unit stdev).'''

import numpy as np
from numpy import genfromtxt

class SwitchingRegression:

    ''' total number of data points '''
    number = 200

    #each column of data is a data point,[1, x1, x2, x3, x4, y].T is a column,
    data = np.zeros(shape = (7, number), dtype=float)

    #Data generation-----------------------------
    def generateData(self):

        #first regime intercept and slope
        a_1 = 20
        b_1 = 5
        #c_1 = 2
        #d_1 = -2
        e_1 = 2

        #second regime intercept and slope
        a_2 = 50
        b_2 = 10
        c_2 = -7
        #d_2 = -2.12
        #e_2 = 0.82

        noise_g = np.random.normal(0, 1, self.number/2)
        #noise_x4 = np.random.uniform(-.001, .001, self.number/2)

        #generate the independent variable randomly 200X4 matrix
        imp_1 = [np.random.uniform(-5,5) for i in range(self.number/2)]
        #a1 = list(np.random.normal(0.0, 1, self.number/2))
        noise_1 = list((x + y) for x,y in zip(list(x*2.2 for x in imp_1), list(noise_g)))

        x_regime1 = np.array([[1 for i in range(self.number/2)], imp_1,
                              [np.random.uniform(-5,5) for i in range(self.number/2)], [np.random.uniform(-5,5) for i in
                                range(self.number/2)] ,[np.random.uniform(-5,5) for i in range(self.number/2)], noise_1])

        #calculate the dependent variable without noise
        y_axis_regime1 = np.column_stack(np.array([a_1, b_1, 0, 0, e_1])).dot(x_regime1[:5, :])
        #1  print y_axis_regime1.shape

        #adding noise to the dependent variable
        y_axis_regime1 = y_axis_regime1 + np.random.normal(0, 1, self.number/2)

        #concantenating y values
        tmp_data = np.append(x_regime1, y_axis_regime1, axis=0)
        for i in range(self.number/2):
            self.data[:, i] = tmp_data[:, i]

        #generate second cluster points
        imp_2 = [np.random.uniform(-5,5) for i in range(self.number/2)]
        #b1 = list(np.random.normal(0.0,1,self.number/2))
        noise_2 = list((x + y) for x,y in zip(list(x*2.2 for x in imp_2), list(np.random.normal(0, 1, self.number/2))))

        x_regime2 = np.array([[1 for i in range(self.number/2)], imp_2,
                              [np.random.uniform(-5,5) for i in range(self.number/2)], [np.random.uniform(-5,5) for i in
                                range(self.number/2)], [np.random.uniform(-5,5) for i in range(self.number/2)], noise_2])

        y_axis_regime2 = np.column_stack(np.array([a_2, b_2, c_2, 0, 0])).dot(x_regime2[:5, :])

        #adding noise to the dependent variable
        y_axis_regime2 = y_axis_regime2 + np.random.normal(0, 1, self.number/2)

        tmp_data = np.append(x_regime2, y_axis_regime2, axis=0)
        for i in range(self.number/2, self.number, 1):
            self.data[:, i] = tmp_data[:, i-(self.number/2)]
        #print 'data:', self.data
        print 'shape:', self.data.T.shape

        np.savetxt("../input/out_m.csv", self.data, delimiter=",")

    def load_data(self):
        self.data = genfromtxt('input/out_m.csv', delimiter=',')
        print self.data

if __name__ == '__main__':
    sr = SwitchingRegression()

    sr.generateData()

    #sr.load_data()

