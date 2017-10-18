
import numpy as np
from numpy import genfromtxt
from scipy import stats

class SwitchingRegression:

    #total number of data points
    number = 10000
    dim = 8

    raw_data = genfromtxt('../input/cor_syn_data.csv', delimiter=',', skip_header=1).T

    #each column of data is a data point [1, x1, x2, x3, x4, y].T is a column
    data = np.zeros(shape = (dim+1, number), dtype=float)

    #Data generation-----------------------------
    def generateData(self):

        #first regime intercept and slope
        beta_1 = np.array([[3, 1.5, 0, 0, 2, 0, 0, 0]])

        #second regime intercept and slope
        beta_2 = np.array([[0, 4, 3, 1, 0, 2, 0, 0]])

        noise_g = np.random.normal(0, 0.25, self.number/2)
        #noise_x4 = np.random.uniform(-.001, .001, self.number/2)

        #slice the independent variable
        x_regime1 = self.raw_data[:, :self.number/2]

        #calculate the dependent variable without noise
        y_axis_regime1 = beta_1.dot(x_regime1)
        #1  print y_axis_regime1.shape

        #adding noise to the dependent variable
        y_axis_regime1 = y_axis_regime1 + noise_g

        #concantenating y values
        tmp_data = np.append(x_regime1, y_axis_regime1, axis=0)
        for i in range(self.number/2):
            self.data[:, i] = tmp_data[:, i]

        #imp_2 = [np.random.uniform(-25,50) for i in range(self.number/2)]

        # generate second cluster points
        x_regime2 = self.raw_data[:, self.number / 2:]
        print beta_2.shape, x_regime2.shape
        y_axis_regime2 = beta_2.dot(x_regime2)

        #adding noise to the dependent variable
        y_axis_regime2 = y_axis_regime2 +  np.random.normal(0, 0.25, self.number/2)

        tmp_data = np.append(x_regime2, y_axis_regime2, axis=0)
        for i in range(self.number/2, self.number, 1):
            self.data[:, i] = tmp_data[:, i-(self.number/2)]
        #print 'data:', self.data
        print 'shape:', self.data.T.shape

        #z-score normalization
        self.data = stats.zscore(self.data, axis=1)

        np.savetxt("../input/cor_data.csv", self.data, delimiter=",")

    def load_data(self):
        self.data = genfromtxt('../input/cor_data.csv', delimiter=',')
        print self.data

if __name__ == '__main__':

    sr = SwitchingRegression()

    sr.generateData()

    #sr.load_data()
