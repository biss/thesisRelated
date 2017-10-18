import numpy as np
import csv

if __name__ == '__main__':
    number = 804
    c = 2
    U = np.random.uniform(0.0, 1.0, [c, number])

    np.savetxt("../input/u_.csv", U, delimiter=",")