__author__ = 'biswajeet'
import numpy as np

#y = np.array([np.random.uniform(-25.0, 50.0, 4),np.random.uniform(-250.0, 500.0, 4),np.random.uniform(-5.0, 5.0, 4)]).T

'''y = np.array([[17, -198, 3],
                [11, 115 , -1],
                [2, 61, 4],
                [29, -11, -3]])

print np.ones(shape=(1,4)).dot(y)

z = np.ones(shape = (1, 3))

print np.append(y, z, axis=0)'''

import pandas as pd

#read the tag file line by line ---> UserID::MovieID::Tag::Timestamp
'''with open('/home/biswajeet/Documents/movieLens/movie_tags.dat','r') as f:
    next(f) # skip first row
    tag_data = pd.DataFrame(l.rstrip().split('::') for l in f)'''

index = range(10,20,1)

columns = ['A','B', 'C']

data = np.array([np.arange(10)]*3).T

df = pd.DataFrame(data, index=index, columns=columns)

print df

A = np.zeros(shape = (10,10))

for i, j, value in zip(df['A'], df['B'], df['C']):
    print i, j , value
    A[i,j] = value

print A