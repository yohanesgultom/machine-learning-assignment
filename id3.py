import numpy as np

#infile = 'playtennis.data'
infile = 'car.data'
data = [line.strip().split(',') for line in open(infile, 'r')]
arr = np.array(data)

# get first column
print arr[:, 0]
