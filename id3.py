import numpy as np

infile = 'car.data.txt'
data = [line.strip().split(',') for line in open(infile, 'r')]
arr = np.array(data)

# get first column
print arr[:, 0]
