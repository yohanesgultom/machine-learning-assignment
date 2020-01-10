"""
Normal Equation

Reference http://anwarruff.com/normal-equation/

@author yohanes.gultom@gmail.com
"""

import numpy as np
import matplotlib.pyplot as plt


# Load the dataset
data = np.loadtxt('ex1data1.txt', delimiter=',')

# Plot the data
x = data[:, 0]
y = data[:, 1]

# number of training samples
m = y.size

# Add a column of ones to X (interception data)
X = np.ones(shape=(m, 2))
X[:, 1] = x

# normal equation with pseudo inverse
theta = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)

# generate the y axis for the hypothesis function
result = X.dot(theta).flatten()

f1 = plt.figure(1)
plot_predictions, = plt.plot(x, result, label='Predictions')
plot_data = plt.scatter(x, y, marker='o', c='b', label="Data")
plt.xlabel('Population of City in 10,000s')
plt.ylabel('Profit in $10,000s')
plt.title('Predictions vs Data')
plt.legend(handles=[plot_predictions, plot_data])
f1.show()

# f2 = plt.figure(2)
# plt.plot(arange(J_history.shape[0]), J_history, c='r')
# plt.xlabel('Iteration')
# plt.ylabel('Cost (MSE)')
# plt.title('Cost over Iteration')
# f2.show()

plt.show()
