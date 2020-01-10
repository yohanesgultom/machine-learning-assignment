"""
# Linear Regression

Basic (single variable) linear regression to predict profit. Written based on http://aimotion.blogspot.co.id/2011/10/machine-learning-with-python-linear.html.

Command: python linreg.py

@author yohanes.gultom@gmail.com
"""

from numpy import loadtxt, zeros, ones, array, linspace, logspace, arange
import matplotlib.pyplot as plt


def compute_cost(X, y, theta):
    '''
    Comput cost for linear regression
    '''
    # Number of training samples
    m = y.size

    predictions = X.dot(theta).flatten()

    sqErrors = (predictions - y) ** 2

    J = (1.0 / (2 * m)) * sqErrors.sum()

    return J


def gradient_descent(X, y, theta, alpha, num_iters):
    '''
    Performs gradient descent to learn theta
    by taking num_items gradient steps with learning
    rate alpha
    '''
    m = y.size
    J_history = zeros(shape=(num_iters, 1))

    for i in range(num_iters):

        predictions = X.dot(theta).flatten()

        errors_x1 = (predictions - y) * X[:, 0]
        errors_x2 = (predictions - y) * X[:, 1]

        theta[0][0] = theta[0][0] - alpha * (1.0 / m) * errors_x1.sum()
        theta[1][0] = theta[1][0] - alpha * (1.0 / m) * errors_x2.sum()

        J_history[i, 0] = compute_cost(X, y, theta)

    return theta, J_history


if __name__ == "__main__":

    # Load the dataset
    data = loadtxt('ex1data1.txt', delimiter=',')

    # Extract the data
    X = data[:, 0]
    y = data[:, 1]

    # number of training samples
    m = y.size

    # Add a column of ones to X (interception data)
    it = ones(shape=(m, 2))
    it[:, 1] = X

    # Initialize theta parameters
    theta = zeros(shape=(2, 1))

    # Some gradient descent settings
    iterations = 1500
    alpha = 0.01

    # compute and display initial cost
    print(compute_cost(it, y, theta))

    theta, J_history = gradient_descent(it, y, theta, alpha, iterations)

    print(theta)
    # Predict values for population sizes of 35,000 and 70,000
    predict1 = array([1, 3.5]).dot(theta).flatten()[0]
    print('For population = 35,000, we predict a profit of {:.4f}'.format(predict1 * 10000))
    predict2 = array([1, 7.0]).dot(theta).flatten()[0]
    print('For population = 70,000, we predict a profit of {:.4f}'.format(predict2 * 10000))

    # Plot the results
    result = it.dot(theta).flatten()
    # hold scatter and plot
    f1 = plt.figure(1)
    plt.subplot2grid((1, 1), (0, 0))
    plot_predictions, = plt.plot(data[:, 0], result, label='Predictions')
    plot_data = plt.scatter(X, y, marker='o', c='b', label='Data')
    plt.xlabel('Population of City in 10,000s')
    plt.ylabel('Profit in $10,000s')
    plt.title('Predictions vs Data')
    plt.legend(handles=[plot_predictions, plot_data])
    f1.show()

    f2 = plt.figure(2)
    plt.plot(arange(J_history.shape[0]), J_history, c='r')
    plt.xlabel('Iteration')
    plt.ylabel('Cost (MSE)')
    plt.title('Cost over Iteration')
    f2.show()

    plt.show()
