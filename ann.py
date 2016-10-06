import numpy as np
import progressbar

# dummy data
X = np.array([[0, 0, 1], [0, 1, 1], [1, 0, 1], [1, 1, 1]])
y = np.array([[0, 1, 1, 0]]).T

# initialize weight of 2 layers
syn0 = 2 * np.random.random((3, 4)) - 1
syn1 = 2 * np.random.random((4, 1)) - 1

# iterations / epochs
epoch = 60000
bar = progressbar.ProgressBar(max_value=epoch)
for j in xrange(epoch):

    # forward propagation
    # using sigmoid activation : 1 / (1 + exp(-Wx))
    l1 = 1 / (1 + np.exp(-(np.dot(X, syn0))))
    l2 = 1 / (1 + np.exp(-(np.dot(l1, syn1))))

    # backpropagation

    # calculate error
    # for each layers
    # using sigmoid derivation: W * (1 - W)
    l2_delta = (y - l2) * (l2 * (1 - l2))
    l1_delta = l2_delta.dot(syn1.T) * (l1 * (1 - l1))

    # update weights
    # of each layers
    syn1 += l1.T.dot(l2_delta)
    syn0 += X.T.dot(l1_delta)

    bar.update(j)

bar.finish()
print ''

# print output layer rounded
print 'Final error (MSE): '
print (l2_delta ** 2).mean()
print ''
print 'Final output: '
print np.around(l2, decimals=1).flatten()
