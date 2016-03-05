# Naive bayes classifier algorithm
# author: yohanes.gultom@gmail.com

import numpy as np
import math
import sys
import time

# NBC class
class NBC(object):

    def __init__(self, attrs, rows, coly):
        self.attrs = attrs
        self.rows = rows
        self.coly = coly
        self.disy = self.distributions(self.rows[:, self.coly])
        self.p = {} # probability cache

    def predict(self, values):
        res = None
        maxp = 0
        for valy, pvaly in self.disy.iteritems():
            p = pvaly
            for c, val in values.iteritems():
                # p(c = val | y = cy)
                p *= self.probability(c, val, valy)
            # choose max p
            if p > maxp:
                maxp = p
                res = valy
        return res

    def probability(self, c, val, valy):
        cache_key = c + '#' + val + '#' + valy
        if cache_key in self.p:
            # retrieve from cache
            return self.p[cache_key]
        else:
            # rows where coly = valy
            rowsvaly = self.rows[self.rows[:, self.coly] == valy]
            # rowsvaly where c = val / len(rowsvaly)
            indexc = self.attrs.tolist().index(c)
            res = len(rowsvaly[rowsvaly[:, indexc] == val]) / float(len(rowsvaly))
            # store in the cache
            self.p[cache_key] = res
            return res

    # probability distributions of array elements
    @staticmethod
    def distributions(arr):
        dis = {}
        n = len(arr)
        for x in arr:
            dis[x] = dis[x] + 1 / float(n) if x in dis else 1 / float(n)
        return dis


# main program
if __name__ == "__main__":
    execution_time = -time.time()

    # read training and testing data from argv
    train_file = sys.argv[1]
    test_file = sys.argv[2]

    # read input file
    raw = [line.strip().split(',') for line in open(train_file, 'r')]
    data = np.array(raw)

    # setup NBC
    coly = -1
    attrs = data[0, :coly] # attributes
    rows = data[1:, :] # datasets (including results)
    yclass = data[0, coly]

    nbc = NBC(attrs, rows, coly)

    # read test file
    rawtest = [line.strip().split(',') for line in open(test_file, 'r')]

    # predict
    total_correct = 0
    for i in range(len(rawtest)):
        # skip header
        if i > 0:
            values = {}
            for j in range(len(attrs)):
                values[attrs[j]] = rawtest[i][j]
            prediction = nbc.predict(values)
            correct = (prediction == rawtest[i][coly])
            total_correct += 1 if correct else 0
            print str(values) + ' ' + yclass + ': ' + str(prediction) + ' (' + str(correct) + ')'

    accuracy = total_correct / float(len(rawtest)-1) * 100
    execution_time += time.time()
    print 'Accuracy: ' + str(accuracy) + '%'
    print 'Execution Time: ' + str(execution_time) + 's'
