# Join-distribution bayes classifier algorithm
# author: yohanes.gultom@gmail.com

import numpy as np
import math
import sys
import time
import pprint

# NBC class
class JBC(object):

    def __init__(self, attrs, rows, coly):
        self.attrs = attrs
        self.rows = rows
        self.coly = coly
        self.truth_table_join = {}
        self.truth_table_ind = {}
        self.default_prediction = None
        # build truth table
        self.build_truth_table()

    def build_truth_table(self):
        count_row = {}
        count_col = {}
        num_rows = len(self.rows)
        for row in self.rows:
            # count join attr
            key = '#'.join(row)
            count_row[key] = count_row[key] + 1 / float(num_rows) if key in count_row else 1 / float(num_rows)
            # count each attr values
            for i in range(len(row)):
                attr = self.attrs[i]
                if attr not in count_col:
                    count_col[attr] = {}
                if row[i] in count_col[attr]:
                    count_col[attr][row[i]] += 1 / float(num_rows)
                else:
                    count_col[attr][row[i]] = 1 / float(num_rows)
        self.truth_table_join = count_row
        self.truth_table_ind = count_col

        # get default prediction (y class with highest probability)
        classy = self.attrs.tolist()[self.coly]
        maxp = 0
        for valy, pvaly in self.truth_table_ind[classy].iteritems():
            if pvaly > maxp:
                maxp = pvaly
                self.default_prediction = valy

    def predict(self, values):
        res = None
        maxp = 0
        classy = self.attrs.tolist()[self.coly]
        base_key = '#'.join([values[attr] for attr in self.attrs[:-1]])
        # get max likelihood from all class y
        for valy, pvaly in self.truth_table_ind[classy].iteritems():
            key = base_key+'#'+valy
            if key in self.truth_table_join:
                # get join probability from truth table
                p = self.truth_table_join[key]
            else:
                # if the join probability not yet exist
                # then it is 1 / number of dataset
                p = 0
            if p > maxp:
                maxp = p
                res = valy
        # no match then return default prediction
        return res if res != None else self.default_prediction

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
    attrs = data[0] # attributes (include result class)
    rows = data[1:, :] # datasets (including results)
    yclass = data[0, coly]

    jbc = JBC(attrs, rows, coly)

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
            prediction = jbc.predict(values)
            correct = (prediction == rawtest[i][coly])
            total_correct += 1 if correct else 0
            print str(values) + ' ' + yclass + ': ' + str(prediction) + ' (' + str(correct) + ')'

    accuracy = total_correct / float(len(rawtest)-1) * 100
    execution_time += time.time()
    print 'Accuracy: ' + str(accuracy) + '%'
    print 'Execution Time: ' + str(execution_time) + 's'
