# id3 algorithm
# author: yohanes.gultom@gmail.com

import numpy as np
import math
import sys

# id3 data structure
class node(object):
    def __init__(self, value, nodes = None):
        self.value = value
        self.children = nodes if nodes != None else []

    def __repr__(self, level=0):
        ret = "\t"*level+repr(self.value)+"\n"
        for child in self.children:
            ret += child.__repr__(level+1)
        return ret

    def add(self, node):
        self.children.append(node)

    def child(self, i):
        return self.children[i]

    def predict(self, values):
        if len(self.children) <= 0:
            return self.value
        else:
            if self.value in values:
                # attribute node
                val = values[self.value]
                for child in self.children:
                    if child.value == val:
                        return child.predict(values)
            else:
                # value node
                return self.children[0].predict(values)


# calculate entropy
def entropy(dis):
    h = 0
    for p in dis:
        h += -p * math.log(p, 2)
    return h

# probability distributions of array elements
def distributions(arr):
    dis = {}
    n = len(arr)
    for x in arr:
        dis[x] = dis[x] + 1 / float(n) if x in dis else 1 / float(n)
    return dis

# information gain of a column in a rows of dataset
def info_gain(hy, disy, rows, col):
    ig = 0
    # distributions of column col
    dis = distributions(rows[:, col])
    # sum
    for c, p in dis.iteritems():
        # entropy y given column col = class c
        rowsc = rows[rows[:, col] == c]
        disc = distributions(rowsc[:, -1])
        hc = entropy(disc.values())
        ig += p * hc
    return hy - ig

# id3 algorithm using id3 structure
def id3(attrs, rows, y):
    # class entropy
    disy = distributions(y)
    hy = entropy(disy.values())

    # find root
    ig = 0
    root = None
    for i in range(len(attrs)):
        igtmp = info_gain(hy, disy, rows, i)
        if igtmp > ig:
            root = i
            ig = igtmp

    tree = node(attrs[root])

    col = root
    dis = distributions(rows[:, col])
    for c, p in dis.iteritems():
        rowsc = rows[rows[:, col] == c]
        disc = distributions(rowsc[:, -1])
        # check if this attribute class c yields same class y
        # if yes then add the class c as an edge with class y as leaf
        if len(disc) == 1:
            tree.add(node(c, [node(disc.keys()[0])]))
        else:
            if len(attrs) > 1:
                # next interation(s)
                subattrs = np.delete(attrs, root) # remove root attributes
                subrows = np.delete(rows, root, 1) # remove root column
                subrows = subrows[rows[:, col] == c]
                suby = subrows[:, -1]
                tree.add(node(c, [id3(subattrs, subrows, suby)]))
            else:
                tree.add(node(c, [node('?')]))
    return tree

# main program
if __name__ == "__main__":

    # read training and testing data from argv
    train_file = sys.argv[1]
    test_file = sys.argv[2]
    # infile = 'playtennis.data'
    # infile = 'car.data'

    # read input file
    raw = [line.strip().split(',') for line in open(train_file, 'r')]
    data = np.array(raw)

    # build (train) id3
    attrs = data[0, :-1] # attributes
    rows = data[1:, :] # datasets (including results)
    yclass = data[0, -1]
    y = data[1:, -1] # y = result
    tree = id3(attrs, rows, y)

    # print the tree
    print tree

    # read test file
    rawtest = [line.strip().split(',') for line in open(test_file, 'r')]

    # predict
    for i in range(len(rawtest)):
        # skip header
        if i > 0:
            values = {}
            for j in range(len(attrs)):
                values[attrs[j]] = rawtest[i][j]
            print str(values) + ' ' + yclass + ': ' + tree.predict(values)
