"""
# Decision Tree ID3
Generative learning decision tree based on Entropy (H) and Information Gain (IG)

## Command: `python id3.py car.data car.test`
Result will be the attributes and predictions at the end of each lines followed by Accuracy & Execution Time:

```
{'LugBoot': 'big', 'Maint': 'low', 'Persons': 'more', 'Safety': 'low', 'Doors': '5more', 'Buying': 'low'} Quality: unacc (True)
{'LugBoot': 'big', 'Maint': 'low', 'Persons': 'more', 'Safety': 'med', 'Doors': '5more', 'Buying': 'low'} Quality: acc (False)
{'LugBoot': 'big', 'Maint': 'low', 'Persons': 'more', 'Safety': 'high', 'Doors': '5more', 'Buying': 'low'} Quality: vgood (True)
Accuracy: 77.1929824561%
Execution Time: 0.871682882309s
```

## Command: `python id3.py car.data car.test --plot`
Add option `--plot` to plot the decision tree in png format. The result is saved in the same directory with file name pattern eg. car.data.png

@author yohanes.gultom@gmail.com
"""

import numpy as np
import math
import sys
import time
import pydot
import ntpath

# id3 data structure
class Node(object):

    graph = pydot.Dot(graph_type='graph')
    counter = 0

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
            # value node
            # or if no path just take the first child
            return self.children[0].predict(values)

    def draw_to_file(self, filename = 'id3', path = ''):
        for child in self.children:
            child.draw(self.value)
        Node.graph.write_png(filename + '.png')
        print('Tree plotted in {}.png'.format(filename))

    def draw(self, parent, label = None):
        if label != None:
            # draw node and edge
            node_id = str(Node.counter) + self.value
            node = pydot.Node(node_id, label=self.value)
            Node.graph.add_node(node)
            edge = pydot.Edge(parent, node)
            edge.set_label(label)
            Node.graph.add_edge(edge)

            # process child nodes
            if len(self.children) > 0:
                for child in self.children:
                    child.draw(node)
            Node.counter += 1
        else:
            # skip edge (value node) but pass parent and label
            self.children[0].draw(parent, self.value)

# id3 algorithm using id3 structure
class ID3(object):

    def __init__(self, attrs, rows, y, coly):
        self.attrs = attrs # array of attributes (exclude y)
        self.rows = rows # data rows (incl y)
        self.y = y # y rows
        self.coly = coly # column position of y
        self.disy = {} # distribution of y
        self.hy = 0 # y class entropy

    def tree(self):
        # class entropy
        self.disy = self.distributions(self.y)
        self.hy = self.entropy(self.disy.values())

        # find root
        ig = 0
        root = None
        for i in range(len(self.attrs)):
            igtmp = self.info_gain(i)
            if igtmp > ig:
                root = i
                ig = igtmp

        tree = Node(self.attrs[root])

        col = root
        dis = self.distributions(self.rows[:, col])
        for c, p in dis.items():
            rowsc = self.rows[self.rows[:, col] == c]
            disc = self.distributions(rowsc[:, self.coly])
            # check if this attribute class c yields same class y
            # if yes then add the class c as an edge with class y as leaf
            if len(disc) == 1:
                class_y = list(disc.keys())[0]
                tree.add(Node(c, [Node(class_y)]))
            else:
                if len(attrs) > 1:
                    # next interation(s)
                    subattrs = np.delete(self.attrs, root) # remove root attributes
                    subrows = np.delete(self.rows, root, 1) # remove root column
                    subrows = subrows[self.rows[:, col] == c]
                    suby = subrows[:, self.coly]
                    tree.add(Node(c, [ID3(subattrs, subrows, suby, self.coly).tree()]))
                else:
                    # find highest probability class
                    max_c = max(disc, key=disc.get)
                    tree.add(Node(c, [Node(max_c)]))
        return tree

    # information gain of a column in a rows of dataset
    def info_gain(self, col):
        ig = 0
        # distributions of column col
        dis = self.distributions(self.rows[:, col])
        # sum
        for c, p in dis.items():
            # entropy y given column col = class c
            rowsc = self.rows[self.rows[:, col] == c]
            disc = self.distributions(rowsc[:, self.coly])
            hc = self.entropy(disc.values())
            ig += p * hc
        return self.hy - ig

    # calculate entropy
    @staticmethod
    def entropy(dis):
        h = 0
        for p in dis:
            h += -p * math.log(p, 2)
        return h

    # probability distributions of array elements
    @staticmethod
    def distributions(arr):
        dis = {}
        n = len(arr)        
        for x in arr:
            dis[x] = dis[x] + 1 / n if x in dis else 1 / n
        return dis

# extract filename from path regardless of OS
def path_leaf(path):
    head, tail = ntpath.split(path)
    return tail or ntpath.basename(head)

# main program
if __name__ == "__main__":
    execution_time = -time.time()

    # read training and testing data from argv
    train_file = sys.argv[1]
    test_file = sys.argv[2]
    plot = True if len(sys.argv) >= 4 and sys.argv[3] == '--plot' else False # draw tree or not


    # read input file
    raw = [line.strip().split(',') for line in open(train_file, 'r')]
    data = np.array(raw)

    # build (train) id3
    coly = -1
    attrs = data[0, :coly] # attributes
    rows = data[1:, :] # datasets (including results)
    yclass = data[0, coly]
    y = data[1:, coly] # y = result
    tree = ID3(attrs, rows, y, coly).tree()

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
            prediction = tree.predict(values)
            correct = (prediction == rawtest[i][-1])
            total_correct += 1 if correct else 0
            print('{} {}: {} ({})'.format(values, yclass, prediction, correct))

    accuracy = total_correct / float(len(rawtest)-1) * 100
    execution_time += time.time()
    print('Accuracy: {}%'.format(accuracy))
    print('Execution Time: {}s'.format(execution_time))

    # print the tree
    # print tree
    if plot:
        tree.draw_to_file(train_file)

