# Bayesian Network with EM Algorithm
# by yohanes.gultom@gmail.com
import json
import sys

if __name__ == "__main__":

    # load data
    with open(sys.argv[1], 'r') as f:
         model = json.load(f)
         # append id
         for id, state in model.iteritems():
             state['id'] = id

    print model
