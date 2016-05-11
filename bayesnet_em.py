# Bayesian Network with EM Algorithm
# by yohanes.gultom@gmail.com
import json
import sys
import itertools
import pprint

def generate_rules(model):
    for index, state in model.iteritems():
        values = state['values']
        rules = []
        if state.has_key('depends'):
            givens = combinations(state['depends'], model)
            for val in values:
                for given in givens:
                    rules.append({ 'value' : val, 'probability' : None, 'given' : given })
        else:
            for val in values:
                rules.append({ 'value' : val, 'probability' : None, 'given' : {} })
        state['rules'] = rules

def combinations(ids, model):
    head = model[ids[0]]
    other_ids = ids[1:]
    comb = []
    for val in head['values']:
        if len(other_ids) <= 0:
            comb.append({ head['id'] : val })
        else:
            for other_id in other_ids:
                other = model[other_id]
                for other_val in other['values']:
                    comb.append({ head['id'] : val, other_id : other_val })
    return comb

def populate_prob(state, data):
    s = state['id']
    rules = state['rules']
    incompletes = []
    for rule in rules:
        count = 0
        match = 0
        complete = True
        for row in data:
            # check if match given
            match_given = True
            null_given = []
            pprint.pprint(rule)
            for key, val in rule['given'].iteritems():
                # check incomplete
                if row[key] == None:
                    # match_given = match_given and True
                    null_given.append(key)
                else:
                    match_given = match_given and (row[key] == val)
            # process only if matches given
            if match_given:
                # count total
                if len(null_given) == 0:
                    count = count + 1
                else:
                    count = count + 0.5
                # check if match rule
                if row[s] == rule['value']:
                    match = match + 1
                # check incomplete
                elif row[s] == None:
                    null_given.append(key)
                    match = match + 1 / float(len(state['values']))
                # mark rule incompleteness
                if len(null_given) > 0:
                    complete = False
        # calculate probability & mark incompleteness
        rule['probability'] = match / float(count)
        rule['complete'] = complete

def populate_probs(model, data):
    for id, state in model.iteritems():
        populate_prob(state, data)

def em(model):
    for s in model:
        print s['id']

if __name__ == "__main__":

    # load data
    with open(sys.argv[1], 'r') as f:
         model = json.load(f)
         # append id
         for id, state in model.iteritems():
             state['id'] = id

    # load data
    with open(sys.argv[2], 'r') as f:
         data = json.load(f)

    generate_rules(model)
    populate_probs(model, data)
    em(model)
    # pprint.pprint(model)
