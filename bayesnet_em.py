"""
# Bayesian Network with EM (Expectation-Maximization)

Usage:
python bayesnet_em.py [model file] [data file] [optional: EM threshold] [optional: EM initial value]

Example:
python bayesnet_em.py flu-alergy.model.json flu-alergy.data.json
python bayesnet_em.py flu-alergy.model.json flu-alergy.data.json 0.0001 0.6


@author yohanes.gultom@gmail.com
"""

import json
import sys
import itertools
import pprint
import math

# initial em value
CONST_E_VALUE = 0.5
CONST_THRESHOLD = 0.0

def generate_rules(model):
    for index, state in model.items():
        values = state['values']
        rules = []
        if 'depends' in state:
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
    for rule in rules:
        calculate_prob(s, rule, data)

def calculate_prob(s, rule, data):
    count = 0
    match = 0
    complete = True
    numerator_predict = None
    denominator_predict = None
    for row in data:
        # check if match given
        match_given = True
        null_given = []
        for key, val in rule['given'].items():
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
                denominator_predict = rule['denominator_predict'] if 'denominator_predict' in rule else CONST_E_VALUE
                count = count + denominator_predict
            # check if match rule
            if row[s] == rule['value']:
                match = match + 1
            # check incomplete
            elif row[s] == None:
                null_given.append(key)
                numerator_predict = rule['numerator_predict'] if 'numerator_predict' in rule else CONST_E_VALUE
                match = match + numerator_predict
            # mark rule incompleteness
            if len(null_given) > 0:
                complete = False
    # calculate probability & mark incompleteness
    rule['probability'] = match / float(count)
    rule['complete'] = complete
    rule['numerator_predict'] = numerator_predict
    rule['denominator_predict'] = denominator_predict


def populate_probs(model, data):
    for id, state in model.items():
        populate_prob(state, data)

def em(model, data):
    sum_diff = 0.0
    found = False
    for id, state in model.items():
        for rule in state['rules']:
            if rule['complete'] == False:
                if 'diff' not in rule or rule['diff'] > CONST_THRESHOLD:
                    found = True if found == False else found
                    # print rule
                    if rule['numerator_predict'] != None:
                        rule['diff'] = math.fabs(rule['numerator_predict'] - rule['probability'])
                        # print('numerator_predict = ' + str(rule['numerator_predict']))
                        # print('probability = ' + str(rule['probability']))
                        # print('diff = ' + str(rule['diff']))
                        rule['numerator_predict'] = rule['probability']
                        sum_diff = sum_diff + rule['diff']
                    if rule['denominator_predict'] != None:
                        rule['diff'] = math.fabs(rule['denominator_predict'] - rule['probability'])
                        # print('denominator_predict = ' + str(rule['denominator_predict']))
                        # print('probability = ' + str(rule['probability']))
                        # print('diff = ' + str(rule['diff']))
                        rule['denominator_predict'] = rule['probability']
                        sum_diff = sum_diff + rule['diff']
                    calculate_prob(id, rule, data)
    print('Total diff = ' + str(sum_diff))
    return sum_diff

def print_incomplete_rules(model):
    for id, state in model.items():
        for rule in state['rules']:
            if rule['complete'] == False:
                rulestr = 'P(' + id + '=' + str(rule['value'])
                if (len(rule['given']) > 0):
                    rulestr = rulestr + ' | '
                    count = 0
                    for given_id, given_val in rule['given'].items():
                        if count > 0:
                            rulestr = rulestr + ', '
                        rulestr = rulestr + given_id + '=' + str(given_val)
                        count = count + 1
                rulestr = rulestr + ')'
                print(rulestr + ' = ' + str(rule['probability']))

if __name__ == "__main__":

    # load data
    with open(sys.argv[1], 'r') as f:
         model = json.load(f)
         # append id
         for id, state in model.items():
             state['id'] = id

    # load data
    with open(sys.argv[2], 'r') as f:
         data = json.load(f)

    if len(sys.argv) >= 4:
        CONST_THRESHOLD = float(sys.argv[3])

    if len(sys.argv) >= 5:
        CONST_E_VALUE = float(sys.argv[4])

    print('CONST_THRESHOLD = ' + str(CONST_THRESHOLD))
    print('CONST_E_VALUE = ' + str(CONST_E_VALUE))

    # generate rules from model
    generate_rules(model)

    # calculate probability from data
    populate_probs(model, data)

    # do em algorithm
    diff = em(model, data)
    while diff > CONST_THRESHOLD:
        diff = em(model, data)

    print_incomplete_rules(model)
