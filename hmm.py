# Hidden Markov Model (HMM)
# by yohanes.gultom@gmail.com
import json
import sys

def max_key(dic):
    max_key = None
    curmax = 0
    for key, val in dic.iteritems():
        if val > curmax:
            curmax = val
            max_key = key
    return max_key

def reach_state(curstate, question, model):
    prob = 1
    t = question['q']['t']
    for i in range(1, t):
        # greedy if not yet arriving at t-1
        next = max_key(curstate['a']) if i < (t-1) else question['q']['state']
        if curstate['a'][next] == None:
            sys.exit('unknown state: ' + next)
        print str(i) + ' : ' + curstate['label'] + ' -> ' + model[next]['label'] + ' ' + str(curstate['a'][next])
        prob = prob * curstate['a'][next]
        curstate = model[next]
    return prob

def reach_emission(curstate, question, model):
    prob = 1
    t = question['o']['t']
    for i in range(1, t+1):
        if i == t:
            # get selected emission prob
            emission = question['o']['state']
            e = curstate['e'][emission]
            if e == None:
                sys.exit('unknown emission: ' + emission)
            print emission + ' ' + str(e)
            prob = prob * e
        else:
            # always greedy
            next = max_key(curstate['a'])
            print str(i) + ' : ' + curstate['label'] + ' -> ' + model[next]['label'] + ' ' + str(curstate['a'][next])
            prob = prob * curstate['a'][next]
            curstate = model[next]
    return prob

def best_path(curstate, question, model):
    prob = 1
    t = question['b']['t']
    best = [ curstate ]
    best_str = ''
    prev_state = None
    for i in range(t):
        # greedy if not yet arriving at t-1
        if i < (t-1):
            next = max_key(curstate['a'])
            prob = prob * curstate['a'][next]
            curstate = model[next]
            best.append(curstate)
            if prev_state != None:
                best_str = best_str + ' -- ' + str(prev_state['a'][curstate['id']]) + ' --> '
            best_str = best_str + curstate['label']
            prev_state = curstate

        else: # if i >= (t-1)
            emission = max_key(curstate['e'])
            p_e = curstate['e'][emission]
            prob = prob * p_e
            best_str = best_str + ' -- ' + str(p_e) + ' --> (' + emission + ')'

    return prob, best_str

# Usage:
# 1. Asking question
# python hmm.py [model file] [first state] [questions]
# Questions:
# -q [transition number] [state]
# -o [transition number] [emission]
#
# Example:
# python hmm.py hakim.json happy -q 2 happy
# python hmm.py hakim.json happy -o 2 frown
# python hmm.py hakim.json happy -q 2 happy -o 2 frown
# python hmm.py hakim.json happy -o 100 yell
#
# 2. Find best path (greedy)
# python hmm.py [model file] [first state] -b [transition number]
#
# Example:
# python hmm.py hakim_e.json happy -b 5

if __name__ == "__main__":

    # load data
    with open(sys.argv[1], 'r') as f:
         model = json.load(f)
         # append id
         for id, state in model.iteritems():
             state['id'] = id

    # first state in model
    first_state = sys.argv[2]

    # read question
    question = { 'q': None, 'o': None, 'b': None}
    for i in range(3, len(sys.argv)):
        if sys.argv[i] == '-q':
            question['q'] = {
                'index': i,
                't' : int(sys.argv[i+1]),
                'state' : sys.argv[i+2]
            }
        elif sys.argv[i] == '-o':
            question['o'] = {
                'index': i,
                't' : int(sys.argv[i+1]),
                'state' : sys.argv[i+2]
            }
        elif sys.argv[i] == '-b':
            question['b'] = {
                't' : int(sys.argv[i+1])
            }

    # answer
    curstate = model[first_state]
    if question['b'] != None:
        prob, best = best_path(curstate, question, model)
        print 'Best path (greedy): ' + best

    elif question['q'] != None and question['o'] != None:
        prob_q = reach_state(curstate, question, model)
        curstate = model[first_state] # reset
        prob_o = reach_emission(curstate, question, model)
        if  question['q']['index'] < question['o']['index']:
            # P(q|o) = P(q).p(o)/p(o)
            prob = prob_q * prob_o / prob_o
        else:
            # P(o|q) = P(q).p(o)/p(q)
            prob = prob_q * prob_o / prob_q

    elif question['q'] != None:
        prob = reach_state(curstate, question, model)

    else: # if question['o'] != None
        prob = reach_emission(curstate, question, model)

    print 'Probability: ' + str(prob)
