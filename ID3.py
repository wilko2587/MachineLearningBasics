from node import Node
import math
import parse

def get_entropy(splits):
  entropy = 0 #start with zero and we'll incrementally add to it
  full_size = sum([len(split) for split in splits]) #total number of examples
  epsilon=1e-5
  for split in splits:
    sub_sum = len(split)
    sum_0 = split.count('0')
    sum_1 = split.count('1')

    entropy = entropy + sub_sum/full_size * (- sum_0/sub_sum * math.log(sum_0/sub_sum+epsilon,2) - sum_1/sub_sum * math.log(sum_1/sub_sum+epsilon,2))
  return entropy


def get_splits(data,var,target='Class'):
  split0 = [example[target] for example in data if example[var]=='0']
  split1 = [example[target] for example in data if example[var]=='1']
  splits = [split0,split1]
  return splits


def get_parent_split(data,target='Class'):
  split = [example[target] for example in data]
  return [split]


def best_split(examples,default):
  information_gains = {} #list to contain the entropies of different nodes
  variables = list(examples[0].keys())
  variables.remove(default) #remove the target variable

  for var in variables: #loop through the variables
    parent_split =  get_parent_split(examples,target=default)
    children_split =  get_splits(examples,var,target=default)

    parent_entropy = get_entropy(parent_split)
    children_entropy = get_entropy(children_split)

    information_gain = parent_entropy - children_entropy

    information_gains[information_gain] = var

  maxvalue = max(information_gains.keys())
  bestvariable = information_gains[maxvalue]
  return bestvariable


def ID3(examples, default):
  '''
  Takes in an array of examples, and returns a tree (an instance of Node) 
  trained on the examples.  Each example is a dictionary of attribute:value pairs,
  and the target class variable is a special attribute with the name "Class".
  Any missing attributes are denoted with a value of "?"
  '''

  used_variables = []
  len_variables = len(examples[0].keys()) - 1 #number of variables excluding the target

  while len(used_variables) < len_variables: #loop until all variables are used

    split_variable = best_split(examples,default)

    Node(split_variable)

    used_variables.append(split_variable)


x = ID3(parse.parse('tennis.data'),'Class')



def prune(node, examples):
  '''
  Takes in a trained tree and a validation set of examples.  Prunes nodes in order
  to improve accuracy on the validation data; the precise pruning strategy is up to you.
  '''

def test(node, examples):
  '''
  Takes in a trained tree and a test set of examples.  Returns the accuracy (fraction
  of examples the tree classifies correctly).
  '''


def evaluate(node, example):
  '''
  Takes in a tree and one example.  Returns the Class value that the tree
  assigns to the example.
  '''

