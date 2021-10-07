from node import Node
import math
import parse

_debug = False

def debug(*string):
  if _debug:
    print(*string)
  else:
    pass
  return


def most_frequent(List):
  counter = 0
  num = List[0]

  for i in List:
    curr_frequency = List.count(i)
    if (curr_frequency > counter):
      counter = curr_frequency
      num = i
  return num

def get_entropy(splits):
    entropy = 0  # start with zero and we'll incrementally add to it
    full_size = sum([len(split) for split in splits])  # total number of examples
    epsilon = 1e-5
    for split in splits:
        sub_sum = len(split)
        sum_0 = split.count('0')
        sum_1 = split.count('1')

        entropy = entropy + sub_sum / full_size * (
                    - sum_0 / sub_sum * math.log(sum_0 / sub_sum + epsilon, 2) - sum_1 / sub_sum * math.log(
                sum_1 / sub_sum + epsilon, 2))

    return entropy


def get_examples_split(data, var):
  split0 = [example for example in data if example[var] == '0']
  split1 = [example for example in data if example[var] == '1']
  splits = [split0, split1]
  return splits


def get_numerical_split(data, var, target='Class'):
    split0 = [example[target] for example in data if example[var] == '0']
    split1 = [example[target] for example in data if example[var] == '1']

    splits = [split0, split1]
    return splits


def get_parent_split(data, target='Class'):
    split = [example[target] for example in data]
    return [split]


def best_split(examples, default, vars_to_ignore = []):
    information_gains = {}  # list to contain the entropies of different nodes
    variables = list(examples[0].keys())
    variables.remove(default)  # remove the target variable

    for var in vars_to_ignore:
        variables.remove(var) # remove the vars_to_ignore

    for var in variables:  # loop through the variables
        debug('    splitting by ',var)
        parent_split = get_parent_split(examples, target=default)
        children_split = get_numerical_split(examples, var, target=default)

        debug('    parent split:', parent_split)
        debug('    children split:', children_split)

        #NB: if a variable doesn't split the data at all (ie: all the examples have Temperature = 1, and we split
        # by temperature, the get_entropy() will break because of a zero-division error. I add in a fix by removing
        # any sublists that are "[]" (ie empty),

        for count in range(children_split.count([])): #remove all empty lists from children_split
            children_split.remove([])

        parent_entropy = get_entropy(parent_split)
        children_entropy = get_entropy(children_split)

        information_gain = parent_entropy - children_entropy

        information_gains[information_gain] = var

    maxvalue = max(information_gains.keys())
    bestvariable = information_gains[maxvalue]

    debug('    --------')
    if maxvalue > 0:
      return bestvariable
    else:
      return "Information Gain was Zero!"


def ID3(examples, default):
  '''
  Takes in an array of examples, and returns a tree (an instance of Node) 
  trained on the examples.  Each example is a dictionary of attribute:value pairs,
  and the target class variable is a special attribute with the name "Class".
  Any missing attributes are denoted with a value of "?"
  '''

  depth = 0 #variable to record the depth we're looking at
  TopNode = Node('TopNode')
  TopNode.data = examples
  TopNode.depth = depth
  tree = {0:[TopNode]} #keys = depth, values = list of nodes at that depth
  end_condition = False

  while depth < 10 and end_condition == False:  # loop until an arbitrary depth limit is reached, or break condition is met
    for current_node in tree[depth]:

      debug('\n=== check1 ===')
      debug('current_node: ',current_node.label)
      debug('current node data: ',current_node.data)
      debug('used prior vars: ',current_node.prior_variables)

      split_variable = best_split(current_node.data, default, vars_to_ignore = current_node.prior_variables)  # the best variable to use

      if split_variable == "Information Gain was Zero!":
        debug('Leaf Node Reached!!!')

        #now we need to determine what target the leaf node is supposed to contain. ie: whatever target is most populus
        current_node.label = "Leaf" #mark the node as a "leaf" node
        most_common_class = most_frequent([example[default] for example in current_node.data])
        current_node.classification = most_common_class

        # if we're at the current max depth of the tree (no new children to move onto)
        # AND if "current_node" is the last node at the current depth to iterate over
        # AND we're in this if-statement
        # Then we're officially finished! the tree is fully built
        max_depth_reached = depth == max(tree.keys()) #this will be True or False
        last_index_reached = True if tree[depth].index(current_node) == len(tree[depth])-1 else False
        if max_depth_reached and last_index_reached:
            debug("Tree fully built")
            return tree #return and exit
        else:
            pass #do nothing... the tree isn't fully built yet.

      else:
        current_node.label = split_variable #rename the current node appropriately
        child_data = get_examples_split(current_node.data, split_variable) #result of splitting the data

        debug('=== check2 ===')
        debug('tree: ',tree)
        debug('current depth: ',depth)
        debug('input data: ',current_node.data)
        debug('split variable: ', split_variable)
        debug('result 0 ', child_data[0])
        debug('result 1 ', child_data[1])

        #now we have the results of splitting at the "current_node" node, we can make the children nodes
        #we don't know what the children nodes will be yet.. so lets call them "childa" and "childb"
        #as placeholders. result[0] will be fed to childa, result[1] will be fed to childb. So we can
        #fill in the "data" attribute with these. We also know their depth is the "current_node" node's depth + 1

        childa = Node('childa')
        childb = Node('childb')
        childa.data = child_data[0]
        childb.data = child_data[1]
        childa.depth = current_node.depth + 1
        childb.depth = current_node.depth + 1
        childa.prior_variables = current_node.prior_variables + [split_variable] #we record the prior variables used further up the tree
        childb.prior_variables = current_node.prior_variables + [split_variable]

        if depth+1 not in tree.keys(): #if this depth level doesn't exist in the tree yet, we create it
          tree[depth+1] = []

        tree[depth+1].append(childa) #add the children to the tree
        tree[depth+1].append(childb)

        current_node.children['0'] = childa
        current_node.children['1'] = childb

        debug('=== check3 ===')
        debug('new tree: ', tree)
        debug('childa data:',childa.data)
        debug('childb data:',childb.data)

        #great. We're there. After the first iteration of this, we have a top node of the tree with the correct
        #  variable to split the data... and we also have set up the next 'depth' of the tree which will be looked at
        #  on the next while-loop iteration.
    depth += 1
  debug('max depth reached!')
  return tree


#changed the accuracy function a little so that it fits the prune code

def accuracy(score, examples):
    return float(score) / len(examples) * 100


# def prune(node, examples):
#   '''
#   Takes in a trained tree and a validation set of examples.  Prunes nodes in order
#   to improve accuracy on the validation data; the precise pruning strategy is up to you.
#   '''
#
#   depth = 0 #variable to record the depth we're looking at
#   TopNode = Node('TopNode')
#   TopNode.data = examples
#   TopNode.depth = depth
#   tree = {0:[TopNode]} #keys = depth, values = list of nodes at that depth
#   end_condition = False
#
#   def prune_node(node, examples):
#     # If leaf node
#       if len(node.child) == 0:  # rename the ".child" after adapting to the ".childa" or ".childb"
#           accuracy_before_pruning = accuracy(score, examples)
#           node.pruned = True
#
#        # If accuracy does not improve, no pruning
#           if accuracy_before_pruning >= accuracy(score, examples):
#               node.pruned = False
#           return
#
#       for value, child_node in node.child.items():
#           prune_node(child_node, val_instances)
#
#     # Prune when we reach the end of the recursion
#       accuracy_before_pruning = accuracy(score, examples)
#       node.pruned = True
#
#       if accuracy_before_pruning >= accuracy(score, examples):
#           node.pruned = False
#
#   return prune_node(node, examples)


def test(node, examples):
   '''
  Takes in a trained tree and a test set of examples.  Returns the accuracy (fraction
  of examples the tree classifies correctly).
  '''

def missing_vals(file):
  dict_original = parse.parse(file)
  dict_mostfreq = most_frequent(dict_original)

  for each in dict_original:
      for all in each:
          if each[all] == '?':
              each[all] = dict_mostfreq[all]

  return dict_original

def evaluate(tree, example):
  '''
  Takes in a tree and one example.  Returns the Class value that the tree
  assigns to the example.
  '''

  node = tree[0][0]
  while True:
      nextnode = node.evaluate(example) #move to the next node
      if nextnode.label == "Leaf": #if the next node is a leaf, we need to return its classification
          return nextnode.classification
      else:
          node = nextnode


#lets see how it does on the full datasets (i haven't split into train/test sets here...)
file = 'tennis.data'
print('running {} set'.format(file))
tree = ID3(parse.parse(file),"Class")
examples = parse.parse(file)
print('example | Class | ID3 Guess')

score = 0

for i in range(len(examples)):
  example = examples[i]
  guess = evaluate(tree, example)
  real = example['Class']
  score += 1 if real == guess else 0
  print(i,'       ',guess,'       ',real)

#print('accuracy: {}%'.format(float(score)/len(examples)*100))

print(accuracy(score, examples)) 

print('''
At the moment ID3() returns a dictionary holding the tree structure + all the nodes. I think looking at the description demeter gave for ID3 (above in the code), its supposed to just return the instance of "Node" which is the top node...'
in theory, this should contain all the information for the tree, so I think maybe get rid of the dictionary "tree" I return,
and just return the top node, then rewrite my evaluate() function above...
      ''')