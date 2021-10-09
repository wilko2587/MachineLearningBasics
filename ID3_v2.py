from node import Node
from copy import deepcopy
import math
import parse
from utilities import most_frequent

_debug_id3 = False
_debug_pruning = False


def debug_ID3(*string):
    if _debug_id3:
        print(*string)
    else:
        pass
    return


def debug_Pruning(*string):
    if _debug_pruning:
        print(*string)
    else:
        pass
    return


def get_entropy(splits):
    entropy = 0  # start with zero and we'll incrementally add to it
    full_size = sum([len(split) for split in splits])  # total number of examples
    epsilon = 1e-5
    for split in splits:
        if len(split) > 0:
            split = [int(i) for i in split]  # again, to fix an inconsistency whether data is integer or string
        sub_sum = len(split)
        sum_0 = split.count(0)
        sum_1 = split.count(1)

        entropy = entropy + sub_sum / full_size * (
                - sum_0 / sub_sum * math.log(sum_0 / sub_sum + epsilon, 2) - sum_1 / sub_sum * math.log(
            sum_1 / sub_sum + epsilon, 2))

    return entropy


def get_examples_split(examples, var):
    '''
    "examples" is a list of dictionaries
    separates 'examples' into two subsets, depending on the value of variable 'var' within each dictionary.
    Also returns the two variables that defines the split
    '''

    vars = list(set([example[var] for example in examples])) #use these three lines to identify the two possible values within 'var'
    if len(vars) == 1:
        vars = vars + ['dummyvar'] #add on "dummyvar" if a second value isn't found. this wont change any functionality

    split0 = [example for example in examples if example[var] == vars[0]]
    split1 = [example for example in examples if example[var] == vars[1]]
    splits = [split0, split1]
    return splits, vars[0], vars[1]


def get_numerical_split(examples, var, target='Class'):
    '''
    "examples" is a list of dictionaries
    separates 'examples' into two subsets, depending on the value of variable 'var' within each dictionary
    '''

    vars = list(set([example[var] for example in examples])) #use these three lines to identify the two possible values within 'var'
    if len(vars) == 1:
        vars = vars + ['dummyvar'] #add on "dummyvar" if a second value isn't found. this wont change any functionality

    split0 = [0 for example in examples if example[var] == vars[0]]
    split1 = [1 for example in examples if example[var] == vars[1]]
    splits = [split0, split1]
    return splits


def get_parent_split(examples, target='Class'):
    '''
    returns the "target" variable within examples in an arbitrary binary format. Eg, with:
    examples = [ {'a':'y', 'b':'n', 'c':'n'} , {'a':'y', 'b':'n', 'c':'y'} , {'a':'y', 'b':'y', 'c':'n' } ]
    target = 'c'
    returns [ [1, 0, 1] ] (reflecting [ 'n', 'y', 'n' ] as the values of c)
    '''
    split = [example[target] for example in examples]
    # here I'm going to convert "split" which can be any set of variables (ie ['y','n','y']) into binary [1,0,1]
    binary_split = [int(i == split[0]) for i in split]
    return [binary_split]


def best_split(examples, target, vars_to_ignore=[]):
    information_gains = {}  # list to contain the entropies of different nodes
    variables = list(examples[0].keys())
    variables.remove(target)  # remove the target variable

    for var in vars_to_ignore:
        variables.remove(var)  # remove the vars_to_ignore

    for var in variables:  # loop through the variables
        parent_split = get_parent_split(examples, target=target)
        children_split = get_numerical_split(examples, var, target=target)

        for count in range(children_split.count([])):  # remove all empty lists from children_split
            children_split.remove([])

        parent_entropy = get_entropy(parent_split)
        children_entropy = get_entropy(children_split)

        information_gain = parent_entropy - children_entropy

        information_gains[information_gain] = var

    maxvalue = max(information_gains.keys())
    bestvariable = information_gains[maxvalue]

    if maxvalue > 0:
        return bestvariable
    else:
        return "Information Gain was Zero!"


def ID3(examples, default, target="Class"):
    '''
  Takes in an array of examples, and returns a tree (an instance of Node)
  trained on the examples.  Each example is a dictionary of attribute:value pairs,
  and the target class variable is a special attribute with the name "Class".
  Any missing attributes are denoted with a value of "?"
  '''

    TopNode = Node('TopNode', default=default)
    TopNode.data = examples  # .data records the subset of examples that are inputted to a node
    fringe = [TopNode]

    while len(fringe) != 0:  # loop until an arbitrary depth limit is reached, or break condition is met
        current_node = fringe.pop(0)
        debug_ID3('current node: ', current_node.label)
        debug_ID3('data: ', current_node.data)
        split_variable = best_split(current_node.data, target,
                                    vars_to_ignore=current_node.prior_variables)  # the best variable to use
        debug_ID3(current_node.label + ' splitting by: ', split_variable)
        # this following if statement will be entered if the current_node is found to be a leaf node (ie: if there's no worthwhile split_variable to be found using information gain)
        if split_variable == "Information Gain was Zero!":
            debug_ID3('No more information gain achieved... Leaf Node Reached!!!')

            # now we need to determine what target the leaf node is supposed to contain. ie: whatever target is most populus
            current_node.label = "Leaf"  # mark the node as a "leaf" node
            most_common_class = most_frequent([example[target] for example in current_node.data], default=default)
            current_node.classification = most_common_class
            debug_ID3('Assigning leaf with Class: ', most_common_class)

            # debug_ID3('tree: ', TopNode.draw())

            if len(fringe) == 0:  # if len(fringe) == 0, it means theres nothing left to explore. This is an end condition
                debug_ID3("Tree fully built")
                return TopNode  # return and exit

        else:
            # if we get down here, current node is not a leaf node (there's a good variable to split the node by)
            current_node.label = split_variable  # rename the current node appropriately
            child_data = get_examples_split(current_node.data, split_variable)  # result of splitting the data

            # now we have the results of splitting at the "current_node" node, we can make the children nodes
            # we don't know what the children nodes will be yet.. so lets call them "childa" and "childb"
            # as placeholders. the subset of examples held in child_data[0] will be fed to childa, child_data[1] will be fed to childb. So we can
            # fill in the "data" attribute with these. We also know their depth is the "current_node" node's depth + 1

            childa = Node('childa', default=default)
            childb = Node('childb', default=default)
            childa.parent = current_node
            childb.parent = current_node
            childa.data = child_data[0]
            childb.data = child_data[1]
            childa.prior_variables = current_node.prior_variables + [
                split_variable]  # we record the prior variables used further up the tree
            childb.prior_variables = current_node.prior_variables + [split_variable]
            childa.parentanswer = 0
            childb.parentanswer = 1

            current_node.children[0] = childa
            current_node.children[1] = childb

            # great. We're there. After the first iteration of this, we have a top node of the tree with the correct
            #  variable to split the data... and we also have set up the next 'depth' of the tree which will be looked at
            #  on the next while-loop iteration.

            # last thing we need to do is update the fringe to contain the children
            fringe = fringe + [childa, childb]
            # debug_ID3('tree: ', TopNode.draw())

    print('WARNING.... shouldnt have got here!')
    return TopNode


# changed the accuracy function a little so that it fits the prune code

def accuracy(score, examples):
    return float(score) / len(examples)


def missing_vals(file):
    dict_original = parse.parse(file)
    dict_mostfreq = most_frequent(dict_original)

    for each in dict_original:
        for all in each:
            if each[all] == '?':
                each[all] = dict_mostfreq[all]

    return dict_original


def evaluate(node, example):
    '''
  Takes in a tree and one example.  Returns the Class value that the tree
  assigns to the example.
  '''

    while True:
        if node.label == "Leaf":  # if the next node is a leaf, we need to return its classification
            return node.classification
        else:
            node = node._evaluate(example)


def test(node, examples):
    '''
  Takes in a trained tree and a test set of examples.  Returns the accuracy (fraction
  of examples the tree classifies correctly).
  '''
    score = 0
    for example in examples:
        our_guess = evaluate(node, example)
        true_value = example['Class']
        score += 1 if int(true_value) == int(our_guess) else 0

    return accuracy(score, examples)


def prune(node, examples, acceptible_accuracy_decline=0.05,
          target="Class"):
    '''
  Takes in a trained tree and a validation set of examples.  Prunes nodes in order
  to improve accuracy on the validation data; the precise pruning strategy is up to you.
  '''

    TopNode = node
    node.data = examples
    debug_Pruning('----')
    debug_Pruning(examples)
    debug_Pruning('----')

    fringe = [node]
    while len(fringe) != 0:
        debug_Pruning(' \n fringe before runthrough: ', fringe)
        currentnode = fringe.pop(0)
        debug_Pruning('currentnode: ', currentnode, currentnode.label)
        if currentnode.label != "Leaf":  # if node is a leaf node, we don't need to prune it by definition
            children = currentnode.children
            for child in list(children.keys()):
                child_examples = currentnode._pass_on_data(child)
                currentnode.children[child].data = child_examples
                tree_accuracy_before = test(TopNode, examples)  # test the whole tree without pruning any children
                child_copy = deepcopy(currentnode.children[
                                          child])  # save a preseved currentnode so we can revert changes if the pruning isn't beneficial
                debug_Pruning(currentnode.label, currentnode.children,
                              [child.label for child in currentnode.children.values()])
                # turn a child into a leaf node
                debug_Pruning('try turning ', child, ' into leaf! ({})'.format(currentnode.children[child].label))
                currentnode.children[child].label = "Leaf"
                currentnode.children[child].leafify("Class")
                tree_accuracy_after = test(TopNode, examples)
                debug_Pruning('change in accuracy: ', tree_accuracy_before, tree_accuracy_after)
                if tree_accuracy_before - tree_accuracy_after < acceptible_accuracy_decline:
                    debug_Pruning("    PRUNED! ", currentnode.label, child)
                else:
                    debug_Pruning('    Not pruning...')
                    currentnode.children[child] = child_copy
                    fringe = fringe + [
                        child_copy]  # if we're not cutting off the node, add it to the fringe to check its children
            debug_Pruning('fringe after runthrough: ', fringe)
    return TopNode

##lets see how it does on the full datasets (i haven't split into train/test sets here...)
# file = 'tennis.data'
# debug_Pruning('running {} set'.format(file))
# examples = parse.parse(file)
# tree = ID3(examples,0)
# tree.draw()
# tree = prune(tree,examples, acceptible_accuracy_decline=0.2)
# print('Pruned!')
# tree.draw()
