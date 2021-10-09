global depth
from utilities import most_frequent


class Node:
  def __init__(self,label, default = None):
    self.data = None
    self.label = label #label will represent the variable the node splits by
    self.children = {}
    self.prior_variables = []
    self.classification = None
    self.depth = None
    self.default = default
    self.parentanswer = "NA"
    self.parent = None

  def _evaluate(self,example):
    '''
    returns the children node that "example" will be propagated to
    ie: if a node splits on "Humidity", with child "a" receiving humitidites of 0
    and child "b" reveiving humidities of 1, then this method will return an instance of
    child "b" when given an example dictionary of:
    example = {"Temperature":1, "Humidity":1, ... }
    '''

    splitvariable = self.label #we've engineered it so .label stores the split variable of the node (ie: the question the node asks)
    key = example[splitvariable]
    return self.children[int(key)]

  def _pass_on_data(self,childkey):
    '''
    returns the subset of "examples" which will be propagated onto the child
    specified by "childkey". Returns the subset as a list
    '''

    splitvariable = self.label
    sub_examples = [example for example in self.data if example[splitvariable] == childkey]
    return sub_examples

  def leafify(self,var):
    '''
    function to turn a parent node directly into a leaf.
    leaf classification will be decided by the most populated "class" in
    self.data. var is the target variable (usually "Class")
    '''

    self.label = "Leaf"
    targets = [example[var] for example in self.data]
    self.classification = most_frequent(targets, default = self.default)
    return None

  def draw(self):
    '''
    function to illustrate/print the tree below it (children and children's children etc)
    helpful for debugging
    '''

    fringe = [self]
    print('\n -------')
    print('''Parent | Answer  | label  |  child1 label | child2 label |''')
    while len(fringe) != 0:
      currentnode = fringe.pop(0)
      children = list(currentnode.children.values())
      parent = currentnode.parent.label if currentnode.parent != None else "N/A"
      if len(children) != 0:
        print(parent, '|', currentnode.parentanswer, '|', currentnode.label, '|',
              children[0].parentanswer + '_' + children[0].label, '|',
              children[1].parentanswer + '_' + children[1].label, '|')
      else:
        print(parent, '|', currentnode.parentanswer, '|', currentnode.label, '|', "Guess: ", currentnode.classification)

      fringe = fringe + list(children)

    print('------ \n')
