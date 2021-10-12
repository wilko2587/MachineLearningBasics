global depth
from utilities import most_frequent


class Node:
  def __init__(self,label, default = None):
    self.data = None
    self.label = label #label will represent the variable the node splits by, "Leaf"
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
    return self.children[key]

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