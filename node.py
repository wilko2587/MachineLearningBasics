class Node:
  def __init__(self,label):
    self.data = None
    self.label = label #label will represent the variable the node splits by
    self.children = {}
    self.prior_variables = []
    self.depth = None
    self.classification = None

  def evaluate(self,example):
    splitvariable = self.label #we've engineered it so .label stores the split variable of the node (ie: the question the node asks)
    key = example[splitvariable]
    return self.children[key]