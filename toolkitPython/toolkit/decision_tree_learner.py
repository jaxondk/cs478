from __future__ import (absolute_import, division,
                        print_function, unicode_literals)

from .supervised_learner import SupervisedLearner
from .matrix import Matrix
import numpy as np
import matplotlib.pyplot as plt

class Node():
  name = None # string
  features = [] # Matrix. Features of this node's instance set
  labels = [] # Array. Labels of this node's instance set
  parent = None # Node
  children = [] # Node []
  out = None 

  def __init__(self, name, features, labels, parent):
    self.name = name
    self.parent = parent

  def addChild(self, child):
    self.children.append(child)

  ### Check for leaf node. If all the labels are equal for the current set, then this is a leaf node
  def isLeafNode(self):
    firstLabel = self.labels[0]
    if(all(label == firstLabel for label in self.labels[1:])):
      print(self.name + ' is a leaf node')
      return True
    else:
      return False

  # TODO
  def noMoreAttributes(self):
    pass

  # Recursive algorithm.
  def id3(self):
    if (self.isLeafNode() or self.noMoreAttributes()):
      return

    ### Calc entropy for entire set at c
    ### For all attributes available at c:
        ### Calc entropy of attribute
        ### Calc info gain of attribute
    ### Choose attribute A with highest info gain. Split on A’s possible values
    ### Make c = next node from A’s possible values

class DecisionTreeLearner(SupervisedLearner):
    root = None

    def __init__(self):
        pass

    def train(self, features, labels):
        """
        :type features: Matrix
        :type labels: Matrix
        """
        self.root = Node('root', features, labels.col(0), None)
        self.root.id3()

    def predict(self, features, labels):
        """
        :type features: [float]
        :type labels: [float]
        """
        
