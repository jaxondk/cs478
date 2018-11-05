from __future__ import (absolute_import, division,
                        print_function, unicode_literals)

from .supervised_learner import SupervisedLearner
from .matrix import Matrix
import numpy as np
import matplotlib.pyplot as plt

class Node():
  name = None # string
  instances = None # Matrix. Features of this node's instance set
  labels = [] # Array. Labels of this node's instance set
  parent = None # Node
  children = [] # Node []
  # availableAttributes = [] # Array of column indices. These features are not already determined
  out = None 

  def __init__(self, name, instances, labels, parent):
    self.name = name
    self.instances = instances
    self.labels = labels
    self.parent = parent
    # self.availableAttributes = availableAttributes

  def addChild(self, child):
    self.children.append(child)

  ### Check for leaf node. If all the labels are equal for the current set, then this is a leaf node
  def isPureLeafNode(self):
    firstLabel = self.labels[0]
    if(all(label == firstLabel for label in self.labels[1:])):
      print(self.name + ' is a leaf node')
      return True
    else:
      return False

  # TODO
  def noMoreAttributes(self):
    return self.instances.cols == 0

  # Calc entropy for entire set at current node
  def calcEntropySet(self, labels):
    classes = list(set(labels))
    entropy = 0
    for i in classes:
      p = labels.count(i)/len(labels)
      entropy -= p * np.log2(p)
    return entropy


  def calcEntropyAttributes(self):
    entropy = 0
    attrs = range(self.instances.cols)
    entropy_attrs = np.zeros(len(attrs))
    counts = np.empty(len(attrs), object)
    ### Run through instances and count how many of each attribute value there is
    for i in range(self.instances.rows):
      for a in attrs:
        if (type(counts[a]) == type(None)):
          counts[a] = np.zeros(self.instances.value_count(a))
        count_i = int(self.instances.get(i, a))
        counts[a][count_i] += 1
    print('Counts matrix: ', counts)
    input('pause')
    

  # Recursive algorithm.
  def id3(self):
    if (self.isPureLeafNode() or self.noMoreAttributes()):
      return
    ### For each attribute available at current node, calc info of attribute
    # info_s = self.calcEntropySet(self.labels)
    info_attrs = self.calcEntropyAttributes()

    ### Choose attribute A with highest info gain. Split on A’s possible values
    ### Make current node = next node from A’s possible values
    # NOTE: when splitting, must use the init_from f(x) of matrix and then set matrix.data manually. 
    # This will keep all the metadata about attributes from the arff file

class DecisionTreeLearner(SupervisedLearner):
    root = None

    def __init__(self):
        pass

    def train(self, instances, labels):
        """
        :type instances: Matrix
        :type labels: Matrix
        """
        # availableAttributes = range(0, len(instances.row(0)))
        self.root = Node('root', instances, labels.col(0), None)
        self.root.id3()

    def predict(self, instances, labels):
        """
        :type instances: [float]
        :type labels: [float]
        """
        
