from __future__ import (absolute_import, division,
                        print_function, unicode_literals)

from .supervised_learner import SupervisedLearner
from .matrix import Matrix
import numpy as np
import matplotlib.pyplot as plt

class Node():
  name = None # string
  instances = None # Matrix. Features of this node's instance set
  labels = None # Matrix. Labels of this node's instance set
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
    firstLabel = self.labels.get(0,0)
    if(all(label == firstLabel for label in self.labels.col(0))):
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
    # This will be jagged array of nAttr x (Val count of attr). This is Si in the formula
    instance_counts = np.empty(len(attrs), object)
    # This is an array of jagged arrays (of same shape as instance_counts). This is numerator of pi of Info(Si)
    # There is an array of jagged arrays for every possible label
    label_counts = np.empty((self.labels.value_count(0), len(attrs)), object)
    # print('instance_counts matrix: ', instance_counts)
    # print('label_counts matrix: ', label_counts)
    ### Run through instances and count how many of each attribute value there is
    for i in range(self.instances.rows):
      for a in attrs:
        if (type(instance_counts[a]) == type(None)):
          instance_counts[a] = np.zeros(self.instances.value_count(a))
        attr_val_i = int(self.instances.get(i, a))
        instance_counts[a][attr_val_i] += 1
        l = int(self.labels.get(i,0))
        if (type(label_counts[l][a]) == type(None)):
          label_counts[l][a] = np.zeros(self.instances.value_count(a))
        label_counts[l][a][attr_val_i] += 1
    print('instance_counts matrix: ', instance_counts)
    print('label_counts matrix: ', label_counts)
    input('pause')
    # entropy_attrs = (instance_counts / self.instances.rows) * 


    

  # Recursive algorithm.
  def id3(self):
    if (self.isPureLeafNode() or self.noMoreAttributes()):
      return
    ### For each attribute available at current node, calc info of attribute
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
        self.root = Node('root', instances, labels, None)
        self.root.id3()

    def predict(self, instances, labels):
        """
        :type instances: [float]
        :type labels: [float]
        """
        
