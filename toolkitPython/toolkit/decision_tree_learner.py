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

  def calcInfoS_a(self, a, vc, attr_counts, label_counts): 
    denominator = attr_counts[a, :vc]
    print('denom:', denominator)
    print('denom shape', denominator.shape)
    numerators = []
    for av in range(vc):
      numerators.append(label_counts[:, a][:vc, av])
    numerators = np.array(numerators)
    print('numers:', numerators)
    print('numers shape', numerators.shape)
    p = numerators / denominator[:, None] #this allows you to correctly broadcast denom, even though it is (valCount x null) in shape
    unsummed = np.where(p > 0, p * np.log2(p), 0) #Note: where still runs the np.log part even if p>0, just doesn't return it. So still get warning
    ret = -(unsummed).sum(axis=1)
    print('ret', ret)
    return ret

  def calcEntropyAttributes(self):
    entropy = 0
    attrs = range(self.instances.cols)
    entropy_attrs = np.zeros(len(attrs))
    maxAttr = self.instances.maxValueCount()
    # This will be jagged array of nAttr x (Val count of attr). This is Si in the formula
    attr_counts = np.zeros((len(attrs), maxAttr))
    # This is an array of jagged arrays (of same shape as attr_counts). This is numerator of pi of Info(Si). There is an array of jagged arrays for every possible label
    label_counts = np.zeros((self.labels.value_count(0), len(attrs), maxAttr))
    
    ### Run through instances and count how many of each attribute value there is
    for i in range(self.instances.rows):
      for a in attrs:
        attr_val_i = int(self.instances.get(i, a))
        attr_counts[a][attr_val_i] += 1
        l = int(self.labels.get(i,0))
        label_counts[l][a][attr_val_i] += 1
    for a in attrs:
      vc = self.instances.value_count(a) # all the vc indexing stuff is to take care of garbage columns that were added to avoid jagged arrays
      fraction = (attr_counts[a, :vc] / self.instances.rows)
      entropy_attrs[a] = np.sum(fraction * self.calcInfoS_a(a, vc, attr_counts, label_counts))
    return entropy_attrs
    

  # Recursive algorithm.
  def id3(self):
    if (self.isPureLeafNode() or self.noMoreAttributes()):
      return
    entropy_attrs = self.calcEntropyAttributes()
    print('entropy of attrs\n', entropy_attrs)
    ### Choose attribute A with lowest entropy (or highest gain). Split on A’s possible values
    attrForSplit = np.argmin(entropy_attrs)
    print('Split on {0}'.format(self.instances.attr_name(attrForSplit)))
    input('pause')
    # TODO - do split. Add all nodes from split to this node's children.
    # NOTE: when splitting, must use the init_from f(x) of matrix and then set matrix.data manually.
    # This will keep all the metadata about attributes from the arff file
    ### Make current node = next node from A’s possible values. Do this in loop so that when one node is done doing id3, continues with sibling

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
        
