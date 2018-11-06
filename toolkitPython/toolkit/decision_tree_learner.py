from __future__ import (absolute_import, division,
                        print_function, unicode_literals)

from .supervised_learner import SupervisedLearner
from .matrix import Matrix
import numpy as np
import matplotlib.pyplot as plt

class Node():
  name = None # string
  attrForSplit = None # index
  avFromSplit = None # attribute value for split
  instances = None # Matrix. Features of this node's instance set
  labels = None # Matrix. Labels of this node's instance set
  parent = None # Node
  children = None # Node dict. key=attribute value for the split, value=Node
  availableAttributes = None # Array of column indices. These features have not been split on yet
  out = None 

  def print(self):
    print('----- Node: {0} -----'.format(self.name))
    print('Applicable patterns: {0}'.format(len(self.instances.data)))
    self.instances.printData(self.labels)
    # self.labels.printData()
    # print('Available attributes: {0}'.format(self.availableAttributes))

  def __init__(self, name, avFromSplit, instances, labels, parent, availableAttributes):
    self.name = name
    self.avFromSplit = avFromSplit
    self.instances = instances
    self.labels = labels
    self.parent = parent
    self.availableAttributes = availableAttributes
    self.children = {}

  def split(self, attrForSplit):
    self.attrForSplit = attrForSplit
    availableAttributes = [a for a in self.availableAttributes if a != attrForSplit]
    for av in range(self.instances.value_count(attrForSplit)):
      name = '{0}={1}'.format(self.instances.attr_name(attrForSplit), self.instances.attr_value(attrForSplit, av))
      row_indices = [r for r in range(self.instances.rows) if self.instances.get(r, attrForSplit) == av]
      instances = self.instances.getSubset(row_indices)
      labels = self.labels.getSubset(row_indices)
      child = Node(name, av, instances, labels, self, availableAttributes)
      self.addChild(child)

  def addChild(self, child):
    self.children[child.avFromSplit] = child

  ### Check for leaf node. If all the labels are equal for the current set, then this is a leaf node
  def isPureLeafNode(self):
    firstLabel = self.labels.get(0,0)
    return all(label == firstLabel for label in self.labels.col(0))

  # TODO
  def noMoreAttributes(self):
    return len(self.availableAttributes) == 0

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
    # print('denom:', denominator)
    # print('denom shape', denominator.shape)
    numerators = []
    for av in range(vc):
      numerators.append(label_counts[:, a][:vc, av])
    numerators = np.array(numerators)
    # print('numers:', numerators)
    # print('numers shape', numerators.shape)
    p = numerators / denominator[:, None] #this allows you to correctly broadcast denom, even though it is (valCount x null) in shape
    unsummed = np.where(p > 0, p * np.log2(p), 0) #Note: where still runs the np.log part even if p>0, just doesn't return it. So still get warning
    ret = -(unsummed).sum(axis=1)
    # print('ret', ret)
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
    self.print()
    # input('pause')
    if (self.isPureLeafNode()):
      self.out = self.labels.get(0, 0)
      print('{0} is a leaf node. Out={1}'.format(self.name, self.out))
      return
    elif (self.noMoreAttributes()):
      self.out = max(self.labels.col(0), key=self.labels.col(0).count) # returns mode of array
      print('{0} has no more attributes to split on. Out={1}'.format(self.name, self.out))
      return
    entropy_attrs = self.calcEntropyAttributes()
    print('entropy of attrs\n', entropy_attrs)
    attrForSplit = np.argmin(entropy_attrs)
    print('Split on {0}'.format(self.instances.attr_name(attrForSplit)))
    self.split(attrForSplit)
    ### Make current node = next node from A’s possible values. Do this in loop so that when one node is done doing id3, continues with sibling
    for child in self.children.values():
      child.id3()
    print('Done with node {0}'.format(self.name))

  # Recursive Function
  def predict(self, instance):
    print('Child for prediction: {0}'.format(self.name))
    # if no children, return self.out as prediction
    if(len(self.children) == 0):
      print('{0} has no children, returning {1} for prediction'.format(self.name, self.out))
      return self.out
    # grab the attribute value for attrForSplit from instance
    av = instance[self.attrForSplit]
    # go to the child who has the same avForSplit and call its predict
    return self.children[av].predict(instance)

class DecisionTreeLearner(SupervisedLearner):
    root = None

    def __init__(self):
        pass

    def train(self, instances, labels):
        """
        :type instances: Matrix
        :type labels: Matrix
        """
        availableAttributes = range(len(instances.row(0)))
        self.root = Node('root', None, instances, labels, None, availableAttributes)
        self.root.id3()

    def predict(self, instance, labels):
        """
        :type instance: [float]
        :type labels: [float]
        """
        del labels[:]
        label = self.root.predict(instance)
        print('Label: ', label)
        labels += [label]

        
