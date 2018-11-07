from __future__ import (absolute_import, division,
                        print_function, unicode_literals)

from .supervised_learner import SupervisedLearner
from .matrix import Matrix
import numpy as np
import matplotlib.pyplot as plt

class Node():
  ###### NOTE: these are CLASS variables. they are different than self.VARNAME.
  # name = None # string
  # attrForSplit = None # index
  # avFromSplit = None # attribute value for split
  # instances = None # Matrix. Features of this node's instance set
  # labels = None # Matrix. Labels of this node's instance set
  # parent = None # Node
  # children = None # Node dict. key=attribute value for the split, value=Node
  # availableAttributes = None # Array of column indices. These features have not been split on yet
  # out = None 

  def print(self):
    print('----- Node: {0} -----'.format(self.name))
    print('Applicable patterns: {0}'.format(len(self.instances.data)))
    self.instances.printData(self.labels)

  def __init__(self, name, avFromSplit, instances, labels, parent, availableAttributes):
    self.name = name
    self.avFromSplit = avFromSplit
    self.instances = instances
    self.labels = labels
    self.parent = parent
    self.children = {}
    self.availableAttributes = availableAttributes

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

  def calcInfoS_a(self, a, vc, attr_counts, label_counts): 
    denominator = attr_counts[a, :vc]
    numerators = []
    for av in range(vc):
      numerators.append(label_counts[:, a][:vc, av])
    numerators = np.array(numerators)
    p = numerators / denominator[:, None] #this allows you to correctly broadcast denom, even though it is (valCount x null) in shape
    unsummed = np.where(p > 0, p * np.log2(p), 0) #Note: where still runs the np.log part even if p>0, just doesn't return it. So still get warning
    return -(unsummed).sum(axis=1)

  def calcEntropyAttributes(self):
    entropy = 0
    attrs = range(self.instances.cols)
    entropy_attrs = np.zeros(len(attrs))
    maxAttr = self.instances.maxValueCount()
    # This will be  array of nAttr x (maxValCount of attrs). This is Si in the formula
    attr_counts = np.zeros((len(attrs), maxAttr))
    # This is an array of arrays (of same shape as attr_counts). This is numerator of pi of Info(Si). There is an array of arrays for every possible label
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
      infoS_a = self.calcInfoS_a(a, vc, attr_counts, label_counts)
      entropy_attrs[a] = np.sum(fraction * infoS_a)
    return entropy_attrs

  # Recursive algorithm.
  def id3(self):
    # self.print()
    if(self.instances.rows == 0):
      self.out = self.parent.labels.most_common_value(0)
      # print('No instances for {0}. Out={1}'.format(self.name, self.out))
      return
    elif (self.isPureLeafNode()):
      self.out = self.labels.get(0, 0)
      # print('{0} is a leaf node. Out={1}'.format(self.name, self.out))
      return
    elif (self.noMoreAttributes()):
      self.out = self.labels.most_common_value(0)
      # print('{0} has no more attributes to split on. Out={1}'.format(self.name, self.out))
      return
    entropy_attrs = self.calcEntropyAttributes()
    attrForSplit = np.argmin(entropy_attrs)
    self.split(attrForSplit)
    ### Make current node = next node from A’s possible values. Do this in loop so that when one node is done doing id3, continues with sibling
    for child in self.children.values():
      child.id3()

  # Recursive Function
  def predict(self, instance):
    # if no children, return self.out as prediction
    if(len(self.children) == 0):
      return self.out
    # grab the attribute value for attrForSplit from instance
    av = instance[self.attrForSplit]
    # go to the child who has the same avForSplit and call its predict
    return self.children[av].predict(instance)

class DecisionTreeLearner(SupervisedLearner):
    def __init__(self):
        pass

    def fillMissingValues(self, instances):
      missing_val = float('Inf')
      for c in range(instances.cols):
        attr_mode = instances.most_common_value(c)
        newCol = [attr_mode if(x == missing_val) else x for x in instances.col(c)]
        data = np.array(instances.data)
        data[:,c] = newCol
        instances.data = data.tolist()

    def train(self, instances, labels):
        """
        :type instances: Matrix
        :type labels: Matrix
        """
        self.fillMissingValues(instances)
        instances.print()
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
        labels += [label]

        
