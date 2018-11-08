from __future__ import (absolute_import, division,
                        print_function, unicode_literals)

from .supervised_learner import SupervisedLearner
from .matrix import Matrix
import numpy as np
import matplotlib.pyplot as plt

class Node():
  """
  name: string
  attrForSplit: int (index)
  avFromSplit: type(attribute value for split)
  instances: Matrix. Features of this node's instance set
  labels: Matrix. Labels of this node's instance set
  parent: Node
  children: Node dict. key=attribute value for the split, value=Node
  availableAttributes: int[]. Array of column indices. These features have not been split on yet
  out: float. What this node will predict. Only set on leaf nodes
  """
  total_nodes_in_tree = 0
  total_levels_in_tree = 1

  def print(self, withPatterns=True, spacing=''):
    print(spacing+'----- Node: {0} -----'.format(self.name))
    print(spacing+'Available attributes: {0}'.format(
        [self.instances.attr_name(a) for a in self.availableAttributes]))
    if (hasattr(self, 'attrForSplit')):
      print(spacing+'Attribute this node splits on: {0}'.format(self.instances.attr_name(self.attrForSplit)))
    else:
      print(spacing+'Leaf Node. Out={0}'.format(self.labels.attr_value(0, self.out)))
    if (withPatterns): 
      print(spacing+'Applicable patterns: {0}'.format(len(self.instances.data)))
      self.instances.printData(self.labels, spacing)

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

  def noMoreAttributes(self):
    return len(self.availableAttributes) == 0

  def calcInfoS_a(self, a, vc, attr_counts, label_counts): 
    denominator = attr_counts[a, :vc]
    numerators = []
    for av in range(vc):
      numerators.append(label_counts[:, a][:vc, av])
    numerators = np.array(numerators)
    p = numerators / denominator[:, None] #this allows you to correctly broadcast denominator despite its shape being (valCount,)
    unsummed = np.where(p > 0, p * np.log2(p), 0) #Note: 'where' still runs the np.log part even if p>0, just doesn't return it. So still get warning
    return -(unsummed).sum(axis=1)

  # TODO - rewrite calcEntropyAttributes to use self.availableAttributes for attrs
  def calcEntropyAttributes(self):
    entropy = 0
    attrs = range(self.instances.cols) 
    entropy_attrs = np.zeros(len(attrs))
    maxAttr = self.instances.maxValueCount(attrs)
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
    entropy_available_attrs = {i: e for i, e in enumerate(entropy_attrs) if i in self.availableAttributes}
    return entropy_available_attrs

  # Recursive algorithm.
  def id3(self):
    Node.total_nodes_in_tree += 1 
    # self.print()
    if(self.instances.rows == 0):
      self.out = self.parent.labels.most_common_value(0)
      # print('No. instances for {0}. Out={1}'.format(self.name, self.out))
      return
    elif (self.isPureLeafNode()):
      self.out = self.labels.get(0, 0)
      # print('{0} is a leaf node. Out={1}'.format(self.name, self.out))
      return
    elif (self.noMoreAttributes()):
      self.out = self.labels.most_common_value(0)
      # print('{0} has no more attributes to split on. Out={1}'.format(self.name, self.out))
      return
    entropy_available_attrs = self.calcEntropyAttributes()
    attrForSplit = min(entropy_available_attrs, key=entropy_available_attrs.get)
    self.split(attrForSplit)
    Node.total_levels_in_tree += 1
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

  def printTree(self, spacing):
    self.print(withPatterns=False, spacing=spacing)
    spacing += '\t'
    for c in self.children.values():
      c.printTree(spacing)


class DecisionTreeLearner(SupervisedLearner):
    MISSING_VAL = float('Inf')

    def __init__(self):
        pass

    def fillMissingValues(self, instances):
      for c in range(instances.cols):
        self.attr_modes.append(instances.most_common_value(c))
        newCol = [self.attr_modes[c] if(x == DecisionTreeLearner.MISSING_VAL) else x for x in instances.col(c)]
        data = np.array(instances.data)
        data[:,c] = newCol
        instances.data = data.tolist()
    
    def fillMissingValueForPredict(self, instance):
        return [self.attr_modes[i] if (x == DecisionTreeLearner.MISSING_VAL) else x for i, x in enumerate(instance)]

    def train(self, instances, labels, val_instances=None, val_labels=None, test_instances=None, test_labels=None):
        """
        :type instances: Matrix
        :type labels: Matrix
        """
        Node.total_nodes_in_tree = 0
        self.attr_modes = []
        self.fillMissingValues(instances)
        availableAttributes = range(len(instances.row(0)))
        self.root = Node('root', None, instances, labels, None, availableAttributes)
        self.root.id3()
        # self.root.printTree('')
        if(val_instances != None):
          self.bssf, _ = self.measure_accuracy(val_instances, val_labels)
          test_acc, _ = self.measure_accuracy(test_instances, test_labels)
          # print('Validation accuracy of unpruned tree:', self.bssf)
          print('Test accuracy of unpruned tree:', test_acc)
          print('Number of nodes in unpruned tree', Node.total_nodes_in_tree)
          print('Number of levels in unpruned tree', Node.total_levels_in_tree)
          Node.total_nodes_in_tree = 0
          Node.total_levels_in_tree = 1
          self.prune(self.root, val_instances, val_labels)
          # print('Validation accuracy after pruning finished', self.bssf)
          print('Number of nodes in pruned tree', Node.total_nodes_in_tree)
          print('Number of levels in pruned tree', Node.total_levels_in_tree)

    # Recursive f(x)
    def prune(self, node, val_instances, val_labels):
      Node.total_nodes_in_tree += 1
      # if during training this node was assigned an out, then it's a leaf node.
      if(hasattr(node, 'out')):
        return
      # Otherwise, temporarily prune subtree and check if no loss in accuracy
      children = node.children
      node.children = [] #prune children
      node.out = node.labels.most_common_value(0)
      val_acc, _ = self.measure_accuracy(val_instances, val_labels)
      # If no loss, pruning should be permanent
      if (val_acc >= self.bssf):
        self.bssf = val_acc 
        return
      # Otherwise, undo the prune and recurse on children
      else:
        node.children = children
        del node.out
        Node.total_levels_in_tree += 1
        for c in node.children.values():
         self.prune(c, val_instances, val_labels)

    def predict(self, instance, labels):
        """
        :type instance: [float]
        :type labels: [float]
        """
        del labels[:]
        instance = self.fillMissingValueForPredict(instance)
        label = self.root.predict(instance)
        labels += [label]

        
