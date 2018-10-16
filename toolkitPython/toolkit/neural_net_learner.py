from __future__ import (absolute_import, division, print_function, unicode_literals)

from .supervised_learner import SupervisedLearner
from .matrix import Matrix
import numpy as np
import matplotlib.pyplot as plt


class NeuralNetLearner(SupervisedLearner):
    labels = []
    weightMatrices = []
    deltaWeightMatrices = []
    biasWeights = []
    biasDeltaWeights = []
    activationList = []
    errorList = [] # list of lists. Each list represents error values for the nodes in a layer
    nNodesPerHiddenLayer = None
    nHiddenLayers = None
    nOutputNodes = None
    EPOCHS = 1
    LEARNING_RATE = None
    MOMENTUM = None

    def __init__(self):
        pass

    # def initTestWeights(self):

    def initHyperParamsHW(self):
        self.nHiddenLayers = 1
        self.nNodesPerHiddenLayer = 2
        self.nOutputNodes = 1
        self.LEARNING_RATE = 1
        self.MOMENTUM = 0

    # Part 1 - iris dataset
    def initHyperParams1(self, nFeatures):
        self.nNodesPerHiddenLayer = nFeatures * 2
        self.nHiddenLayers = 1
        self.LEARNING_RATE = .1
        self.MOMENTUM = 0
    
    def initWeightMatrices(self, nFeatures, initVal):
        ### init weight matrix for input layer to first hidden layer
        # (nFeatures x nNodesPerHiddenLayer)
        self.weightMatrices.append(np.full((nFeatures, self.nNodesPerHiddenLayer), initVal if initVal else np.random.normal()))
        self.deltaWeightMatrices.append(np.zeros((nFeatures, self.nNodesPerHiddenLayer)))
        self.biasWeights.append(np.full(self.nNodesPerHiddenLayer, initVal if initVal else np.random.normal()))

        ### init weight matrices for inner hidden layers
        # (nNodesPerHiddenLayer x nNodesPerHiddenLayer)
        for l in range(self.nHiddenLayers-1): 
            self.weightMatrices.append(np.full((self.nNodesPerHiddenLayer, self.nNodesPerHiddenLayer), initVal if initVal else np.random.normal()))
            self.deltaWeightMatrices.append(np.zeros((self.nNodesPerHiddenLayer, self.nNodesPerHiddenLayer)))
            self.biasWeights.append(np.full(self.nNodesPerHiddenLayer, initVal if initVal else np.random.normal()))

        ### init weight matrix for last hidden layer to output layer
        # (nNodesPerHiddenlayer x nOutputNodes)
        self.weightMatrices.append(np.full((self.nNodesPerHiddenLayer, self.nOutputNodes), initVal if initVal else np.random.normal()))
        self.deltaWeightMatrices.append(np.zeros((self.nNodesPerHiddenLayer, self.nOutputNodes)))
        self.biasWeights.append(np.full(self.nOutputNodes, initVal if initVal else np.random.normal()))

        ### Init other structures (shape only)
        for l in range(self.nHiddenLayers+1):
            self.errorList.append([]) 
            self.biasDeltaWeights.append([])

    def forwardProp(self, instance):
        self.activationList.append(instance) # the input nodes do not have activation f(x), just consider incoming instance as their output
        nodeInput = instance
        for l in range(self.nHiddenLayers+1):
            print('layer: ', l)
            activation = self.activationFromInput(nodeInput, l)
            self.activationList.append(activation)
            nodeInput = activation

    # sigmoid activation
    def activationFromInput(self, nodeInput, layer):
        print('nodeInput:', nodeInput)
        net = np.dot(nodeInput, self.weightMatrices[layer]) + self.biasWeights[layer] 
        print('Net: ', net)
        activation = 1/(1+np.exp(-net))
        print('Activation: ', activation)
        return activation

    # accurate for hw
    def computeErrorOutputLayer(self, target):
        # TODO - convert target to 1 hot encoding. I think this is needed when you have more than one output node
        out = self.activationList[self.nHiddenLayers+1]
        self.errorList[self.nHiddenLayers] = (target - out) * out * (1 - out)
        print('Error list after doing output layer error:', self.errorList)

    # accurate for hw
    def computeErrorHiddenLayer(self, j):
        error = np.dot(self.errorList[j+1], self.weightMatrices[j+1].T) * (self.activationList[j+1] * (1 - self.activationList[j+1]))
        self.errorList[j] = error
    # accurate for hw
    def computeError(self, target):
        self.computeErrorOutputLayer(target)
        for l in range(self.nHiddenLayers-1, -1, -1):
            self.computeErrorHiddenLayer(l)

    # calc errors for all the layers first, then calc delta weights for all layers, then update all the weights
    def backProp(self, target):
        self.computeError(target)
        for l in range(self.nHiddenLayers, -1, -1):
            self.deltaWeightMatrices[l] = self.LEARNING_RATE * np.dot(self.activationList[l].T, self.errorList[l])
            # TODO - update bias weights as well
        print('Delta weights after BP:', self.deltaWeightMatrices)
        input('BP done')

    def train(self, features, labels):
        """
        :type features: Matrix
        :type labels: Matrix
        """
        nFeatures = features.cols
        nInstances = features.rows
        print('nfeatures: ', nFeatures)
        print('ninstances: ', nInstances)

        self.initHyperParamsHW()
        self.initWeightMatrices(nFeatures, 1)
        print('weights:',self.weightMatrices)
        print('bias weights:',self.biasWeights)
        for e in range(self.EPOCHS):
            # TODO - from spec: "training set randomization at each epoch". I think this just means shuffle
            for i in range(features.rows):
                self.activationList.clear() # Have a feeling this is needed between instances
                instance = np.atleast_2d(features.row(i))
                self.forwardProp(instance)
                self.backProp(labels.row(i))

        

    def predict(self, features, labels):
        """
        :type features: [float]
        :type labels: [float]
        """
        #pick the output node/class with highest activation

        del labels[:]
        labels += self.labels


'''
Notes for lab:

WEIGHTS
weight matrix will be f+1 x h | f = # features and h = # of hidden nodes in latent layer. +1 for bias
Have a weight matrix per layer (excluding input layer)

DELTA WEIGHTS
same shape as weights. may just be a local variable

'''
