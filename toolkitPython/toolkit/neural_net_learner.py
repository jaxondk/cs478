from __future__ import (absolute_import, division, print_function, unicode_literals)

from .supervised_learner import SupervisedLearner
from .matrix import Matrix
import numpy as np
import matplotlib.pyplot as plt


class NeuralNetLearner(SupervisedLearner):
    labels = []
    weightMatrices = []
    biasWeights = []
    activationList = []
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
        self.biasWeights.append(np.full(self.nNodesPerHiddenLayer, initVal if initVal else np.random.normal()))

        ### init weight matrices for inner hidden layers
        # (nNodesPerHiddenLayer x nNodesPerHiddenLayer)
        for l in range(self.nHiddenLayers-1): 
            self.weightMatrices.append(np.full((self.nNodesPerHiddenLayer, self.nNodesPerHiddenLayer), initVal if initVal else np.random.normal()))
            self.biasWeights.append(np.full(self.nNodesPerHiddenLayer, initVal if initVal else np.random.normal()))

        ### init weight matrix for last hidden layer to output layer
        # (nNodesPerHiddenlayer x nOutputNodes)
        self.weightMatrices.append(np.full((self.nNodesPerHiddenLayer, self.nOutputNodes), initVal if initVal else np.random.normal()))
        self.biasWeights.append(np.full(self.nOutputNodes, initVal if initVal else np.random.normal()))

    def forwardProp(self, instance):
        nodeInput = instance
        for l in range(self.nHiddenLayers+1):
            print('layer: ', l)
            activation = self.activationFromInput(nodeInput, l)
            self.activationList.append(activation)
            nodeInput = activation

        out = self.activationList[self.nHiddenLayers]
        print('Out:', out)
        input('Pause')
        return out

    # sigmoid activation
    def activationFromInput(self, nodeInput, layer):
        print('nodeInput:', nodeInput)
        net = np.dot(nodeInput, self.weightMatrices[layer]) + self.biasWeights[layer] 
        print('Net: ', net)
        activation = 1/(1+np.exp(-net))
        print('Activation: ', activation)
        return activation

    def computeError(self):
        pass

    def backProp(self):
        pass

    def train(self, features, labels):
        """
        :type features: Matrix
        :type labels: Matrix
        """
        print(features.row(0))
        nFeatures = features.cols
        nInstances = features.rows
        print('nfeatures: ', nFeatures)
        print('ninstances: ', nInstances)

        self.initHyperParamsHW()
        self.initWeightMatrices(nFeatures, 1)
        print('weights:',self.weightMatrices)
        print('bias weights:',self.biasWeights)
        for e in range(self.EPOCHS):
            # TODO - from spec: "training set randomization at each epoch"
            for i in range(features.rows):
                instance = np.atleast_2d(features.row(i))
                out = self.forwardProp(instance)
                print('Out: ', out)
                self.computeError()
                self.backProp()

        

    def predict(self, features, labels):
        """
        :type features: [float]
        :type labels: [float]
        """
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
