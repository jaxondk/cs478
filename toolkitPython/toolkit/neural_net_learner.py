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
    nNodesPerHiddenLayer = 2
    nHiddenLayers = 1
    nOutputNodes = 1
    # nTotalLayers = nHiddenLayers + 2
    EPOCHS = 5

    def __init__(self):
        pass
    
    def initWeightMatrices(self, nFeatures):
        ### init weight matrix for input layer to first hidden layer
        # (nFeatures x nNodesPerHiddenLayer)
        self.weightMatrices.append(np.full((nFeatures, self.nNodesPerHiddenLayer), np.random.normal()))
        self.biasWeights.append(np.full(self.nNodesPerHiddenLayer, np.random.normal()))

        ### init weight matrices for inner hidden layers
        # (nNodesPerHiddenLayer x nNodesPerHiddenLayer)
        for l in range(self.nHiddenLayers-1): 
            self.weightMatrices.append(np.full((self.nNodesPerHiddenLayer, self.nNodesPerHiddenLayer), np.random.normal()))
            self.biasWeights.append(np.full(self.nNodesPerHiddenLayer, np.random.normal()))

        ### init weight matrix for last hidden layer to output layer
        # (nNodesPerHiddenlayer x nOutputNodes)
        self.weightMatrices.append(np.full((self.nNodesPerHiddenLayer, self.nOutputNodes), np.random.normal()))
        self.biasWeights.append(np.full(self.nOutputNodes, np.random.normal()))

    def forwardProp(self, instance):
        input = instance
        for l in range(self.nHiddenLayers):
            net = np.dot(input, self.weightMatrices[l]) + self.biasWeights[l] 
            self.activationList.append(self.activation(net))
            input = self.activationList[l]
        out = self.activationList[self.nHiddenLayers-1]
        return out

    # sigmoid activation
    def activation(self, net):
        return 1/(1+np.exp(-net))

    def computeError(self):
        pass

    def backProp(self):
        pass

    def train(self, features, labels):
        """
        :type features: Matrix
        :type labels: Matrix
        """
        nFeatures = features.cols
        nInstances = features.rows
        print('nfeatures: ', nFeatures)
        print('ninstances: ', nInstances)
        self.initWeightMatrices(nFeatures)
        print('weights:',self.weightMatrices)
        print('bias weights:',self.biasWeights)

        for e in range(self.EPOCHS):
            for i in range(features.rows):
                input = np.atleast_2d(features.row(i))
                out = self.forwardProp(input)
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
