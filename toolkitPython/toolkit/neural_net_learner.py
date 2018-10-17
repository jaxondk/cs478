from __future__ import (absolute_import, division, print_function, unicode_literals)

from .supervised_learner import SupervisedLearner
from .matrix import Matrix
import numpy as np
import matplotlib.pyplot as plt


'''
For example 2 dataset (continuous):
1. labels.value_count tells you how many output nodes to have. But for continuous, this returns 0. Need 1 output node
2. For continuous output, target needs no modification. For nominal, must do 1-hot encoding
3. In predict, if it's continuous you just return whatever the output. If nominal, you return index of the highest output node (just like multiperceptron).
'''

class NeuralNetLearner(SupervisedLearner):
    labels = []
    weightMatrices = [] #represents weight networks between layers. len(weightMatrices) = Total layers - 1
    deltaWeightMatrices = []
    biasWeights = []
    deltaBiasWeights = []
    activationList = [] #List of lists. Each list represents the output/activation of the layer. Input layer included
    errorList = [] # list of lists. Each list represents error values for the nodes in a layer. Input layer excluded (len(errorList) = len(activationList) - 1)
    nNodesPerHiddenLayer = None
    nHiddenLayers = None
    nOutputNodes = None
    EPOCHS = 3
    LEARNING_RATE = None
    MOMENTUM = 1

    def __init__(self):
        pass

    def initHyperParamsHW(self):
        self.nHiddenLayers = 1
        self.nNodesPerHiddenLayer = 2
        self.LEARNING_RATE = 1
        self.MOMENTUM = 0

    def initHyperParamsIris(self, nFeatures):
        self.nNodesPerHiddenLayer = nFeatures * 2
        self.nHiddenLayers = 1
        self.LEARNING_RATE = .1
        self.MOMENTUM = 0

    def initHyperParamsEx2(self):
        self.nNodesPerHiddenLayer = 3
        self.nHiddenLayers = 1
        self.LEARNING_RATE = 0.175
        self.MOMENTUM = 0.9

    def changeWeightsForEx2(self):
        self.weightMatrices[0][0] = [-0.03, 0.04, 0.03]
        self.weightMatrices[0][1] = [0.03, -0.02, 0.02]
        self.weightMatrices[1][0] = [-0.01]
        self.weightMatrices[1][1] = [0.03]
        self.weightMatrices[1][2] = [0.02]
        self.biasWeights[0] = np.array([-0.01, 0.01, -0.02])
        self.biasWeights[1] = np.array([0.02])
        # input('pause')
        pass
    
    def initWeightMatrices(self, nFeatures, initVal=None):
        ### Init shape of structures
        for l in range(self.nHiddenLayers+1):
            self.errorList.append([]) 

        ### init weight matrix for input layer to first hidden layer
        # (nFeatures x nNodesPerHiddenLayer)
        self.weightMatrices.append(np.full((nFeatures, self.nNodesPerHiddenLayer), initVal if initVal else np.random.normal()))
        self.deltaWeightMatrices.append(np.zeros((nFeatures, self.nNodesPerHiddenLayer)))
        self.biasWeights.append(np.full(self.nNodesPerHiddenLayer, initVal if initVal else np.random.normal()))
        self.deltaBiasWeights.append(np.zeros(self.nNodesPerHiddenLayer))

        ### init weight matrices for inner hidden layers
        # (nNodesPerHiddenLayer x nNodesPerHiddenLayer)
        for l in range(self.nHiddenLayers-1): 
            self.weightMatrices.append(np.full((self.nNodesPerHiddenLayer, self.nNodesPerHiddenLayer), initVal if initVal else np.random.normal()))
            self.deltaWeightMatrices.append(np.zeros((self.nNodesPerHiddenLayer, self.nNodesPerHiddenLayer)))
            self.biasWeights.append(np.full(self.nNodesPerHiddenLayer, initVal if initVal else np.random.normal()))
            self.deltaBiasWeights.append(np.zeros(self.nNodesPerHiddenLayer))

        ### init weight matrix for last hidden layer to output layer
        # (nNodesPerHiddenlayer x nOutputNodes)
        self.weightMatrices.append(np.full((self.nNodesPerHiddenLayer, self.nOutputNodes), initVal if initVal else np.random.normal()))
        self.deltaWeightMatrices.append(np.zeros((self.nNodesPerHiddenLayer, self.nOutputNodes)))
        self.biasWeights.append(np.full(self.nOutputNodes, initVal if initVal else np.random.normal()))
        self.deltaBiasWeights.append(np.zeros(self.nOutputNodes))

    def forwardProp(self, instance):
        self.activationList.append(instance) # the input nodes do not have activation f(x), just consider incoming instance as their output
        nodeInput = instance
        for l in range(self.nHiddenLayers+1):
            activation = self.activationFromInput(nodeInput, l)
            self.activationList.append(activation)
            nodeInput = activation

    # sigmoid activation
    def activationFromInput(self, nodeInput, layer):
        net = np.dot(nodeInput, self.weightMatrices[layer]) + self.biasWeights[layer] 
        # print('Net: ', net)
        activation = 1/(1+np.exp(-net))
        # print('Activation: ', activation)
        return activation

    # accurate for hw
    def computeErrorOutputLayer(self, target):
        # TODO - convert target to 1 hot encoding. I think this is needed when you have more than one output node
        out = self.activationList[self.nHiddenLayers+1]
        self.errorList[self.nHiddenLayers] = (target - out) * out * (1 - out)
        # print('Error list after doing output layer error:', self.errorList)

    # accurate for hw
    def computeErrorHiddenLayer(self, j):
        error = np.dot(self.errorList[j+1], self.weightMatrices[j+1].T) * (self.activationList[j+1] * (1 - self.activationList[j+1]))
        self.errorList[j] = error
    # accurate for hw
    def computeError(self, target):
        self.computeErrorOutputLayer(target)
        for l in range(self.nHiddenLayers-1, -1, -1):
            self.computeErrorHiddenLayer(l)
    
    def updateWeights(self):
        for l in range(self.nHiddenLayers, -1, -1):
            deltaW = self.LEARNING_RATE * np.dot(self.activationList[l].T, self.errorList[l]) + self.deltaWeightMatrices[l] * self.MOMENTUM
            deltaB = self.LEARNING_RATE * np.dot([[1]], self.errorList[l]) + self.deltaBiasWeights[l] * self.MOMENTUM
            self.weightMatrices[l] = np.array(self.weightMatrices[l]) + deltaW
            self.biasWeights[l] = np.array(self.biasWeights[l]) + deltaB
            self.deltaWeightMatrices[l] = deltaW
            self.deltaBiasWeights[l] = deltaB

    # calc errors for all the layers first, then calc delta weights for all layers, then update all the weights
    def backProp(self, target):
        self.computeError(target)
        print('Error', self.errorList)
        self.updateWeights()
        print('weights after BP:', self.weightMatrices)
        print('bias weights after BP:',self.biasWeights)
        # input('BP done')

    def train(self, features, labels):
        """
        :type features: Matrix
        :type labels: Matrix
        """
        nFeatures = features.cols
        # print('nfeatures', nFeatures)
        self.nOutputNodes = labels.value_count(0) if labels.value_count(0) != 0 else 1
        self.initHyperParamsEx2()
        self.initWeightMatrices(nFeatures)
        self.changeWeightsForEx2()

        for e in range(self.EPOCHS):
            if(e>0): input('pause')
            print('EPOCH', e+1)
            # TODO - from spec: "training set randomization at each epoch". I think this just means shuffle
            for i in range(features.rows):
                instance = np.atleast_2d(features.row(i))
                print('Pattern: ',instance)
                self.forwardProp(instance)
                print('Activations', self.activationList)
                self.backProp(labels.row(i))
                self.activationList.clear() # Have a feeling this is needed between instances
                # self.errorList.clear()

        

    def predict(self, features, labels):
        """
        :type features: [float]
        :type labels: [float]
        """
        #pick the output node/class with highest activation
        # TODO
        del labels[:]
        labels += [1,1,1,1,1,1,1,1,1,1,1,1]
        


'''
Notes for lab:

WEIGHTS
weight matrix will be f+1 x h | f = # features and h = # of hidden nodes in latent layer. +1 for bias
Have a weight matrix per layer (excluding input layer)

DELTA WEIGHTS
same shape as weights. may just be a local variable

'''
