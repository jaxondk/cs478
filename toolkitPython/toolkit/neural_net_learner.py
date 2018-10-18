from __future__ import (absolute_import, division, print_function, unicode_literals)

from .supervised_learner import SupervisedLearner
from .matrix import Matrix
import numpy as np
import matplotlib.pyplot as plt


'''
For example 2 dataset (continuous):
DONE 1. labels.value_count tells you how many output nodes to have. But for continuous, this returns 0. Need 1 output node
DONE 2. For continuous output, target needs no modification. For nominal, must do 1-hot encoding
DONE 3. In predict, if it's continuous you just return whatever the output. If nominal, you return index of the highest output node (just like multiperceptron).
'''

class NeuralNetLearner(SupervisedLearner):
    weightMatrices = [] #represents weight networks between layers. len(weightMatrices) = Total layers - 1
    deltaWeightMatrices = []
    biasWeights = []
    deltaBiasWeights = []
    activationList = [] #List of lists. Each list represents the output/activation of the layer. Input layer included
    errorList = [] # list of lists. Each list represents error values for the nodes in a layer. Input layer excluded (len(errorList) = len(activationList) - 1)
    nNodesPerHiddenLayer = None
    nHiddenLayers = None
    nOutputNodes = None
    isContinuous = None
    EPOCHS = None
    STALL_NUM_EPOCHS = None # set in initHyperParamsVowel
    LEARNING_RATE = None # set in initHyperParamsVowel
    MOMENTUM = None 
    # For vowel analysis only
    finalTrainMSE = []
    finalValMSE = []
    finalTestMSE = []
    epochsRequired = []

    def __init__(self):
        pass

    # def initHyperParamsIris(self, nFeatures):
    #     self.nNodesPerHiddenLayer = nFeatures * 2
    #     self.nHiddenLayers = 1
    #     self.LEARNING_RATE = .1
    #     self.MOMENTUM = 0
    #     np.random.seed(0) 

    def initHyperParamsVowel(self, nFeatures):
        self.nNodesPerHiddenLayer = 64
        self.nHiddenLayers = 1
        self.LEARNING_RATE = .15
        self.MOMENTUM = 0
        self.STALL_NUM_EPOCHS = 75
        self.EPOCHS = 400
        np.random.seed(0)
    
    def initWeightMatrices(self, nFeatures, initVal=None):
        self.weightMatrices.clear()
        self.deltaWeightMatrices.clear()
        self.biasWeights.clear()
        self.deltaBiasWeights.clear()
        self.errorList.clear()

        ### Init shape of structures
        for _ in range(self.nHiddenLayers+1):
            self.errorList.append([]) 

        ### init weight matrix for input layer to first hidden layer
        # (nFeatures x nNodesPerHiddenLayer)
        self.weightMatrices.append(np.full((nFeatures, self.nNodesPerHiddenLayer), \
            initVal if initVal else np.random.normal(size=(nFeatures, self.nNodesPerHiddenLayer))))
        self.deltaWeightMatrices.append(np.zeros((nFeatures, self.nNodesPerHiddenLayer)))
        self.biasWeights.append(np.full(self.nNodesPerHiddenLayer, initVal if initVal else np.random.normal(size=self.nNodesPerHiddenLayer)))
        self.deltaBiasWeights.append(np.zeros(self.nNodesPerHiddenLayer))

        ### init weight matrices for inner hidden layers
        # (nNodesPerHiddenLayer x nNodesPerHiddenLayer)
        for _ in range(self.nHiddenLayers-1): 
            self.weightMatrices.append(np.full((self.nNodesPerHiddenLayer, self.nNodesPerHiddenLayer), \
                initVal if initVal else np.random.normal(size=(self.nNodesPerHiddenLayer, self.nNodesPerHiddenLayer))))
            self.deltaWeightMatrices.append(np.zeros((self.nNodesPerHiddenLayer, self.nNodesPerHiddenLayer)))
            self.biasWeights.append(np.full(self.nNodesPerHiddenLayer, initVal if initVal else np.random.normal(size=self.nNodesPerHiddenLayer)))
            self.deltaBiasWeights.append(np.zeros(self.nNodesPerHiddenLayer))

        ### init weight matrix for last hidden layer to output layer
        # (nNodesPerHiddenlayer x nOutputNodes)
        self.weightMatrices.append(np.full((self.nNodesPerHiddenLayer, self.nOutputNodes), \
            initVal if initVal else np.random.normal(size=(self.nNodesPerHiddenLayer, self.nOutputNodes))))
        self.deltaWeightMatrices.append(np.zeros((self.nNodesPerHiddenLayer, self.nOutputNodes)))
        self.biasWeights.append(np.full(self.nOutputNodes, initVal if initVal else np.random.normal(size=self.nOutputNodes)))
        self.deltaBiasWeights.append(np.zeros(self.nOutputNodes))

    def forwardProp(self, instance):
        self.activationList.append(instance) # the input nodes do not have activation f(x), just consider incoming instance as their output
        # TODO - don't use append here. build your activationList shape beforehand and index in. Then you don't have to keep clearing activationList
        nodeInput = instance
        for l in range(self.nHiddenLayers+1):
            activation = self.activationFromInput(nodeInput, l)
            self.activationList.append(activation)
            nodeInput = activation

    # sigmoid activation
    def activationFromInput(self, nodeInput, layer):
        net = np.dot(nodeInput, self.weightMatrices[layer]) + self.biasWeights[layer] 
        activation = 1/(1+np.exp(-net))
        return activation

    def computeErrorOutputLayer(self, target):
        # TODO - convert target to 1 hot encoding. I think this is needed when you have more than one output node
        out = self.activationList[self.nHiddenLayers+1]
        self.errorList[self.nHiddenLayers] = (target - out) * out * (1 - out)

    def computeErrorHiddenLayer(self, j):
        error = np.dot(self.errorList[j+1], self.weightMatrices[j+1].T) * (self.activationList[j+1] * (1 - self.activationList[j+1]))
        self.errorList[j] = error

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

    def oneHot(self, label):
        target = np.zeros(self.nOutputNodes)
        target[int(label)] = 1
        return target


    def backProp(self, target):
        self.computeError(target)
        self.updateWeights()

    def trainModel(self, row, label):
        instance = np.atleast_2d(row)
        self.forwardProp(instance)
        target = label if self.isContinuous else self.oneHot(label[0])
        self.backProp(target)
        self.activationList.clear() # this is needed between instances b/c we append to it throughout the algorithm
        
    def realTrain(self, features, labels, validationFeatures, validationLabels, testFeatures, testLabels):
        """
        :type features: Matrix
        :type labels: Matrix
        """

        nFeatures = features.cols
        self.isContinuous = labels.value_count(0) == 0
        self.nOutputNodes = labels.value_count(0) if not self.isContinuous else 1
        self.initHyperParamsVowel(nFeatures)
        self.initWeightMatrices(nFeatures)

        bssf_mse = 99999
        noImprovementCount = 0
        for e in range(self.EPOCHS):
            # if(e>0): input('pause')
            # print('EPOCH', e+1)
            features.shuffle(labels)
            for i in range(features.rows):
                self.trainModel(features.row(i), labels.row(i))

            _, vMSE = self.measure_accuracy(validationFeatures, validationLabels)
            if(vMSE >= bssf_mse):
                noImprovementCount += 1
            else:
                noImprovementCount = 0
                bssf_mse = vMSE
            if(noImprovementCount == self.STALL_NUM_EPOCHS):
                # print('MSE has stalled for {0} epochs, ending training on epoch {1}'.format(self.STALL_NUM_EPOCHS, e))
                break
        self.epochsRequired.append(e+1)

    #wrapper around train so that we can do some analysis
    def train(self, features, labels, validationFeatures, validationLabels, testFeatures, testLabels):
        learningRates = [.1, .15, .25, .5, .75, 1]
        for lr in learningRates:
            print('LR:', lr)
            self.LEARNING_RATE = lr
            self.realTrain(features, labels, validationFeatures, validationLabels, testFeatures, testLabels)
        # self.plotVowelMSE(self.finalTrainMSE, self.finalValMSE, self.finalTestMSE, learningRates) # For vowel MSE vs. LR analysis only
        self.plotVowelEpochs(self.epochsRequired, learningRates)

    def plotIrisMSE(self, trainMSE, valMSE, valAccuracy):
        plt.plot(range(len(trainMSE)), trainMSE, label='Train MSE') # labels make a legend when you call plt.legend(...)
        plt.plot(range(len(valMSE)), valMSE, label='Val MSE') 
        plt.plot(range(len(valAccuracy)), valAccuracy, label='Val Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('MSE/Accuracy')
        plt.title('IRIS: MSE/Accuracy vs. Epoch')
        plt.legend(loc='center right')
        plt.show()


    def plotVowelMSE(self, trainMSE, valMSE, testMSE, LRs):
        plt.plot(LRs, trainMSE, label='Train MSE')
        plt.plot(LRs, valMSE, label='Val MSE') 
        plt.plot(LRs, testMSE, label='Test MSE') 
        plt.xlabel('LR')
        plt.ylabel('MSE')
        plt.title('VOWEL: Final MSE vs. LR')
        plt.legend(loc='lower right')
        plt.xticks(LRs)
        plt.show()

    def plotVowelEpochs(self, epochsRequired, LRs):
        plt.plot(LRs, epochsRequired)
        plt.xlabel('LR')
        plt.ylabel('Epochs')
        plt.title('VOWEL: Epochs Required vs. LR, Stop={0} Epochs'.format(self.STALL_NUM_EPOCHS))
        plt.xticks(LRs)
        plt.show()

    # If continuous, you just return whatever the output node was.
    # If nominal, you return index of the highest output node (just like multiperceptron).
    def predict(self, featureRow, pred):
        """
        :type features: [float]
        :type preds: [float]
        """
        # TODO
        del pred[:]

        self.forwardProp(np.atleast_2d(featureRow))
        outputNodePreds = self.activationList[self.nHiddenLayers+1][0]
        self.activationList.clear()

        finalLabel = outputNodePreds[0] if self.isContinuous else np.argmax(outputNodePreds)
        pred += [finalLabel]
