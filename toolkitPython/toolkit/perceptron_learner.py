from __future__ import (absolute_import, division, print_function, unicode_literals)

from .supervised_learner import SupervisedLearner
from .matrix import Matrix
import numpy as np


class PerceptronLearner(SupervisedLearner):
    MAX_EPOCHS = 3 
    STALL_NUM_EPOCHS = 5
    LEARNING_RATE = .1
    labels = []
    weights = []

    def __init__(self):
        pass

    def initWeights(self, numFeatures):
        self.weights = np.zeros(numFeatures+1) #+1 for bias weight
        # print('Initial weights: ', self.weights)

    # Change in weights = c(t-z)x_i
    def updateWeights(self, pattern, label, out):
        change = self.LEARNING_RATE * (label - out) * pattern
        # print('Change in weights: ', change)
        self.weights += change
        # print('Updated weights: ', self.weights)

    # Sum(w_i*x_i)
    # This will use the latest weight vector in self.weights
    def calcNet(self, pattern):
        net = (self.weights * pattern).sum()
        # print("Net: ", net)
        return net

    def trainModel(self, instance, label):
        pattern, out = self.predictOne(instance)
        self.updateWeights(pattern, label, out)

    def train(self, features, labels):
        """
        :type features: Matrix
        :type labels: Matrix
        """
        self.labels = []
        self.initWeights(features.cols)

        # Train over several epochs. Remember, an epoch is an iteration over the whole training set
        for e in range(self.MAX_EPOCHS):
            print("Epoch " + str(e))
            # Should not need to iterate through instances due to broadcasting?... Not sure how though
            for i in range(features.rows):
                self.trainModel(features.row(i), labels.row(i)[0])

            # TODO: check accuracy. 
            accuracy = self.measure_accuracy(features, labels)
            print('accuracy: ', accuracy)
            # TODO: If accuracy stalls for STALL_NUM_EPOCHS, break

    def predictOne(self, instance):
        """
        @return: a tuple: (the instance with bias appended, the prediction for that instance)
        """
        pattern = np.append(np.array(instance), 1) #include a bias
        # print('Pattern w/ bias: ', pattern)
        net = self.calcNet(pattern)
        pred = 1.0 if (net > 0) else 0.0
        # self.labels.append(pred)
        return (pattern, pred)

    def predict(self, featureRow, pred):
        """
        :type featureRow: [float]
        :type pred: [float] - The manager is expecting an array, but it will be an array of length 1 containing the one prediction
        """
        del pred[:]
        _, out = self.predictOne(featureRow)
        pred.append(out)
