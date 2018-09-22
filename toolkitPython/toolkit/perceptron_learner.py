from __future__ import (absolute_import, division, print_function, unicode_literals)

from .supervised_learner import SupervisedLearner
from .matrix import Matrix
import numpy as np
import matplotlib.pyplot as plt

class PerceptronLearner(SupervisedLearner):
    MAX_EPOCHS = 15 
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
    def calcNet(self, pattern, weights):
        net = (weights * pattern).sum()
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
        prev_accuracy = 0.0
        noImprovementCount = 0
        for e in range(self.MAX_EPOCHS):
            print("Epoch " + str(e))
            for i in range(features.rows):
                self.trainModel(features.row(i), labels.row(i)[0])

            # Calc accuracy and check if it has stalled
            accuracy = self.measure_accuracy(features, labels)
            print('accuracy: ', accuracy)
            if(accuracy == prev_accuracy):
                noImprovementCount += 1
            else:
                noImprovementCount = 0
            if(noImprovementCount == self.STALL_NUM_EPOCHS):
                print('Accuracy has stalled for {0} epochs, ending training'.format(self.STALL_NUM_EPOCHS))
                break
            prev_accuracy = accuracy

        self.plotSeparability(self.weights, features, labels)


    def plotSeparability(self, weights, features, labels):
        # plot points
        class1_x, class2_x, class1_y, class2_y = [], [], [], []
        for r in range(features.rows):
            if(labels.row(r)[0] == 0):
                class1_x.append(features.get(r, 0))
                class1_y.append(features.get(r, 1))
            else:
                class2_x.append(features.get(r, 0))
                class2_y.append(features.get(r, 1))
        plt.plot(class1_x, class1_y, 'bs')
        plt.plot(class2_x, class2_y, 'g^')
        # plot decision line
        m = -weights[0]/weights[1]
        b = weights[2]/weights[1]
        xs = np.array([features.column_min(0), features.column_max(0)])
        ys = m*xs + b
        plt.plot(xs, ys)
        plt.show()
    
    def predictOne(self, instance):
        """
        @return: a tuple: (the instance with bias appended, the prediction for that instance)
        """
        pattern = np.append(np.array(instance), 1) #include a bias
        # print('Pattern w/ bias: ', pattern)
        net = self.calcNet(pattern, self.weights)
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
