from __future__ import (absolute_import, division, print_function, unicode_literals)

from .supervised_learner import SupervisedLearner
from .matrix import Matrix
import numpy as np
import matplotlib.pyplot as plt

class PerceptronLearner(SupervisedLearner):
    MAX_EPOCHS = 35 
    STALL_NUM_EPOCHS = 5
    LEARNING_RATE = .1
    labels = []
    weights = []
    debug = False

    def __init__(self):
        pass

    def initWeights(self, numFeatures):
        self.weights = np.zeros(numFeatures+1) #+1 for bias weight
        if(self.debug): print('Initial weights: ', self.weights)

    # Change in weights = c(t-z)x_i
    def updateWeights(self, pattern, label, out):
        if(self.debug): print('Target/label: ', label)
        change = self.LEARNING_RATE * (label - out) * pattern
        if(self.debug): print('Change in weights: ', change)
        self.weights += change
        if(self.debug): print('Updated weights: ', self.weights)

    # Sum(w_i*x_i)
    def calcNet(self, pattern, weights):
        net = (weights * pattern).sum()
        if(self.debug): print("Net: ", net)
        return net

    def trainModel(self, instance, label):
        pattern, out, _ = self.predictOne(instance)
        self.updateWeights(pattern, label, out)

    def train(self, features, labels):
        """
        :type features: Matrix
        :type labels: Matrix
        """
        self.initWeights(features.cols)

        # Train over several epochs. Remember, an epoch is an iteration over the whole training set
        prev_accuracy = 0.0
        noImprovementCount = 0
        # avgMisclassRate = []
        # totalMisclass = 0.0
        for e in range(self.MAX_EPOCHS):
            features.shuffle(buddy=labels)
            if(self.debug): print("Epoch " + str(e))
            for i in range(features.rows):
                self.trainModel(features.row(i), labels.row(i)[0])

            # Calc accuracy and check if it has stalled
            accuracy, _ = self.measure_accuracy(features, labels)
            # totalMisclass += (1 - accuracy)
            # avgMisclassRate.append(totalMisclass/(e+1))
            # print('misclass: ', (1-accuracy))
            if(self.debug): print('accuracy: ', accuracy)
            if(accuracy <= prev_accuracy):
                noImprovementCount += 1
            else:
                noImprovementCount = 0
                prev_accuracy = accuracy
            if(noImprovementCount == self.STALL_NUM_EPOCHS):
                print('Accuracy has stalled for {0} epochs, ending training on epoch {1}'.format(self.STALL_NUM_EPOCHS, e))
                break
            

        # print('final weight vector: ', self.weights)
        # self.plotSeparability(self.weights, features, labels)
        # self.plotMisclassification(avgMisclassRate)

    def plotMisclassification(self, avgMissclassRate):
        plt.plot(range(self.MAX_EPOCHS), avgMissclassRate)
        plt.xlabel('Epoch')
        plt.ylabel('Avg. Misclassification Rate')
        plt.title('Avg. Misclassification Rate vs. Epoch')
        plt.show()

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
        # plt.title('Linearly Separable Data')
        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')
        plt.show()
    
    def predictOne(self, instance):
        """
        @return: a tuple: (the instance with bias appended, the prediction for that instance, the net)
        """
        pattern = np.append(np.array(instance), 1) #include a bias
        if(self.debug): print('Pattern w/ bias: ', pattern)
        net = self.calcNet(pattern, self.weights)
        pred = 1.0 if (net > 0) else 0.0
        return (pattern, pred, net)

    def predict(self, featureRow, pred):
        """
        :type featureRow: [float]
        :type pred: [float] - The manager is expecting an array, but it will be an array of length 1 containing the one prediction
        """
        del pred[:]
        _, out, _ = self.predictOne(featureRow)
        pred.append(out)
