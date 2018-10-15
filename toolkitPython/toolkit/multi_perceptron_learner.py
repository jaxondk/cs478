from __future__ import (absolute_import, division, print_function, unicode_literals)

from .supervised_learner import SupervisedLearner
from .perceptron_learner import PerceptronLearner
from .matrix import Matrix
import numpy as np
import matplotlib.pyplot as plt

class MultiPerceptronLearner(SupervisedLearner):
    transformedLabels = []
    perceptrons = []

    def __init__(self):
        pass

    def train(self, features, labels):
        """
        :type features: Matrix
        :type labels: Matrix
        """
        classes = np.unique(labels.col(0))

        # for i in range(1):
        for i in range(len(classes)):
            perceptron = PerceptronLearner()
            transformedLabelMatrix = Matrix(labels, 0, 0, labels.rows, labels.cols) # Clone label matrix
            transformedLabelMatrix.setCol(0, list(map(lambda label: 0 if(label != i) else 1, labels.col(0)))) # make multi-class labels binary
            perceptron.train(features, transformedLabelMatrix)
            self.perceptrons.append(perceptron)
            self.transformedLabels.append(transformedLabelMatrix)


    def predict(self, featureRow, pred):
        """
        :type featureRow: [float]
        :type pred: [float] - The manager is expecting an array, but it will be an array of length 1 containing the one prediction
        """
        del pred[:]
        possiblePreds = []
        nets = []
        for i in range(len(self.perceptrons)):
            _, out, net = self.perceptrons[i].predictOne(featureRow)
            # print('Perceptron {0}\'s prediction: {1}'.format(i, out))
            possiblePreds.append(out)
            nets.append(net)

        # The max prediction will come from the perceptron that fired. This perceptron's index represents the label it trained on
        maxes = [i for i, prediction in enumerate(possiblePreds) if prediction == max(possiblePreds)]
        if(len(maxes) == 1):
            finalLabel = maxes[0]
        else:
            # print('nets', nets)
            finalLabel = nets.index(max(nets, key=abs))
        # print('Final label: ', finalLabel)
        pred.append(finalLabel)
