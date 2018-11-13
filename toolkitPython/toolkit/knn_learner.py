from __future__ import (absolute_import, division, print_function, unicode_literals)

from .supervised_learner import SupervisedLearner
from .matrix import Matrix
import numpy as np
import matplotlib.pyplot as plt

class KNNLearner(SupervisedLearner):

    def __init__(self):
        pass

    def train(self, features, labels):
        """
        :type features: Matrix
        :type labels: Matrix
        """
        # TODO - do some reduction here to make training set smaller
        self.npFeatures = np.array(features.data)
        self.npLabels = np.array(labels)

    def euclidean(self, p1, p2):
        summation = np.sum((p1 - p2)**2, axis=1)
        return np.sqrt(summation)

    def manhattan(self, p1, p2):
        summation = np.sum(np.abs(p1-p2), axis=1)
        return summation

    def predict(self, featureRow, pred):
        """
        :type featureRow: [float]
        :type pred: [float]. After predict f(x), len(pred) = 1
        """
        del pred[:]
        ### Initialize k and other hyperparams. Can also wrap here to test multiple k's
        k = 3
        weighting = False
        regression = False

        ### Measure distance to all stored instances. 
        ### Sort from lowest distance to highest. 
        ### Keep track of original index into instances for voting
        distances = self.manhattan(np.array(featureRow), self.npFeatures)
        print(distances)
        if(weighting):
            pass
        else:
            pass

        ### k nearest instances vote on output class
        if (regression):
            pass   
        else:
            pass

        # Save prediction

