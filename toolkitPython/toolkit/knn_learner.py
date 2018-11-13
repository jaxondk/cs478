from __future__ import (absolute_import, division, print_function, unicode_literals)

from .supervised_learner import SupervisedLearner
from .matrix import Matrix
import numpy as np
import matplotlib.pyplot as plt


def mode(a, axis=0):
        # taken from scipy code
        # https://github.com/scipy/scipy/blob/master/scipy/stats/stats.py#L609
        scores = np.unique(np.ravel(a))       # get ALL unique values
        testshape = list(a.shape)
        testshape[axis] = 1
        oldmostfreq = np.zeros(testshape)
        oldcounts = np.zeros(testshape)

        for score in scores:
            template = (a == score)
            counts = np.expand_dims(np.sum(template, axis), axis)
            mostfrequent = np.where(counts > oldcounts, score, oldmostfreq)
            oldcounts = np.maximum(counts, oldcounts)
            oldmostfreq = mostfrequent

        return mostfrequent, oldcounts

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
        self.npLabels = np.array(labels.data)

    def euclidean(self, p1, p2):
        summation = np.sum((p1 - p2)**2, axis=1)
        return np.sqrt(summation)

    def manhattan(self, p1, p2):
        summation = np.sum(np.abs(p1-p2), axis=1)
        return summation

    def predict(self, featureRow, out):
        """
        :type featureRow: [float]
        :type out: [float]. After predict f(x), len(out) = 1
        """
        ### Initialize k and other hyperparams. Can also wrap here to test multiple k's
        k = 3
        weighting = False
        regression = False

        # TODO - remove this, just for testing
        featureRow = [.5, .2]

        ### Measure distance to all stored instances. Keep k nearest
        distances = self.manhattan(np.array(featureRow), self.npFeatures)
        # argpartition sorts only k elements so its faster than a sort. Returns indices to the k minimum elements in ascending order
        min_indices = np.argpartition(distances, range(k))[:k] 

        pred = None
        ### k nearest instances vote on output class
        if(weighting):
            if (regression):
                pass
            else:
                pass
        else:
            if (regression):
                pass
            else:
                print(self.npLabels)
                labelsOfKNeighbors = self.npLabels[min_indices]
                print(labelsOfKNeighbors)
                pred, _ = mode(labelsOfKNeighbors)
                print(pred)
                
        out.append(pred[0][0]) # the prediction is a double nested array because of toolkit stupidity

