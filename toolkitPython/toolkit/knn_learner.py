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
        self.regression = labels.value_count(0) == 0
        # self.dataRow = 16

    def euclidean(self, p1, p2):
        summation = np.sum((p1 - p2)**2, axis=1)
        return np.sqrt(summation)

    def manhattan(self, p1, p2):
        summation = np.sum(np.abs(p1-p2), axis=1)
        return summation

    def calcWeights(self, distances, min_indices):
        return (1/distances[min_indices])**2

    def calcWeightedVotes(self, distances, min_indices, labelsOfKNN):
        weightsKNN = self.calcWeights(distances, min_indices)
        candidateClasses = np.unique(labelsOfKNN)
        weightedVotes = np.zeros(len(candidateClasses))
        for c in candidateClasses:
            indices = [i for i, l in enumerate(labelsOfKNN) if l == c]
            weightedVotes[int(c)] = np.sum(weightsKNN[indices])
        return weightedVotes

    def predictWeightedRegression(self, distances, min_indices, labelsOfKNN):
        weights = self.calcWeights(distances, min_indices)
        numer = np.sum(weights * labelsOfKNN)
        denom = np.sum(weights)
        return numer/denom

    def predict(self, featureRow, out):
        """
        :type featureRow: [float]
        :type out: [float]. After predict f(x), len(out) = 1
        """
        ### Initialize k and other hyperparams. Can also wrap here to test multiple k's
        k = 3
        weighting = False

        ### Measure distance to all stored instances. Keep k nearest
        distances = self.manhattan(np.array(featureRow), self.npFeatures)
        # argpartition sorts only k elements so it's faster than a sort. Returns indices to the k minimum elements in ascending order
        min_indices = np.argpartition(distances, range(k))[:k] 
        labelsOfKNN = self.npLabels[min_indices][:, 0]

        ### k nearest instances vote on output class. Voting scheme depends on if you do weighted voting and if you want knn regression
        pred = None
        if(weighting):
            if (self.regression):
                pred = self.predictWeightedRegression(distances, min_indices, labelsOfKNN)
            else:
                # The highest weighted vote wins
                weightedVotes = self.calcWeightedVotes(distances, min_indices, labelsOfKNN)
                pred = np.argmax(weightedVotes)
        else:
            if (self.regression):
                # Output the mean of the continuous labels
                pred = np.mean(labelsOfKNN)
            else:
                # The most frequent vote wins
                pred, _ = mode(labelsOfKNN)
                pred = pred[0]
                # print('Row: {0} | pred: {1}'.format(self.dataRow, pred))

        # self.dataRow += 1
        del out[:]
        out.append(pred)
                
        

