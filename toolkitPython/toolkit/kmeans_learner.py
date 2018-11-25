from __future__ import (absolute_import, division, print_function, unicode_literals)

from .supervised_learner import SupervisedLearner
from .matrix import Matrix
import numpy as np
import matplotlib.pyplot as plt
from random import randint


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

class KmeansLearner(SupervisedLearner):

    def __init__(self):
        pass

    def train(self, features, labels):
        """
        :type features: Matrix
        :type labels: Matrix
        """
        self.npFeatures = np.array(features.data)
        
        ### Discover nominal columns for handling distance of nominal attributes
        self.nominalColumns = []
        for c in range(features.cols):
            if(features.value_count(c) != 0):
                self.nominalColumns.append(c)
        self.nominalColumns = np.array(self.nominalColumns)

        ### Choose k and initialize starting centroids
        k = 5
        randomize = False
        centroids = np.zeros((k, features.cols))
        if(randomize):
            for c in range(k):
                centroids[c] = features.row(randint(0, features.rows-1))
        else:
            firstKFeatures = Matrix(features, 0, 0, k, features.cols)
            centroids = np.array(firstKFeatures.data)

        ### K-means algorithm
        prev_SSE = np.inf
        current_SSE = 999999999
        distancesFromCentroid = np.zeros((k, features.rows))
        # clusters = np.full((k, features.rows), False) # point p is a member of cluster c if clusters[c,p] = True
        while(current_SSE != prev_SSE):
            # Get the distances from each centroid for every point
            for c in range(k):
                distancesFromCentroid[c] = self.heom(centroids[c], self.npFeatures)

            # Assign each point to its closest (minimum distance) cluster and recalculate centroids
            clusterAssignments = np.argmin(distancesFromCentroid, axis=0)
            for c in range(k):
                currentClusterPtIndices = np.where(clusterAssignments == c)[0]
                currentCluster = self.npFeatures[currentClusterPtIndices]
                centroids[c] = self.calcCentroid(currentCluster)

            # Update SSE
            print('centroids', centroids)
            input('pause')

    def calcCentroid(self, currentCluster):
        # Calculate average of each attribute for cluster (ignoring unknowns)
        currentCluster_masked = np.ma.masked_where(currentCluster == np.inf, currentCluster)
        centroid = np.average(currentCluster_masked, axis=0)
        # Replace avgs for nominal attributes with their modes
        centroid[self.nominalColumns], _ = mode(currentCluster_masked[:,self.nominalColumns])
        centroid.fill_value = np.inf
        return centroid.filled() # unmask so we have inf back

    # Heterogeneous Euclidean-Overlap Metric, from "Improved Heterogeneous Distance Functions", Journal of Artificial Intelligence Research 6 (1997) 1-34
    def heom(self, p1, storedInstances):
        diff = p1 - storedInstances
        diff[np.abs(diff) == np.inf] = 1
        diff[np.isnan(diff)] = 1 # NaN's will come up if inf - inf occurs 
        nominalDiffs = diff[:,self.nominalColumns]
        nominalDiffs[nominalDiffs != 0] = 1
        diff[:, self.nominalColumns] = nominalDiffs
        summation = np.sum(diff**2, axis = 1)
        return np.sqrt(summation)
    
    def predict(self, featureRow, out):
        """
        :type featureRow: [float]
        :type out: [float]. After predict f(x), len(out) = 1
        """
        
        pred = None

        del out[:]
        out.append(pred)
                
        

