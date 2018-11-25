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
        self.outfile = open('kmeans.out', 'a')
        self.npFeatures = np.array(features.data)

        ### Discover nominal columns for handling distance of nominal attributes
        self.nominalColumns = []
        for c in range(features.cols):
            if(features.value_count(c) != 0):
                self.nominalColumns.append(c)
        self.nominalColumns = np.array(self.nominalColumns)

        ### Choose k and initialize starting centroids
        self.k = 7
        print('K={0}'.format(self.k))
        self.outfile.write('K={0}\n'.format(self.k))
        randomize = True
        initial_centroids = np.zeros((self.k, features.cols))
        if(randomize):
            for c in range(self.k):
                initial_centroids[c] = features.row(randint(0, features.rows-1))
        else:
            firstKFeatures = Matrix(features, 0, 0, self.k, features.cols)
            initial_centroids = np.array(firstKFeatures.data)

        self.kmeans(initial_centroids)
        self.outfile.close()

    def kmeans(self, initial_centroids):
        distancesFromCentroid = np.zeros((self.k, len(self.npFeatures)))
        current_SSE = np.inf
        i = 1
        centroids = initial_centroids
        while(True):
            # Get the distances from each centroid for every point and assign each point to its closest (minimum distance) cluster
            for c in range(self.k):
                distancesFromCentroid[c] = self.heom(centroids[c], self.npFeatures)
            clusterAssignments = np.argmin(distancesFromCentroid, axis=0)

            # Update SSE
            prev_SSE = current_SSE
            current_SSE, clusterSSEs = self.calcSSE(distancesFromCentroid, clusterAssignments)
            self.printIteration(i, centroids, clusterAssignments, current_SSE, clusterSSEs)
            if(current_SSE == prev_SSE):
                print('SSE has converged at iteration', i)
                break

            # Calc next centroids
            next_centroids = np.zeros_like(centroids)
            for c in range(self.k):
                currentClusterPtIndices = np.where(clusterAssignments == c)[0]
                currentCluster = self.npFeatures[currentClusterPtIndices]
                next_centroids[c] = self.calcCentroid(currentCluster)

            centroids = next_centroids
            i += 1
    
    def printIteration(self, i, centroids, clusterAssignments, current_SSE, clusterSSEs):
        print('------- Iteration {0} -------'.format(i))
        for c in range(self.k):
            print('--- Cluster {0} ---'.format(c))
            print('Centroid:', centroids[c])
            print('Size of cluster: ', len(np.where(clusterAssignments == c)[0]))
            print('SSE of cluster: ', clusterSSEs[c])
        print('Total SSE of Iteration: ', current_SSE)
        self.outfile.write('Total SSE of Iteration {0} = {1}\n'.format(i, current_SSE))

    def calcSSE(self, distances, clusterAssignments):
        clusterSSEs = np.zeros(self.k)
        for c in range(self.k):
            currentClusterPtIndices = np.where(clusterAssignments == c)[0]
            clusterSSEs[c] = np.sum(distances[c, currentClusterPtIndices])
        return np.sum(clusterSSEs), clusterSSEs
    
    def calcCentroid(self, currentCluster):
        # Calculate average of each attribute for cluster (ignoring unknowns)
        currentCluster_masked = np.ma.masked_where(currentCluster == np.inf, currentCluster)
        centroid = np.average(currentCluster_masked, axis=0)
        # Replace avgs for nominal attributes with their modes
        if(len(self.nominalColumns) > 0):
            centroid[self.nominalColumns], _ = mode(currentCluster_masked[:,self.nominalColumns])
        centroid.fill_value = np.inf
        return centroid.filled() # unmask so we have inf back

    # Heterogeneous Euclidean-Overlap Metric, from "Improved Heterogeneous Distance Functions", Journal of Artificial Intelligence Research 6 (1997) 1-34
    def heom(self, p1, storedInstances):
        diff = p1 - storedInstances
        diff[np.abs(diff) == np.inf] = 1
        diff[np.isnan(diff)] = 1 # NaN's will come up if inf - inf occurs 
        if(len(self.nominalColumns) > 0):
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
        
        pred = 1

        del out[:]
        out.append(pred)
                
        

