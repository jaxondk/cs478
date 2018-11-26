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
        self.k = 5
        randomize = False
        print('K={0}'.format(self.k))
        self.outfile.write('K={0}\n'.format(self.k))
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
        iteration_SSE = np.inf
        i = 1
        centroids = initial_centroids
        while(True):
            # Get the distances from each centroid for every point and assign each point to its closest (minimum distance) cluster
            for c in range(self.k):
                distancesFromCentroid[c] = self.heom(centroids[c], self.npFeatures)
            cluster_assignments = np.argmin(distancesFromCentroid, axis=0)
            cluster_indices = []
            for c in range(self.k):
                curr_cluster_indices = np.where(cluster_assignments == c)[0]
                cluster_indices.append(curr_cluster_indices)

            # Update SSE
            prev_SSE = iteration_SSE
            iteration_SSE, cluster_SSEs = self.calcSSE(distancesFromCentroid, cluster_indices)
            self.printIteration(i, centroids, cluster_indices, iteration_SSE, cluster_SSEs)
            if(iteration_SSE == prev_SSE):
                print('SSE has converged at iteration', i)
                break

            # Calc next centroids
            next_centroids = np.zeros_like(centroids)
            for c in range(self.k):
                curr_cluster = self.npFeatures[cluster_indices[c]]
                next_centroids[c] = self.calcCentroid(curr_cluster)

            centroids = next_centroids
            i += 1
    
    def printIteration(self, i, centroids, cluster_indices, iteration_SSE, cluster_SSEs):
        print('------- Iteration {0} -------'.format(i))
        for c in range(self.k):
            print('--- Cluster {0} ---'.format(c))
            print('Centroid:', centroids[c])
            print('Size of cluster: ', len(cluster_indices[c]))
            print('SSE of cluster: ', cluster_SSEs[c])
        print('Total SSE of Iteration: ', iteration_SSE)
        self.outfile.write('Total SSE of Iteration {0} = {1}\n'.format(i, iteration_SSE))
    
    def calcSSE(self, distances, cluster_indices):
        cluster_SSEs = np.zeros(self.k)
        distances_sqd = distances**2
        for c in range(self.k):
            cluster_SSEs[c] = np.sum(distances_sqd[c, cluster_indices[c]])
        return np.sum(cluster_SSEs), cluster_SSEs
    
    def calcCentroid(self, curr_cluster):
        # Calculate average of each attribute for cluster (ignoring unknowns)
        currentCluster_masked = np.ma.masked_where(curr_cluster == np.inf, curr_cluster)
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
                
        

