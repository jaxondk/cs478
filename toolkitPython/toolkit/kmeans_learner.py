from __future__ import (absolute_import, division, print_function, unicode_literals)

from .supervised_learner import SupervisedLearner
from .matrix import Matrix
import numpy as np
import matplotlib.pyplot as plt
from random import randint

class KmeansLearner(SupervisedLearner):

    def __init__(self):
        pass

    def train(self, features, labels):
        """
        :type features: Matrix
        :type labels: Matrix
        """
        ### Choose k and initialize starting centroids
        k = 5
        randomize = False
        centroids = np.zeros((k, features.cols))
        if(randomize):
            for c in range(len(centroids)):
                centroids[c] = features.row(randint(0, features.rows-1))
        else:
            firstKFeatures = Matrix(features, 0, 0, k, features.cols)
            centroids = np.array(firstKFeatures.data)
        print('centroids', centroids)
        input('pause')


        
        
    # Heterogeneous Euclidean-Overlap Metric, from "Improved Heterogeneous Distance Functions", Journal of Artificial Intelligence Research 6 (1997) 1-34
    def heom(self, p1, storedInstances):
        diff = p1 - storedInstances
        medians = np.median(diff, axis=0)
        for a in range(len(p1)):
            diff[:, a][np.abs(diff[:, a]) == np.inf] = medians[a]

        nominalDiffs = diff[:,self.nominalColumns]
        nominalDiffs[nominalDiffs != 0] = 1
        diff[:, self.nominalColumns] = nominalDiffs
        summation = np.sum(diff**2, axis = 1)
        return np.sqrt(summation)

    def euclidean(self, p1, storedInstances):
        summation = np.sum((p1 - storedInstances)**2, axis=1)
        return np.sqrt(summation)

    def manhattan(self, p1, storedInstances):
        summation = np.sum(np.abs(p1-storedInstances), axis=1)
        return summation
    
    def predict(self, featureRow, out):
        """
        :type featureRow: [float]
        :type out: [float]. After predict f(x), len(out) = 1
        """
        
        pred = None

        del out[:]
        out.append(pred)
                
        

