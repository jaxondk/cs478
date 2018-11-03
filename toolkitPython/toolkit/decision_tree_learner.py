from __future__ import (absolute_import, division,
                        print_function, unicode_literals)

from .supervised_learner import SupervisedLearner
from .matrix import Matrix
import numpy as np
import matplotlib.pyplot as plt


class DecisionTreeLearner(SupervisedLearner):
    

    def __init__(self):
        pass

    def train(self, features, labels):
        """
        :type features: Matrix
        :type labels: Matrix
        """
        self.id3(features, labels)

    def predict(self, features, labels):
        """
        :type features: [float]
        :type labels: [float]
        """
        
