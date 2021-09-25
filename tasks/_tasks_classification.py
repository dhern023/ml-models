# -*- coding: utf-8 -*-
"""
Common sklearn classifier models

A good place for Step 5. Create model task (left up to user)
"""

import d6tflow
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.neural_network import MLPClassifier

class TaskSklearnSVC(d6tflow.tasks.TaskCache):
    """ Support Vector Machine Classifier """

    kernel = d6tflow.Parameter(default = 'linear')

    def run(self):
        model_svm = SVC(kernel = self.kernel)

        self.save(model_svm)

class TaskSklearnKNN(d6tflow.tasks.TaskCache):
    """ K-Nearest Neighbors """

    def run(self):
        model_knn = KNeighborsClassifier()

        self.save(model_knn)

class TaskSklearnRFC(d6tflow.tasks.TaskCache):
    """Random Forest Classifier"""

    n_estimators = d6tflow.IntParameter(default = 100)
    criterion = d6tflow.Parameter(default = 'gini')
    min_samples_split = d6tflow.FloatParameter(default = 0.1)
    oob_score = d6tflow.BoolParameter(default = True)

    def run(self):
        model = RandomForestClassifier(
            n_estimators = self.n_estimators,
            criterion = self.criterion,
            min_samples_split = self.min_samples_split,
            oob_score = self.oob_score
            )

        self.save(model)

class TaskSklearnSGD(d6tflow.tasks.TaskCache):
    """ Stochastic Gradient Descent for Logistic Regression """

    loss = d6tflow.Parameter(default = 'hinge')
    penalty = d6tflow.Parameter(default = 'l2')
    max_iter_sgd = d6tflow.IntParameter(default = 1000)
    alpha = d6tflow.FloatParameter(default = 0.0001)

    def run(self):
        model = SGDClassifier(
            loss = self.loss,
            penalty = self.penalty,
            max_iter = self.max_iter_sgd,
            alpha = self.alpha
            )

        self.save(model)

class TaskSklearnMLP(d6tflow.tasks.TaskCache):
    """Multi-Layer Perceptron"""

    hidden_layer_sizes = d6tflow.TupleParameter(default = (20,200))
    max_iter_mlp = d6tflow.IntParameter(default = 5000)

    def run(self):
        model = MLPClassifier(
            hidden_layer_sizes = self.hidden_layer_sizes,
            max_iter = self.max_iter_mlp
            )

        self.save(model)