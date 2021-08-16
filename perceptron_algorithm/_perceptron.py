#!/usr/bin/env python
# coding: utf-8

"""
A refactor of Binary Perception algorithm from 
    Groken Machine Learning Chapter 5: 
    Sentiment analysis with the perceptron algorithm
    by Luis Serrano

Can handle multi-dimensional features
"""

# Importing packages
from matplotlib import pyplot as plt
import numpy
import tqdm

# Helpers (Plotting) =========================================

def plot_scatter(x_iterable, y_iterable, x_label = "", y_label = "",  legend = None, **kwargs):
    x_array = numpy.array(x_iterable)
    y_array = numpy.array(y_iterable)
    plt.xlabel(x_label)
    plt.xlabel(y_label)
    if legend is not None:
        plt.legend(legend)
    plt.scatter(x_array, y_array, **kwargs)

def plot_plane(normal, constant, **kwargs):
    """
    Normal vector is the vector normal to the plane

    Obviously trying more than three dimemsions will cause some trouble
    """
    # Add an axes by creating x1, ..., x_n-1
    num_dimensions = len(normal) - 1
    grid_points = numpy.meshgrid(*[range(10)]*num_dimensions)
    grid_points = numpy.array(grid_points)
    # calculate corresponding z = -sum( xi*ai )/xn
    z = 0
    for index, coefficient in enumerate(normal[:-1]):
        z += coefficient*grid_points[index]
    z = numpy.divide(z, normal[-1])
    z = numpy.add(z, constant)
    z = -z

    fig = plt.figure()
    ax = fig.add_subplot(111,projection='3d')

    # plot the surface
    ax.plot_surface(*grid_points, z, alpha=0.2)

    return ax

def draw_line(slope, y_intercept, starting=0, ending=8, **kwargs):
    x = numpy.linspace(starting, ending, 1000)
    plt.plot(x, y_intercept + slope*x, **kwargs)

class Perceptron():
    """
    The class interface just makes it easier to namespace
    The amount of lines of code saved is pretty nil.    
    """
    def __init__(self):
        self.array_weights = None
        self.bias = None

# Helpers (Perceptron) =======================================

    @staticmethod
    def step(scalar):
        if scalar >= 0:
            return 1
        else:
            return 0

    def calculate_score(self, array_feature):
        """
        Utilizes the dot function because numpy allows
        vector.dot(scalar) operations
        """
        return array_feature.dot(self.array_weights) + self.bias

    def prediction(self, array_feature):
        score = self.calculate_score(array_feature)
        return Perceptron.step(score)

    def calculate_error(self, array_feature, label):
        """ Correct predictions should have no effect on our adjustment score """
        pred = self.prediction(array_feature)
        if pred == label:
            return 0
        else:
            score = self.calculate_score(array_feature)
            return numpy.abs(score)

# Helpers (Metrics) ==========================================

    def calculate_mean_perceptron_error(self, array_features, array_labels):
        """
        Mean error in this case measures how well the entire line (plane) splits the data
        The lower, the better.
        """
        assert array_features.shape[0] == array_labels.shape[0]
    
        total_error = 0
        for feature, label in zip(array_features, array_labels):
            total_error += self.calculate_error(feature, label)
        return total_error/array_features.shape[0]

# Model ======================================================

    def update_weights_bias(self, array_feature, label, learning_rate = 0.01):
        """
        Perceptron trick v2.
        Shorter version of the perceptron trick taking full advantage of the fact:
            new weights = old weights + learning_rate * (label – prediction) * feature
            bias += learning_rate * (label – prediction)
        """
    
        pred = self.prediction(array_feature)
        self.array_weights = numpy.add(
            self.array_weights, (label-pred)*array_feature*learning_rate
            )
        self.bias += (label-pred)*learning_rate

    def perceptron_algorithm(self, array_features, array_labels, learning_rate = 0.01, num_epochs = 200):
        """
        Loop breaks when converges or if num_epochs is reached
    
        Stores the best weights and bias in case of non-convergence
        """
        assert array_features.shape[0] == array_labels.shape[0]
        
        # Initialize the weights and bias in each run
        self.array_weights = numpy.ones(shape = array_features.shape[1])
        self.bias = 0        

        best_weights = None
        best_bias = None
        iter_errors = []

        # base case
        count = 0
        error = self.calculate_mean_perceptron_error(
            array_features, array_labels)
        iter_errors = [error]
    
        progress_bar = tqdm.tqdm(total = num_epochs)
        while (error >= 1e-16) and (count <= num_epochs):
    
            error = self.calculate_mean_perceptron_error(
                array_features, array_labels)
    
            # Identifies best weights
            if error < iter_errors[-1]:
                best_weights = self.array_weights
                best_bias = self.bias
            iter_errors.append(error)
    
            # Updates weights & bias
            index = numpy.random.randint(0, array_features.shape[0] - 1)
            self.update_weights_bias(
                array_features[index], 
                array_labels[index],
                learning_rate)
    
            count +=1
    
        progress_bar.close()
    
        # Plotting error
        plot_scatter(range(len(iter_errors)), iter_errors)
        plt.title("Mean Perception Error per Iteration")
        plt.show()
        
        return best_weights, best_bias, iter_errors
