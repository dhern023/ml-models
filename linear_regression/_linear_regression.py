#!/usr/bin/env python
# coding: utf-8
"""
A refactor of linear regression from 
    Groken Machine Learning Chapter 3: 
    Linear Regression for a housing dataset
    by Luis Serrano
"""

# ### Importing the necessary packages

from matplotlib import pyplot as plt
import numpy
import tqdm # progress bar

# ### Plotting functions

def plot_scatter(x_iterable, y_iterable, x_label = "", y_label = ""):
    x_array = numpy.array(x_iterable)
    y_array = numpy.array(y_iterable)
    plt.scatter(x_array, y_array)
    plt.xlabel(x_label)
    plt.ylabel(y_label)

def draw_line(slope, y_intercept, color='grey', linewidth=0.7, starting=0, ending=8):
    x = numpy.linspace(starting, ending, 1000)
    plt.plot(x, y_intercept + slope*x, linestyle='-', color=color, linewidth=linewidth)

# ### Error metrics

def rmse(labels, predictions):
    """ The root mean square error function """
    n = len(labels)
    differences = numpy.subtract(labels, predictions)
    return numpy.sqrt(1.0/n * (numpy.dot(differences, differences)))

# ### Coding the tricks

# - Simple trick
# - Absolute trick
# - Square trick

def simple_trick(bias, slope, predictor, current_value):
    """
    y ~ bias + slope*predictor

    The simple trick performs small pertubations
    """
    small_random_1 = numpy.random.random()*0.1
    small_random_2 = numpy.random.random()*0.1

    if predictor == 0:
        return slope, bias

    predicted_value = bias + slope*predictor
    if current_value > predicted_value:
        bias += small_random_2
        if predictor > 0:
            slope += small_random_1
        elif predictor < 0:
            slope -= small_random_1
    if current_value < predicted_value:
        slope -= small_random_1
        if predictor > 0:
            bias -= small_random_2
        elif predictor < 0:
            bias += small_random_2

    return slope, bias

def absolute_trick(bias, slope, predictor, current_value, learning_rate):
    """
    y ~ bias + slope*predictor

    Performs increments wrt a given scalar
    """
    predicted_value = bias + slope*predictor
    if current_value > predicted_value:
        slope += learning_rate*predictor
        bias += learning_rate
    else:
        slope -= learning_rate*predictor
        bias -= learning_rate
    return slope, bias

def square_trick(bias, slope, predictor, current_value, learning_rate):
    """
    y ~ bias + slope*predictor

    Performs increments wrt a given scalar and difference
    """
    predicted_value = bias + slope*predictor
    slope += learning_rate*predictor*(current_value-predicted_value)
    bias += learning_rate*(current_value-predicted_value)
    return slope, bias


# ### Running the linear regression algorithm

def perform_one_epoch(bias, slope, predictor, current_value, trick_function, learning_rate, ):
    """
    There's probably a better way to do this with kwargs
    """
    if learning_rate:
        slope, bias = trick_function(
            bias, slope, predictor, current_value, learning_rate=learning_rate)
    else:
        slope, bias = trick_function(
            bias, slope, predictor, current_value)
    return slope, bias

def linear_regression(
        features,
        labels,
        trick_function = absolute_trick,
        learning_rate=0.01,
        error_metric = rmse,
        epochs = 1000,
        plot_all_epochs = True):
    """
    The linear regression algorithm consists of:
    - Starting with random weights
    - Iterating the square (or simple, or absolute) trick many times.
    - Plotting the error function

    trick_function must follow y ~ b0 + b1x with parameters:
        bias, 
        slope, 
        predictor, 
        current_value,
        learning_rate (optional for simple_trick)
    error_metric must take two arrays and return a scalar
    
    """

    slope = numpy.random.random()
    bias = numpy.random.random()
    errors = []
    for epoch in tqdm.tqdm(range(epochs)):
        if plot_all_epochs:
            draw_line(slope, bias, starting=min(features)-1, ending=max(features)+1)
        predictions = features[0]*slope+bias
        errors.append(error_metric(labels, predictions))
        index_random = numpy.random.randint(0, len(features)-1)
        predictor = features[index_random]
        current_value = labels[index_random]
        slope, bias = perform_one_epoch(
            bias, slope, predictor, current_value, trick_function, learning_rate)
    draw_line(slope, bias, 'black', starting=0, ending=9)
    plot_scatter(features, labels)
    plt.show()
    plt.scatter(range(len(errors)), errors)
    plt.show()
    return slope, bias

def predict_linear_regression_labaled_features(fitted_model, dict_features):
    """ 
    Calculates y ~ const + sum( parameter*value )
    essentially the inner product

    { 'feature name' : value }
    
    Does not assume you have all features present, so prediction may be off.
    Assumes const parameter is not present in dictionary
    """
    list_given_terms = [
        fitted_model.params[key]*value for key, value in dict_features.items()
    ]
    if 'const' not in dict_features:
        constant_value = fitted_model.params['const']
        list_given_terms.append(constant_value)
    
    return sum(list_given_terms)

def predict_linear_regression(fitted_model, array):
    """
    It's possible the array doesn't have enough values
    Assumes array does not have a constant term
    """
    if len(array) == fitted_model.params - 1:
        array = numpy.concatenate(([1],array))
    
    return fitted_model.params @ array
