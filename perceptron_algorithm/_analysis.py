# -*- coding: utf-8 -*-
"""
Verifyies if two labels are linearly separable:
    aka, the perceptron algorithm converges
Also useful for displaying all artifacts.
"""

from _perceptron import (
    Perceptron,
    # Will replace plot package
    plot_plane,
    plot_scatter,
    draw_line)

import argparse
from matplotlib import pyplot as plt
import pandas
import pathlib

parser = argparse.ArgumentParser(description='Train & analyze a perceptron model.')
parser.add_argument('--data', help='fpath of the dataset')
parser.add_argument('--label', help='The name of the label column')
parser.add_argument('--covariates', nargs='+', help='Optional. Model covariates separate by space')
parser.add_argument('--learning_rate', default=0.01, type=float, help='The learning rate')
parser.add_argument('--max_iter', default=200, type=int, help='The maximum number of iterations')
args = parser.parse_args()

if __name__ == "__main__":

    data = pandas.read_csv(pathlib.Path(args.data))

    # Building a model using given covariates
    exog = data.copy()
    endog = exog.pop(args.label).values

    # Uses all covaraites by default.
    if args.covariates:
        statement_converged = "Perceptron converged: {} are linearly separable".format(args.covariates)
        statement_diverged = "Perceptron converged: {} are linearly separable".format(args.covariates)
        exog = exog[args.covariates].values

    else:
        statement_converged = "Perceptron converged: {} are linearly separable".format(exog.columns.tolist())
        statement_diverged = "Perceptron converged: {} are linearly separable".format(exog.columns.tolist())
        exog = exog.values

    # Train Model
    instance_perceptron = Perceptron()
    weights, bias, errors = instance_perceptron.perceptron_algorithm(
        exog,
        endog,
        args.learning_rate,
        args.max_iter)

    # Produce artifacts
    if 0 in errors:        
        print(statement_converged)
    else:
        print(statement_diverged)
    
    if weights.shape[0] == 2:
        plot_scatter(exog[:,0], exog[:,1], c = endog)
        draw_line(
            -weights[0]/weights[1],
            -bias/weights[1],
            ending = sum(endog),
            color='grey', linewidth=1.0, linestyle='dotted')
        plt.legend(["Separating line"])
        plt.show()

    elif weights.shape[0] == 3:
        ax = plot_plane(weights, bias)
        ax.scatter(exog[:,0], exog[:,1], exog[:,2], c = endog)
        plt.show()
    
    else: # 4D plane or higher
        statement_dimensions = "Too many dimensions to plot"
        print(statement_dimensions)