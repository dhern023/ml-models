#!/usr/bin/env python
# coding: utf-8
"""
Trains and exports model weights to pickle.
Assumes covariate column is needed.
"""

import argparse
import pandas
import pathlib
import statsmodels.api as sm

parser = argparse.ArgumentParser(description='Train & export a linear regression model.')
parser.add_argument('--data', help='fpath of the dataset')
parser.add_argument('--label', help='The name of the label column')
parser.add_argument('--covariates', nargs='+', help='Optional. Model covariates separate by space')
parser.add_argument('--out', default='ols_weights.pickle', help='fpath of the output. Always pickled')
args = parser.parse_args()

if __name__ == "__main__":

    data = pandas.read_csv(pathlib.Path(args.data))

    # Building a model using given covariates
    exog = data.copy()
    exog = sm.add_constant(exog) # adds an intercept column
    exog = pandas.get_dummies(exog) # Converts categorical to one-hot
    endog = exog.pop(args.label)

    # Uses all covaraites by default.
    if args.covariates:
        exog = exog[args.covariates]

    # Trains model as labeled item
    model_linear_regression = sm.OLS(
        endog = endog,
        exog = exog)
    results_regression = model_linear_regression.fit()

    # Writes model to binary (pickle)
    results_regression.save(pathlib.Path(args.out))