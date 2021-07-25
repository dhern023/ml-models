# -*- coding: utf-8 -*-
"""
Useful for displaying all model artifacts.
TODO: Add pandas_profiling
"""
# will replace with plot package
import linear_regression._linear_regression as _lr

import argparse
import pandas
import pathlib
import statsmodels.api as sm

parser = argparse.ArgumentParser(description='Train & analyze a linear regression model.')
parser.add_argument('--data', help='fpath of the dataset')
parser.add_argument('--label', help='The name of the label column')
parser.add_argument('--covariates', nargs='+', help='Optional. Model covariates separate by space')
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

    # Train model
    model_regression = sm.OLS(
        endog = endog,
        exog = exog)
    results_regression = model_regression.fit()

    # Produce artifacts
    print(results_regression.summary())

    _lr.plot_scatter(
        results_regression.fittedvalues,
        results_regression.resid,
        x_label = "Fitted Values",
        y_label = "Residual Values")
    sm.qqplot(results_regression.resid_pearson, line = "q")