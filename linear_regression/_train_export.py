#!/usr/bin/env python
# coding: utf-8
"""
Trains and exports model weights to pickle.
Assumes covariate column is needed.
"""
from tasks import _tasks_statsmodels

import argparse
import d6tflow
import pathlib
import statsmodels.api as sm

parser = argparse.ArgumentParser(description='Train & export a linear regression model.')
parser.add_argument('--data', help='fpath of the dataset')
parser.add_argument('--labels', nargs='+', help='Label columns separate by space')
parser.add_argument('--covariates', nargs='+', help='Optional. Model covariates separate by space')
parser.add_argument('--out', default='ols_weights.pickle', help='fpath of the output. Always pickled')
args = parser.parse_args()

def main_statsmodels(args):
    params = {
        'fdata':args.data,
        'add_constant':True,
        'convert_to_one_hot':True,
        'columns_label':args.labels,
        'columns_covariates':args.covariates
        }
    flow = d6tflow.Workflow(
        _tasks_statsmodels.TaskSetupExogEndogData,
        params)
    try:
        flow.run()
    except RuntimeError:
        pass
    return flow

if __name__ == "__main__":

    # Building a model using given covariates
    flow = main_statsmodels(args)
    data = flow.outputLoad(as_dict=True)
    exog = data['data']
    endog = data['labels']

    # Trains model as labeled item
    model_linear_regression = sm.OLS(
        endog = endog,
        exog = exog)
    results_regression = model_linear_regression.fit()

    # Writes model to binary (pickle)
    results_regression.save(pathlib.Path(args.out))