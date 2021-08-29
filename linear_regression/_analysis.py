# -*- coding: utf-8 -*-
"""
Useful for displaying all model artifacts.
TODO: Add pandas_profiling
"""
# will replace with plot package
import linear_regression._linear_regression as _lr
from tasks import _tasks_statsmodels

import argparse
import d6tflow
import statsmodels.api as sm

parser = argparse.ArgumentParser(description='Train & analyze a linear regression model.')
parser.add_argument('--data', help='fpath of the dataset')
parser.add_argument('--labels', nargs='+', help='Label columns separate by space')
parser.add_argument('--covariates', nargs='+', help='Optional. Model covariates separate by space')
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