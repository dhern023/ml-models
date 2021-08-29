# -*- coding: utf-8 -*-
"""
Verifyies if two labels are linearly separable:
    aka, the perceptron algorithm converges
Also useful for displaying all artifacts.
"""
from tasks import _tasks_statsmodels
from _perceptron import (
    # Will replace plot package
    plot_plane,
    plot_scatter,
    draw_line)

import argparse
import d6tflow
from matplotlib import pyplot as plt
import statsmodels.api as sm

parser = argparse.ArgumentParser(description='Train & analyze a perceptron model.')
parser.add_argument('--data', help='fpath of the dataset')
parser.add_argument('--labels', nargs='+', help='Label columns separate by space')
parser.add_argument('--covariates', nargs='+', help='Optional. Model covariates separate by space')
args = parser.parse_args()

def main_statsmodels(args):
    params = {
        'fdata':args.data,
        'add_constant':True,
        'convert_to_one_hot':False,
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

    # Train Model
    model_logistic_regression = sm.Logit(
        endog = endog,
        exog = exog)
    model_logistic_regression.raise_on_perfect_prediction = False
    results_regression = model_logistic_regression.fit()

    # Produce artifacts
    print(results_regression.summary())

    if results_regression.params.shape[0] == 3:
        plot_scatter(exog.iloc[:,1], exog.iloc[:,2], c = endog)
        draw_line(
            -results_regression.params[2]/results_regression.params[1],
            -results_regression.params[0]/results_regression.params[1],
            starting=0,
            ending=sum(endog),
            color='grey', linewidth=1.0, linestyle='dotted')
        plt.legend(["Separating line"])
        plt.show()

    elif results_regression.params.shape[0] == 4:
        ax = plot_plane(
            results_regression.params[1:],
            results_regression.params[0])
        ax.scatter(exog.iloc[:,1], exog.iloc[:,2], exog.iloc[:,3], c = endog)
        plt.show()

    else: # 4D plane or higher
        statement_dimensions = "Too many dimensions to plot"
        print(statement_dimensions)