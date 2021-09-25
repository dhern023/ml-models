# -*- coding: utf-8 -*-
"""
Tasks for building statsmodel pipelines
"""

import _workflow

import d6tflow
import pandas
import pathlib
import statsmodels.api as sm

class TaskLoadDataframe(d6tflow.tasks.TaskCachePandas):
    """
    Loads in a processed dataframe.

    Can handle if the dataframe have been serialized as a pickle
    TODO: Handle JSON
    """
    fdata=d6tflow.Parameter()

    def run(self):
        fname = pathlib.Path(self.fdata)
        if fname.suffix == '.pkl':
            dataframe = pandas.read_pickle(fname)
        else: # deafult to csv reader
            dataframe = pandas.read_csv(fname)
        self.save(dataframe)

@d6tflow.requires(TaskLoadDataframe)
class TaskSetupExogEndogData(d6tflow.tasks.TaskCache):
    """
    Will use all non label columns as covariates unless specified otherwise
    Optional: Convert categorical to one-hot for regression
    """

    add_constant = d6tflow.BoolParameter(default = True)
    convert_to_one_hot = d6tflow.BoolParameter(default=False)
    columns_label = d6tflow.ListParameter()
    columns_covariates = d6tflow.ListParameter(default=None)

    persist=['data', 'labels']

    def run(self):
        dataframe = self.inputLoad()

        exog = dataframe.copy()
        endog = exog.loc[:, list(self.columns_label)]
        exog = exog.drop(columns=list(self.columns_label))

        # Uses all covaraites by default.
        if self.columns_covariates:
            exog = exog[list(self.columns_covariates)]

        if self.add_constant:
            exog = sm.add_constant(exog) # adds an intercept column
        if self.convert_to_one_hot:
            exog = pandas.get_dummies(exog) # Converts categorical to one-hot

        self.save( {'data' : exog, 'labels' : endog } )

@d6tflow.requires({'inputs' : TaskSetupExogEndogData})
class TaskSplitData(d6tflow.tasks.TaskCache):
    split_three_ways = d6tflow.BoolParameter(default = True)
    split_percentage = d6tflow.FloatParameter(default = 0.2)

    def run(self):
        data = self.inputLoad(as_dict=True)
        array_data = data['inputs']['data']
        array_labels = data['inputs']['labels']

        dict_datasets = {}

        if self.split_three_ways:
            dict_datasets['train_data'],\
                dict_datasets['valid_data'],\
                    dict_datasets['eval_data'] = _workflow.split_three_ways(array_data)
            dict_datasets['train_labels'],\
                dict_datasets['valid_labels'],\
                    dict_datasets['eval_labels'] = _workflow.split_three_ways(array_labels)

        else: # split two ways
            dict_datasets['train_data'],\
                dict_datasets['eval_data'] = _workflow.split_two_ways(array_data, self.split_percentage)
            dict_datasets['train_labels'],\
                    dict_datasets['eval_labels'] = _workflow.split_two_ways(array_labels, self.split_percentage)

        self.save(dict_datasets)

