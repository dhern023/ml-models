# -*- coding: utf-8 -*-
"""
Tasks for building machine learning pipelines via Sklearn API

Assumes the following workflow:
    0. Process & pickle your dataset
    1. Load
    2. Setup covariates and labels
    3. Split data
    4. Scale data and keep labels intact
    5. Create model task (left up to user)
    6. Train
    
Notes:
    Useful for models with a single label column to take 
    advantage of straifying and confusion matrix
"""
import _workflow

import d6tflow
import pandas
import pathlib
import sklearn.preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report#, confusion_matrix

class TaskLoadDataframe(d6tflow.tasks.TaskCachePandas):
    """
    Loads in a processed dataframe.

    Can handle if the dataframe have been serialized as a pickle
    TODO: Handle JSON
    """
    fpath=d6tflow.Parameter()

    def run(self):
        fname = pathlib.Path(self.fpath)
        if fname.suffix == '.pkl':
            dataframe = pandas.read_pickle(fname)
        else: # deafult to csv reader
            dataframe = pandas.read_csv(fname)
        self.save(dataframe)

@d6tflow.requires(TaskLoadDataframe)
class TaskSetupData(d6tflow.tasks.TaskCache):
    """
    Will use all non label columns as covariates unless specified otherwise
    """

    columns_label = d6tflow.ListParameter()
    # Optional
    shuffle = d6tflow.BoolParameter(default = False)
    columns_covariates = d6tflow.ListParameter(default=None)

    persist=['data', 'labels']

    def run(self):
        dataframe = self.inputLoad()

        if self.shuffle:
            dataframe = dataframe.sample(frac=1).reset_index(drop=True)

        # Building a model using given covariates
        dataframe_labels = dataframe.loc[:, list(self.columns_label)]
        dataframe_data = dataframe.drop(columns=list(self.columns_label))

        # Uses all covaraites by default.
        if self.columns_covariates:
            dataframe_data = dataframe_data[list(self.columns_covariates)]


        self.save( {'data' : dataframe_data, 'labels' : dataframe_labels } )

@d6tflow.requires({'inputs' : TaskSetupData})
class TaskSplitData(d6tflow.tasks.TaskCache):
    """
    TODO: Add stratification, which
        places the given percentage of each label in test set
    """
    split_three_ways = d6tflow.BoolParameter(default = True)
    split_percentage = d6tflow.FloatParameter(default = 0.2)
    random_state = d6tflow.IntParameter(default = 1)

    def run(self):
        data = self.inputLoad(as_dict=True)
        dataframe_data = data['inputs']['data']
        dataframe_labels = data['inputs']['labels']

        dict_datasets = {}

        if self.split_three_ways:
            dict_datasets['train_data'],\
                dict_datasets['valid_data'],\
                    dict_datasets['eval_data'] = _workflow.split_three_ways(dataframe_data)
            dict_datasets['train_labels'],\
                dict_datasets['valid_labels'],\
                    dict_datasets['eval_labels'] = _workflow.split_three_ways(dataframe_labels)

        else: # split two ways
            dict_datasets['train_data'],\
            dict_datasets['eval_data'],\
            dict_datasets['train_labels'],\
            dict_datasets['eval_labels'] = train_test_split(
                dataframe_data,
                dataframe_labels,
                test_size = self.split_percentage,
                #stratify=dataframe_labels)
                )

        self.save(dict_datasets)

@d6tflow.requires({'datasets': TaskSplitData})
class TaskScaleData(d6tflow.tasks.TaskCache):

    scaler_method = d6tflow.Parameter(default=None)

    def run(self):
        data = self.inputLoad(as_dict=True)
        dict_datasets = {}

        if self.scaler_method == 'min_max':
            metric = sklearn.preprocessing.MinMaxScaler()

        elif self.scaler_method == 'standard':
            metric = sklearn.preprocessing.StandardScaler()

        else:
            metric = None

        dict_datasets['train_data'] = data['datasets']['train_data'].scale(metric.fit_transform)
        dict_datasets['eval_data'] = data['datasets']['eval_data'].scale(metric.fit_transform)
        dict_datasets['train_labels'] = data['datasets']['train_labels']
        dict_datasets['eval_labels'] = data['datasets']['eval_labels']

        self.save(dict_datasets)

@d6tflow.requires({'datasets': TaskScaleData})
class TaskFitSklearnModel(d6tflow.tasks.TaskCache):
    """ 
    Fit sklearn model 
    
    TODO: Save artifacts as part of run
    TODO: Figure out how to include model_task as part of task list
    """

    model_task = d6tflow.TaskParameter()

    def run(self):
        data = self.inputLoad(as_dict=True)
        # Hackery
        model_task = self.model_task
        model_task.run()
        model = model_task.outputLoad()

        # Train model
        model.fit(
            data['datasets']['train_data'].values,
            data['datasets']['train_labels'].values.ravel())
        predictions = model.predict(data['datasets']['eval_data'])
        score = model.score(
            data['datasets']['eval_data'],
            data['datasets']['eval_labels'])
        print('Accuracy is ', score)

        # Collect metrics
        report = classification_report(
            data['datasets']['eval_labels'],
            predictions)
        print(report)

        self.save(model)