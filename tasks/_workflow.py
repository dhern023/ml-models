# -*- coding: utf-8 -*-
"""
Workflow functions for bulding tasks
"""
import numpy
import pandas
import pandas_flavor

def split_three_ways(array):
    """
    Does not perform statisfied sampling
    Assumes shuffled
    Splits into 3:1:1 ratio
    """
    percent_60 = int(.6*array.shape[0])
    percent_80 = int(.8*array.shape[0])

    train, validate, test = numpy.split(
        array, [percent_60, percent_80])
    return train, validate, test

def split_two_ways(array, percentage = 0.2):
    """
    Assumes shuffled
    Percentage must be between 0 and 1 (right inclusive)
    TODO: Address bug in which they don't evenly split
    """
    assert 0 < percentage <= 1
    
    train, test = numpy.split(array, percentage)
    
    return train, test

@pandas_flavor.register_dataframe_method
def scale(dataframe, array_scaler=None):
    """
    Must work on numpy arrays (standard)

    Calls scaler function on dataframe.values,
    then creates new dataframe with same columns
    """
    if array_scaler is None:
        return dataframe

    list_columns = dataframe.columns.tolist()

    array = dataframe.values
    array_scaled = array_scaler(array)
    df = pandas.DataFrame(array_scaled, columns = list_columns)
    
    return df