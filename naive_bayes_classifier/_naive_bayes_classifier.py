#!/usr/bin/env python
# coding: utf-8

"""
Naive Bayes multi-class classifier inplemented from scratch.
Handles zero frequency corrections/smoothing.
"""

import collections
import numpy

# Helpers (Priors) ============================================================

def calculate_frequency_average(series):
    """ Calculates probabilites of occurences of each label out of the entire set """
    try:
        series_averages = series.value_counts() / len(series)
        return series_averages.to_dict()
    except ZeroDivisionError as exception:
        raise exception

# Helpers (Dictionaries) ======================================================

def merge_dicts(dict1, dict2):
    dictionary = dict1.copy()
    dictionary.update(dict2)
    return dictionary

def construct_frequency_dict_from_series(series_text):
    """ Converts text to lower-case then returns list of unique words """
    series_counts = (
        series_text
        .str.lower()
        .str.split()
        .explode()
        .value_counts()
        )
    
    return series_counts.to_dict()

def construct_frequency_dict_from_strings(list_strings):
    """ Converts text to lower-case then returns list of unique words """
    string = " ".join(list_strings)
    string = string.lower()
    list_words = string.split()

    return dict(collections.Counter(list_words))

# Model =======================================================================
def calculate_labeled_frequencies(dict_frequencies_text, dataframe_emails, column_label, column_words):
    """ 
    Constructs a frequency dictionary for list of words 
    Uses a processed dataframe with list words column
    
    Handles zero frequency occurences by adding n(w) / total,
    in which n(w) is the number of occurences of the word
    across all text, and total is the total number of words
    in the text.
    
    Parameter
    ----------
    dict_frequencies_text = { word : n(word) }
    
    """
    list_labels = dataframe_emails[column_label].unique()
    total = sum(dict_frequencies_text.values())
    
    # Doing it this way avoids copy errors with nested dictionaries
    model = {}
    for word in dict_frequencies_text.keys():
        model.setdefault(word, {label : 0 for label in list_labels })

    # Split label column into groups so we can count them directly
    group_labels = dataframe_emails.groupby(column_label)
    for label, label_df in group_labels:
        for list_words in label_df[column_words]:
            for word in list_words:
                model[word][label] += 1
                
    # Handles the zero frequency offset
    for word, dict_frequency in model.items():
        offset = max(dict_frequencies_text[word] / total, 1E-8)
        for key in dict_frequency.keys():
            dict_frequency[key] += offset

    return model

def predict_bayes(word, label, dict_frequencies):
    """ 
    Doesn't use the naive assumption.
    likelihood = (num labeled emails with word) / sum(num labeled emails with word for all labels)
               = P(A | Event_j) / sum ( P (A | Event_i) )
    """
    label_count = dict_frequencies[word][label]
    all_label_counts = sum(dict_frequencies[word].values())

    try:
        return label_count / all_label_counts
    except ZeroDivisionError:
        return 0

# Helpers (Probabilities) =====================================================

def calculate_list_product(list_):
    """ Slightly faster to work on arrays than directly with list """
    return numpy.array(list_).prod()

# Naive Bayes Classifier ======================================================

def setup_naive_bayes(dict_frequencies_new_text, dataframe_emails, column_label, column_words, column_text):
    """ 
    Adds new text to the model, so it can be used to make new predictions 
    This is mostly just an accumulation of the previous cells.
    """
    
    dict_frequencies_whole_text = construct_frequency_dict_from_series(dataframe_emails[column_text])

    dict_model = calculate_labeled_frequencies(
        merge_dicts(dict_frequencies_whole_text, dict_frequencies_new_text), 
        dataframe_emails, 
        column_label, 
        column_words)

    return dict_model

def calculate_naive_bayes(list_words, dict_frequencies, series_labels):
    list_labels = series_labels.unique()
    counts_label = series_labels.value_counts()
    total = len(series_labels)
    
    # Cook up the total number of emails in each label
    dict_naive_bayes = { label : 1 for label in list_labels }
    for word in list_words:
        for label in list_labels:
            probability = dict_frequencies[word][label] / counts_label[label]            
            if probability == 0:
                print(word)
            dict_naive_bayes[label] *= (probability * total)

    # Multiply by the total number of elements for each label 
    for label in list_labels:
        dict_naive_bayes[label] *= counts_label[label]
    
    return dict_naive_bayes

def predict_naive_bayes(email, label_to_predict, dict_frequencies, series_labels):
    """ Uses the naive assumption to predict on a given email """

    # words
    email = email.lower()
    words = set(email.split())

    dict_naive_bayes = calculate_naive_bayes(words, dict_frequencies, series_labels)
    
    numerator = dict_naive_bayes[label_to_predict]
    denominator = sum(dict_naive_bayes.values())

    return numerator/denominator



    
