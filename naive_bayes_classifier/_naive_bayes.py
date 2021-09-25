# -*- coding: utf-8 -*-
#!/usr/bin/env python

"""
Naive Bayes multi-class classifier inplemented from scratch.
Handles zero frequency corrections/smoothing.

TODO: See if I can remove the pandas usage and go full numpy
"""

import collections
import pandas
import numpy

# Helpers ============================================================
def calculate_frequency_average(series):
    """ Calculates probabilites of occurences of each label out of the entire set """
    try:
        series_averages = series.value_counts() / len(series)
        return series_averages.to_dict()
    except ZeroDivisionError as exception:
        raise exception

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

def merge_dicts(dict1, dict2):
    dictionary = dict1.copy()
    dictionary.update(dict2)
    return dictionary

def calculate_list_product(list_):
    """ Slightly faster to work on arrays than directly with list """
    return numpy.array(list_).prod()

class NaiveBayes():
    def __init__(self):
        self.dict_priors = None
        self.dict_frequencies_text = None
        self.dict_model = None

    def calculate_priors(self, series_labels):
        self.dict_priors = calculate_frequency_average(series_labels)

    def calculate_frequencies_text(self, iterable_text):
        """ { word : n(word) } """
        if isinstance(iterable_text, pandas.core.series.Series):            
            dict_frequencies_text = construct_frequency_dict_from_series(iterable_text)
        else:
            dict_frequencies_text = construct_frequency_dict_from_strings(iterable_text)
        if self.dict_frequencies_text is None:
            self.dict_frequencies_text = dict_frequencies_text
        return dict_frequencies_text

    def update_frequencies_text(self, iterable_text):
        """ Useful for predicting on unseen text """
        dict_frequencies_text = self.calculate_frequencies_text(iterable_text)
        self.dict_frequencies_text.update(dict_frequencies_text)

    def calculate_labeled_frequencies(self, dataframe_emails, column_label, column_words):
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
        total = sum(self.dict_frequencies_text.values())
    
        # Doing it this way avoids copy errors with nested dictionaries
        self.dict_model = {}
        for word in self.dict_frequencies_text.keys():
            self.dict_model.setdefault(word, {label : 0 for label in list_labels })
    
        # Split label column into groups so we can count them directly
        group_labels = dataframe_emails.groupby(column_label)
        for label, label_df in group_labels:
            for list_words in label_df[column_words]:
                for word in list_words:
                    self.dict_model[word][label] += 1

        # Handles the zero frequency offset
        for word, dict_frequency in self.dict_model.items():
            offset = max(self.dict_frequencies_text[word] / total, 1E-8)
            for key in dict_frequency.keys():
                dict_frequency[key] += offset
    
    def update_labeled_frequencies(self, iterable_text, dataframe_emails, column_label, column_words):
        """ A pretty costly operation since it retrains the model """
        self.update_frequencies_text(iterable_text)
        self.calculate_labeled_frequencies(dataframe_emails, column_label, column_words)


    def predict_bayes(self, word, label):
        """
        Doesn't use the naive assumption.
        likelihood = (num labeled emails with word) / sum(num labeled emails with word for all labels)
                   = P(A | Event_j) / sum ( P (A | Event_i) )
        """
        label_count = self.dict_model[word][label]
        all_label_counts = sum(self.dict_model[word].values())
    
        try:
            return label_count / all_label_counts
        except ZeroDivisionError:
            return 0

    def calculate_naive_bayes(self, list_words, series_labels):
        list_labels = series_labels.unique()
        counts_label = series_labels.value_counts()
        total = len(series_labels)
    
        # Cook up the total number of emails in each label
        dict_naive_bayes = { label : 1 for label in list_labels }
        for word in list_words:
            for label in list_labels:
                probability = self.dict_model[word][label] / counts_label[label]
                if probability == 0:
                    print(word)
                dict_naive_bayes[label] *= (probability * total)
    
        # Multiply by the total number of elements for each label
        for label in list_labels:
            dict_naive_bayes[label] *= counts_label[label]
    
        return dict_naive_bayes

    def predict_naive_bayes(self, email, label_to_predict, series_labels):
        """ Uses the naive assumption to predict on a given email """
    
        # words
        email = email.lower()
        words = set(email.split())
    
        dict_naive_bayes = self.calculate_naive_bayes(words, series_labels)
    
        numerator = dict_naive_bayes[label_to_predict]
        denominator = sum(dict_naive_bayes.values())
    
        return numerator/denominator


