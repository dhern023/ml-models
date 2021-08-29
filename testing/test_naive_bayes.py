# -*- coding: utf-8 -*-
"""
Preprocess data by adding a column with the (non-repeated)
lowercase words in the email.

TODO: Log the dataframes to a text file to form some sort of summary.
"""

from _naive_bayes import NaiveBayes

import argparse
import pandas
import pathlib

parser = argparse.ArgumentParser(description='Train & analyze a linear regression model.')
parser.add_argument('--data', help='fpath of the dataset')
parser.add_argument('--label', help='The name of the label column')
parser.add_argument('--covariates', nargs='+', help='Optional. Model covariates separate by space')
args = parser.parse_args()

# Helpers (Preprocess) ========================================================

def split_string_into_unique_words(string):
    return list(set(string.split()))

def process_series_email(series_text):
    """ Converts text to lower-case then returns list of unique words """
    series_words = (
        series_text
        .copy() # copies original series
        .str.lower()
        .apply(split_string_into_unique_words)
        )

    return series_words

if __name__ == "__main__":
    # Environment variables

    column_text = "text"
    column_words = "words"
    column_label = args.label
    label_spam = 1
    label_ham = 0

    # Read dataset
    emails = pandas.read_csv(pathlib.Path(args.data))
    emails[column_words] = process_series_email(emails[column_text])

    # Print totals
    print("Number of documents:", len(emails)) # Log this

    # print("Number of spam emails:", num_spam)

    counts_label = emails[column_label].value_counts()
    print(counts_label) # log this)

    list_emails = [
        "lottery sale",
        "Hi mom how are you",
        "Hi MOM how aRe yoU afdjsaklfsdhgjasdhfjklsd",
        "meet me at the lobby of the hotel at nine am",
        "enter the lottery to win three million dollars",
        "buy cheap lottery easy money now",
        "buy cheap lottery easy money",
        "Grokking Machine Learning by Luis Serrano",
        "asdfgh"]

    instance_naive_bayes = NaiveBayes()
    instance_naive_bayes.calculate_priors(emails[column_label])
    instance_naive_bayes.calculate_frequencies_text(emails[column_text])
    instance_naive_bayes.update_frequencies_text(list_emails)
    # Calculate text word frequencies and train the model
    instance_naive_bayes.calculate_labeled_frequencies(
        emails, column_label, column_words)

    # Calculating prior probabilities
    print("Probability of spam:", instance_naive_bayes.dict_priors[label_spam])

    # Some examples (1 is spam, and 0 is ham)
    print(instance_naive_bayes.dict_model['lottery'])
    print(instance_naive_bayes.dict_model['sale'])
    print(instance_naive_bayes.dict_model['already'])

    print(instance_naive_bayes.predict_bayes('lottery', label_spam))
    print(instance_naive_bayes.predict_bayes('sale', label_spam))
    print(instance_naive_bayes.predict_bayes('already', label_spam))
    
    cout = "Probability email is spam: "
    for email in list_emails:
        print(cout, instance_naive_bayes.predict_naive_bayes(email, label_spam, emails[column_label]))
    
    # Do our results make sense?
    # If a classification is surprising, 
    # let's check how often a word like "serrano" appears in spam emails.
    
    print(instance_naive_bayes.dict_model['serrano'])
    print(instance_naive_bayes.predict_bayes('serrano', label_spam))