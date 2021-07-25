# -*- coding: utf-8 -*-
"""

"""
import _linear_regression

import numpy
import matplotlib.pyplot as plt


# ### Defining and plotting our dataset

features = numpy.array([1,2,3,5,6,7])
labels = numpy.array([155, 197, 244, 356,407,448])

dict_data = {
    'features': features,
    'labels'  : labels,
}

print(dict_data)

_linear_regression.plot_scatter(features, labels)
plt.show()

# ### Linear regression using our tricks
# In[2]

# We set the random seed in order to always get the same results.
numpy.random.seed(0)

# This line is for the x-axis to appear in the figure
plt.ylim(0,500)
slope, bias = _linear_regression.linear_regression(
    features,
    labels,
    trick_function = _linear_regression.absolute_trick,
    learning_rate = 0.05,
    error_metric = _linear_regression.rmse,
    epochs = 1000)
print('Price per room:', slope)
print('Base price:', bias)

# This line is for the x-axis to appear in the figure
plt.ylim(0,500)
slope, bias = _linear_regression.linear_regression(
    features,
    labels,
    trick_function = _linear_regression.square_trick,
    learning_rate = 0.01,
    error_metric = _linear_regression.rmse,
    epochs = 10000)
print('Price per room:', slope)
print('Base price:', bias)

# ### Linear regression in Turi Create

# In[3]:

# import turicreate as tc

# data = tc.SFrame(dict_data)

# model = tc.linear_regression.create(data, target='labels')

# model.coefficients

# new_point = tc.SFrame({'features': [4]})

# model.predict(new_point)

### Linear regression in statsmodels
# In[4]

import statsmodels.api as sm

exog = sm.add_constant(dict_data['features']) # adds an intercept column
instance_linear_regression_model = sm.OLS(
    endog = dict_data['labels'], 
    exog = exog)
linear_regression_results_object = instance_linear_regression_model.fit()
print(linear_regression_results_object.summary())

_linear_regression.plot_scatter(
    linear_regression_results_object.fittedvalues, 
    linear_regression_results_object.resid,
    x_label = "Fitted Values",
    y_label = "Residual Values")
sm.qqplot(linear_regression_results_object.resid_pearson, line = "q")