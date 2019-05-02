# -*- coding: utf-8 -*-
""" Evaluates the estimated target values of either a single classifier or ensemble of classifiers
    model using different regression metrics.

# Reference
- [3.3.4. Regression metrics](http://scikit-learn.org/stable/modules/model_evaluation.html#regression-metrics)
"""

from __future__ import print_function
import numpy as np
from sklearn.metrics import explained_variance_score, mean_absolute_error, mean_squared_error, r2_score

# for printing purposes only
classifier_name = 'VGG19_ResNet50_VGG16 - euclidean loss'

# remains constant
y_true = np.load('/home/sandbox/Desktop/EMOTIC_resources/VAD-regression/numpy_matrices/Y_train/y_test.npy')

y_predicted = np.load('/home/sandbox/Desktop/VGG19_ResNet50_VGG16_y_predicted.npy')


# reference http://joshlawman.com/metrics-regression/
MSE = mean_squared_error(y_true = y_true, y_pred=y_predicted)
RMSE = np.sqrt(MSE)
r2 = r2_score(y_true = y_true, y_pred=y_predicted)
explained_var = explained_variance_score(y_true = y_true, y_pred=y_predicted)
MAE = mean_absolute_error(y_true = y_true, y_pred=y_predicted)



print ('------------------------ ', classifier_name, '------------------------ ')
print ('     Mean absolute error (MAE): ', MAE)
print ('      Mean squared error (MSE): ', MSE) # closer to zero are better.
print ('Root mean squared error (RMSE): ', RMSE)
print ('Explained variance score (EVS): ', explained_var) # best possible score is 1.0
print ('          R^2 Score (R2 Score): ', r2) # best possible score is 1.0 and it can be negative

