# -*- coding: utf-8 -*-
from __future__ import print_function
"""
Created on Mon Oct 02 16:14:31 2017

@author: xingshuli
"""
'''
multiple linear regression with SGD Algorithm
y = a0 + a1*x1 + a2*x2 + ... + an*xn

'''

import os

#import numpy as np

import warnings

import matplotlib.pyplot as plt

from sklearn.linear_model.stochastic_gradient import SGDRegressor
from numpy import genfromtxt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score

warnings.filterwarnings("ignore")

datapath =  os.path.join(os.getcwd(), 'dataset.txt')
data = genfromtxt(datapath, delimiter = ',')

X = data[:, :-1]
Y = data[:, -1] 

scaler = MinMaxScaler(feature_range = (0, 1), copy = True)
x = scaler.fit_transform(X)
y = scaler.fit_transform(Y)

'''
regularized training error given by:
E(w, b) = 1/n * sum(L(yi, f(xi))) + alpha * R(w)
Note: L is loss function, R(w) is regularization term (penalty)

For Elastic Net R(w):
R(w) = p/2 * sum(wi^2) + (1 - p) * |wi| where p is given by 1 - l1_ratio

For inverse scaling learning_rate:
lr = eta0 / t^power_t

'''
regr = SGDRegressor(penalty = 'elasticnet', alpha = 0.0001, l1_ratio = 0.25, 
                    learning_rate = 'invscaling', eta0 = 0.01, power_t = 0.25, 
                    loss = 'epsilon_insensitive', epsilon = 0.1, shuffle = True, 
                    fit_intercept = True, n_iter = 1000000, average = False, verbose = 0)

regr.fit(x, y)
data_pred = regr.predict(x)
y_pred = scaler.inverse_transform(data_pred)

print('coefficients: \n', regr.coef_)

#if data is expected to be already centered then intercept_ is not needed
print('intercept: \n', regr.intercept_)

#Calculate mean squared error
print('Mean Squared Error: %.4f' 
      % mean_squared_error(y, data_pred))

#Calculate R^2 (regression score function)
print('Variance score: %.2f' % r2_score(y, data_pred))

x_axis = range(1, 11)
plot1 = plt.plot(x_axis, Y, 'b*', label='original values')
plot2 = plt.plot(x_axis, y_pred, 'r', label='regressor values')
plt.xlabel('x', fontsize='x-large')
plt.ylabel('y', fontsize='x-large')

plt.legend(loc=0)
plt.title('SGD_Regressor')
plt.show()

























































