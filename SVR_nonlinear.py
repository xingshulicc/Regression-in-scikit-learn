# -*- coding: utf-8 -*-
from __future__ import print_function
"""
Created on Tue Oct 03 14:21:20 2017

@author: xingshuli
"""
import os

#import numpy as np
from numpy import genfromtxt

from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler

import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore")

datapath =  os.path.join(os.getcwd(), 'dataset.txt')
data = genfromtxt(datapath, delimiter = ',')

X = data[:, :-1]
Y = data[:, -1] 

scaler = MinMaxScaler(feature_range = (0, 1), copy = True)
x = scaler.fit_transform(X)
y = scaler.fit_transform(Y)


'''
When training an SVM with the Radial Basis Function (RBF) kernel, 
two parameters must be considered: C and gamma. 
A low C makes the decision surface smooth, 
while a high C aims at classifying all training examples correctly. 
gamma defines how much influence a single training example has.
The larger gamma is, the closer other examples must be to be affected.

kernel function: rbf~Radial Basis Function
exp(-r*|x - x'|^2), r is specified by keyword gamma, which must be greater than 0

For decision function:
sum[(ai - ai') * K(xi, x)] + p
dual_coef_ holds the difference ai - ai'
support_vectors_  holds the support vectors, 
intercept_ holds the independent term p
'''

regr = SVR(C = 2.0, epsilon = 0.1, kernel = 'rbf', gamma = 0.5, 
           tol = 0.001, verbose=False, shrinking=True, max_iter = 10000)

regr.fit(x, y)
data_pred = regr.predict(x)
y_pred = scaler.inverse_transform(data_pred)

print('Support_vectors: \n', regr.support_vectors_)
print('Coefficients of the support vector in the decision function: \n', \
      regr.dual_coef_)
print('Constants in decision function: \n', regr.intercept_)

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
plt.title('SVR_nonlinear')
plt.show()
























































