# -*- coding: utf-8 -*-
from __future__ import print_function
"""
Created on Mon Oct 02 14:24:30 2017

@author: xingshuli
"""

'''
multiple linear regression with ordinary least squares fit: 
Here x is Multiple input ~(x1, x2,..., xn)
y = a0 + a1*x1 + a2*x2 + ... + an*xn

'''
import os 

from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score

import matplotlib.pyplot as plt

#import numpy as np
from numpy import genfromtxt #convert data to array format

datapath =  os.path.join(os.getcwd(), 'dataset.txt')
data = genfromtxt(datapath, delimiter = ',')

X = data[:, :-1]
Y = data[:, -1]

regr = linear_model.LinearRegression()
regr.fit(X, Y)
data_pred = regr.predict(X)

print('coefficients: \n', regr.coef_)

#if data is expected to be already centered then intercept_ is not needed
print('intercept: \n', regr.intercept_)

#Calculate mean squared error
print('Mean Squared Error: %.4f' 
      % mean_squared_error(Y, data_pred))

#Calculate R^2 (regression score function)
print('Variance score: %.2f' % r2_score(Y, data_pred))

#plot Y and data_pred, note we have 10 data in total
x_axis = range(1, 11)
plot1 = plt.plot(x_axis, Y, 'b*', label='original values')
plot2 = plt.plot(x_axis, data_pred, 'r', label='regressor values')
plt.xlabel('x', fontsize='x-large')
plt.ylabel('y', fontsize='x-large')

plt.legend(loc=0)
plt.title('LinearRegression')
plt.show()

#x_pre = np.array([[102, 6]])
#y_pre = regr.predict(x_pre)  
#print('Y-Predict: ', y_pre)




























































