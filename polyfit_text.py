# -*- coding: utf-8 -*-
from __future__ import print_function
"""
Created on Mon Oct 02 12:24:33 2017

@author: xingshuli
"""
import numpy as np
import math
import os 
from numpy import genfromtxt 

import matplotlib.pyplot as plt

# calculate 'Pearson' correlation coefficient
def computeCorrelation(X, Y):
    xBar = np.mean(X)
    yBar = np.mean(Y)
    SSR = 0.0
    varX = 0.0
    varY = 0.0
    for i in range(0, len(X)):
        diffXXBar = X[i] - xBar
        diffYYBar = Y[i] - yBar
        SSR += (diffXXBar * diffYYBar)
        varX += diffXXBar**2
        varY += diffYYBar**2
    SST = math.sqrt(varX * varY)
    return SSR / SST


#Polynomial fitting
'''
Here, x is single dimension, and the Polynomial like:
y = a1*x^n + a2*x^n-1 + ... + an*x + a

'''
def polyfit(x, y, degree):
    results = {}
    coeffs = np.polyfit(x, y, degree)
    results['polynomial'] = coeffs.tolist()
    
    p = np.poly1d(coeffs)
    yhat = p(x)
    
    ybar = np.sum(y)/len(y)
    ssreg = np.sum((yhat - ybar)**2)
    sstot = np.sum((y - ybar)**2)
    results['determination'] = ssreg / sstot
           
    return results, yhat


#polyfit(testX, testY, degree)[0] is results
#polyfit(testX, testY, degree)[1] is yhat
# determination is R^2
datapath =  os.path.join(os.getcwd(), '2014_beijing_PM2.5.txt')
data = genfromtxt(datapath, delimiter = None)

degree = 6
testX = sum(data[1:13, 3:4].tolist(), [])
testY = data[1:13, -1].tolist()
polyfitY = polyfit(testX, testY, degree)[1]   
print(polyfit(testX, testY, degree)[0]) 

#plot fitting Curve
plot1 = plt.plot(testX, testY, 'b*', label='original values')
plot2 = plt.plot(testX, polyfitY, 'r', label='polyfit values')

plt.xlabel('x', fontsize='x-large')
plt.ylabel('y', fontsize='x-large')

plt.legend(loc=0)
plt.title('polyfitting')
plt.show()

 
    
    
        
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    