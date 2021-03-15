# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
from pandas.io.parsers import read_csv
import matplotlib.pyplot as plt
import scipy.optimize as opt
from scipy.io import loadmat
from sklearn.preprocessing import PolynomialFeatures

def sigmoid(z):
    return np.divide(1, 1 + np.exp(-z))

def cost(theta, X, Y):
    H = sigmoid(np.dot(X, theta))
    
    return -1 / len(X) * (np.dot(Y, np.log(H)) + np.dot((1-Y), np.log(1-H)))

def regularized_cost(theta, lamb, X, Y):
    return cost(theta, X, Y) + lamb/(2*len(X))*np.sum(np.power(theta, 2))

def gradient(theta, X, Y):
    H = sigmoid(np.dot(X, theta))
    
    return 1 / len(Y) * np.dot(X.T, H - Y)

def regularized_gradient(theta, lamb, X, Y):
    gr = gradient(theta, X, Y)
    reg = lamb/len(X)*theta
    reg[0] = 0 #Theta_0 doesn't have extra cost
    return np.add(gr, reg)

def regularizedLogisticregression(X, Y, reg):
    theta = np.zeros(X.shape[1])
    
    #Optimize theta with fmin_tnc from scipy
    result = opt.fmin_tnc(func=regularized_cost, x0=theta, \
                          fprime=regularized_gradient, args=(reg, X,Y), \
                          disp=False)
    return result[0]
    

def oneVsAll(X, y, n_labels, reg):
    """
        oneVsAll entrena varios clasificadores por regresión logística con término
        de regularización 'reg' y devuelve el resultado en una matriz, donde
        la fila i-ésima corresponde al clasificador de la etiqueta i-ésima
    """

    #Add 1s column to X
    X_vec = np.hstack([np.ones([X.shape[0],1]), X])
    
    #Create thetas matrix
    thetas = np.zeros(shape=(n_labels, X_vec.shape[1]))
    
    #Compute theta for each label
    for i in range(n_labels):
        #Prepare data for label i
        label = i
        if i == 0:
            label = 10
        label_i = np.where(y == label)
        y_i = np.zeros(y.shape[0])
        np.put(y_i, label_i, 1)
        
        #Compute optimized theta
        theta_i = regularizedLogisticregression(X_vec, y_i, reg)
        thetas[i] = theta_i

    print(thetas)
    return thetas

def main():

    data = loadmat('ex3data1.mat')

    y = data['y']
    X = data['X']

    #Plot 10 random examples
    sample = np.random.choice(X.shape[0], 10)
    plt.imshow(X[sample,:].reshape(-1, 20).T)
    plt.axis('off')
    
    #Get thetas
    n_labels = 10
    reg = 0.1
    oneVsAll(X, y, n_labels, reg)

        
 
if __name__ == "__main__":
    main()