# -*- coding: utf-8 -*-
"""
Created on Tue Apr 20 09:33:17 2021

@author: Majo
"""

import numpy as np
import scipy.optimize as opt
from scipy.io import loadmat
import matplotlib.pyplot as plt

def applyH(theta, X):
    #line y = theta0 + theta1*x
    return theta[0] + np.dot(X, theta[1:])

def reg_cost(Theta, X, y, reg):
    diferences = np.subtract(np.dot(X, Theta), y)
    cost = 1/(2*X.shape[0])*(np.sum(np.power(diferences, 2)))
    reg_cost = cost + reg/(2*X.shape[0]) * np.sum(np.power(Theta[1:], 2))
    return reg_cost

def reg_gradient(Theta, X, Y, reg):
    grad = 1/X.shape[0] * np.dot(np.transpose(X), np.subtract(np.dot(X, Theta), Y))
    reg_grad = grad + np.hstack((0, (reg/X.shape[0] * Theta[1:])))
    return reg_grad

def error(Theta, X, y):
    error = 1/(2*X.shape[0]) * np.sum(np.power(np.subtract(np.dot(X, Theta), y), 2))
    return error

def Xpolynomial(X, p):
    Xpol = X
    for i in range(2, p+1):
        Xpol = np.column_stack((Xpol,np.power(X, i)))
    return Xpol

def normalize(X):
    mu = np.array([])
    sigma = np.array([])
    #For each attribute (= column)
    for i in range(np.shape(X)[1]):
        mu = np.append(mu, np.mean(X[:, i]))
        sigma = np.append(sigma, np.std(X[:, i]))
    
    X_norm = np.divide(np.subtract(X, mu), sigma)
    return X_norm, mu, sigma

def regularized_linear_regression(X, y):
    #Vectorize X
    X_vec = np.hstack([np.ones([X.shape[0],1]), X])
    
    #Regularization param
    reg = 1
    
    #Initialize theta
    Theta = np.ones(X_vec.shape[1])
    
    #Compute initial cost
    c = reg_cost(Theta, X_vec, y, reg)
    g = reg_gradient(Theta, X_vec, y, reg)
    
    print("Initial cost: %.4f" % c)
    print("Initial gradient: [%.4f, %.4f]" % (g[0], g[1]))
    
    #Regularization param
    reg = 0
    #Optimize theta
    res = opt.minimize(fun=reg_cost, x0=Theta, args=(X_vec, y, reg), jac=reg_gradient)
    Theta_opt = res['x']
    predictions = np.dot(X_vec, Theta_opt)
    #Plot data points and aproximation line
    plt.figure()
    plt.scatter(X, y, color='orange', marker='x')
    plt.plot(X, predictions, c='blue', label='Regression line')
    plt.xlabel("Change in water level (x)")
    plt.ylabel("Water flowing out of the dam (y)")
    plt.legend()
    plt.savefig('regresion_lineal.png')
    
def learning_curves(X, y, Xval, yval):
    #Vectorize X
    X_vec = np.hstack([np.ones([X.shape[0],1]), X])
    Xval_vec = np.hstack([np.ones([Xval.shape[0],1]), Xval])
    
    #Regularization param
    reg = 0
    
    #Initialize errors
    train_errors = np.zeros(X_vec.shape[0])
    validation_errors = np.zeros(X_vec.shape[0])
    
    for i in range(1,X_vec.shape[0]):
        #Generate the subset
        X_i = X_vec[0:i,:]
        y_i = y[0:i]
        
        #Compute optimum Theta
        Theta = np.ones(X_i.shape[1]) 
        res = opt.minimize(fun=reg_cost, x0=Theta, args=(X_i, y_i, reg), jac=reg_gradient)
        Theta_opt = res['x']
        
        #Compute training and validation error
        train_errors[i] = error(Theta_opt, X_i, y_i)
        validation_errors[i] = error(Theta_opt, Xval_vec, yval)
    
    #Plot errors
    plt.figure()
    plt.plot(range(1,X.shape[0]), train_errors[1:], color='blue', label='Train')
    plt.plot(range(1,X.shape[0]), validation_errors[1:], color='orange', label='Cross Validation')
    plt.xlabel("Number of training examples")
    plt.ylabel("Error")
    plt.title("Learning curve for linear regression")
    plt.legend(loc='upper right')
    plt.savefig('errors.png')


def polynomial_regression(X, y):
    #Transform X
    p = 8
    X_pol = Xpolynomial(X, p)
    X_norm, sigma, mu = normalize(X_pol)
    X_vec_pol = np.hstack([np.ones([X_norm.shape[0],1]), X_norm])
    
    #Regularization param
    reg = 0
    
    #Initialize theta
    Theta = np.ones(X_vec_pol.shape[1])
    
    #Optimize theta
    res = opt.minimize(fun=reg_cost, x0=Theta, args=(X_vec_pol, y, reg), jac=reg_gradient)
    Theta_opt = res['x']
    
    #Create new points
    min_bound = np.amin(X)
    max_bound = np.amax(X)
    X_new = np.arange(min_bound, max_bound, 0.05)
    X_new.resize(X_new.shape[0],1)
    #Polynomial
    X_new_pol = Xpolynomial(X_new, p)
    #Normalize
    X_new_norm = np.divide(np.subtract(X_new_pol, mu), sigma)
    #Vectorize
    X_new_vec = np.hstack([np.ones([X_new_norm.shape[0],1]), X_new_norm])
    
    #Compute predictions
    predictions = np.dot(X_new_vec, Theta_opt)
    
    #Plot data points and aproximated points
    plt.figure()
    plt.scatter(X, y, color='orange', marker='x')
    plt.plot(X_new, predictions, c='blue')
    plt.xlabel("Change in water level (x)")
    plt.ylabel("Water flowing out of the dam (y)")
    plt.legend()
    plt.title("Polynomial regression ($\lambda = 0$)")
    #plt.savefig('regresion_polinomial.png')

def main():
    #Load data
    data = loadmat('ex5data1.mat')
    
    X = data['X']
    y = data['y'].ravel()
    
    Xval = data['Xval']
    yval = data['yval'].ravel()
    
    Xtest = data['Xtest']
    ytest = data['ytest'].ravel()
    
    
    # 5.1: REGRESIÓN LINEAL REGULARIZADA
    regularized_linear_regression(X, y)
    
    # 5.2: CURVAS DE APRENDIZAJE
    learning_curves(X, y, Xval, yval)
    
    # 5.3: POLINOMIAL REGRESSION
    polynomial_regression(X, y)
    
    
    
 
if __name__ == "__main__":
    main()
