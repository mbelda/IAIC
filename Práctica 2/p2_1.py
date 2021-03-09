# -*- coding: utf-8 -*-
"""
Created on Mon Mar  8 19:36:50 2021

@author: Majo
"""
import numpy as np
from pandas.io.parsers import read_csv
import matplotlib.pyplot as plt
import scipy.optimize as opt
from sklearn.preprocessing import PolynomialFeatures


def load_data(file):
    return read_csv(file, header=None).to_numpy().astype(float)

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

def plot_border_line(X, Y, theta):
    x1_min, x1_max = X[:, 0].min(), X[:, 0].max()
    x2_min, x2_max = X[:, 1].min(), X[:, 1].max()
       
    xx1, xx2 = np.meshgrid(np.linspace(x1_min, x1_max),
    np.linspace(x2_min, x2_max))
       
    h = sigmoid(np.c_[np.ones((xx1.ravel().shape[0], 1)),
    xx1.ravel(),
    xx2.ravel()].dot(theta))
    h = h.reshape(xx1.shape)
    
    plt.contour(xx1, xx2, h, [0.5], linewidths=1, colors='r')

def plot_decisionboundary(X, Y, theta, poly):
    x1_min, x1_max = X[:, 0].min(), X[:, 0].max()
    x2_min, x2_max = X[:, 1].min(), X[:, 1].max()
    xx1, xx2 = np.meshgrid(np.linspace(x1_min, x1_max),
    np.linspace(x2_min, x2_max))
    h = sigmoid(poly.fit_transform(np.c_[xx1.ravel(),
    xx2.ravel()]).dot(theta))
    h = h.reshape(xx1.shape)
    plt.contour(xx1, xx2, h, [0.5], linewidths=1, colors='g')

def logisticRegression(X, Y):
    #Add 1s column to X
    X_vec = np.hstack([np.ones([X.shape[0],1]), X])
    #Initial theta
    theta = np.zeros(X_vec.shape[1])
    
    #Initial cost and gradient
    print("Initial cost: " + str(cost(theta, X_vec, Y)))
    print("Initial gradient: " + str(gradient(theta, X_vec, Y)))
    
    #Optimize theta with fmin_tnc from scipy
    result = opt.fmin_tnc(func=cost, x0=theta, fprime=gradient, args=(X_vec,Y), disp=False)
    theta = result[0]
    
    #Cost and gradient with optimized theta
    print("Optimized cost: " + str(cost(theta, X_vec, Y)))
    print("Optimized gradient: " + str(gradient(theta, X_vec, Y)))

    #Plot points and border
    pos = np.where(Y == 1)
    neg = np.where(Y == 0)
    plt.figure()
    plt.scatter(X[pos,0], X[pos,1], marker='+', c='g', label='admitted')
    plt.scatter(X[neg,0], X[neg,1], marker='o', c='k', label='not admitted')
    plot_border_line(X, Y, theta)
    plt.xlabel("Exam 1 score")
    plt.ylabel("Exam 2 score")
    plt.legend(loc='upper right')
    plt.savefig("logistic_regression_line.png")
    
    #Compute hits%
    xDotTheta = np.dot(X_vec, theta)
    X_pos = xDotTheta[pos]
    X_neg = xDotTheta[neg]
    
    hits_pos = np.sum(sigmoid(X_pos) >= 0.5)
    hits_neg = np.sum(sigmoid(X_neg) < 0.5)
    percent = (hits_pos + hits_neg) / X.shape[0] * 100
    
    print("Hit rate: " + str(percent) + "%")
    
def regularizedLogisticregression(X, Y):
    #Transform X to 6 degree polynomial
    poly = PolynomialFeatures(6)
    X_pol = poly.fit_transform(X)
    
    lambdas = [0.0, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.]
   
    for lamb in lambdas:
        #Initial theta
        theta = np.zeros(X_pol.shape[1])
        
        #Initial cost
        print("----------- LAMBDA = " + str(lamb) + " -----------")
        print("Initial cost: " + str(regularized_cost(theta, lamb, X_pol, Y)))
        
        #Optimize theta with fmin_tnc from scipy
        result = opt.fmin_tnc(func=regularized_cost, x0=theta, fprime=regularized_gradient, args=(lamb, X_pol,Y), disp=False)
        theta = result[0]
        
        #Cost with optimized theta
        print("Optimized cost: " + str(regularized_cost(theta, lamb, X_pol, Y)))
        
        #Plot points and border
        pos = np.where(Y == 1)
        neg = np.where(Y == 0)
        plt.figure()
        plt.scatter(X[pos,0], X[pos,1], marker='+', c='g', label='accepted')
        plt.scatter(X[neg,0], X[neg,1], marker='o', c='k', label='not accepted')
        plot_decisionboundary(X, Y, theta, poly)
        plt.xlabel("Microchip test 1")
        plt.ylabel("Microchip test 2")
        plt.legend(loc='upper right')
        plt.title("Border with lambda=%.3f" % lamb)
        plt.savefig("logistic_regression_circle_with_lambda_%.3f.png" % lamb)

def main():
    values = load_data("ex2data1.csv")
    
    X = values[:, :-1]
    Y = values[:,-1]
    print("----------------- LOGISTIC REGRESSION -----------------")
    logisticRegression(X, Y)
    
    values = load_data("ex2data2.csv")
    
    X = values[:, :-1]
    Y = values[:,-1]
    print("----------------- REGULARIZED LOGISTIC REGRESSION -----------------")
    regularizedLogisticregression(X, Y)



if __name__ == "__main__":
    main()