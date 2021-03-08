# -*- coding: utf-8 -*-
"""
Created on Mon Mar  8 19:36:50 2021

@author: Majo
"""
import numpy as np
from pandas.io.parsers import read_csv
import matplotlib.pyplot as plt
import scipy.optimize as opt


def load_data(file):
    return read_csv(file, header=None).to_numpy().astype(float)


def sigmoid(z):
    return np.divide(1, 1 + np.exp(-z))

def cost(theta, X, Y):
    H = sigmoid(np.dot(X, theta))
    
    return -1 / len(X) * (np.dot(Y, np.log(H)) + np.dot((1-Y), np.log(1-H)))

def gradient(theta, X, Y):
    H = sigmoid(np.dot(X, theta))
    
    return 1 / len(Y) * np.dot(X.T, H - Y)

def pinta_frontera_recta(X, Y, theta):
    x1_min, x1_max = X[:, 0].min(), X[:, 0].max()
    x2_min, x2_max = X[:, 1].min(), X[:, 1].max()
       
    xx1, xx2 = np.meshgrid(np.linspace(x1_min, x1_max),
    np.linspace(x2_min, x2_max))
       
    h = sigmoid(np.c_[np.ones((xx1.ravel().shape[0], 1)),
    xx1.ravel(),
    xx2.ravel()].dot(theta))
    h = h.reshape(xx1.shape)
       
    # el cuarto parámetro es el valor de z cuya frontera se
    # quiere pintar
    plt.contour(xx1, xx2, h, [0.5], linewidths=1, colors='r')

def parte1(X, Y):
    X_vec = np.hstack([np.ones([X.shape[0],1]), X])
    theta = np.zeros(X_vec.shape[1])
    
    
    print("Cost theta 0: " + str(cost(theta, X_vec, Y)))
    print("Gradient theta 0: " + str(gradient(theta, X_vec, Y)))
    
    #Optimizamos theta con fmin_tnc de scipy
    result = opt.fmin_tnc(func=cost, x0=theta, fprime=gradient, args=(X_vec,Y), disp=False)
    theta = result [0]
    
    print("Cost theta opt: " + str(cost(theta, X_vec, Y)))
    print("Gradient theta opt: " + str(gradient(theta, X_vec, Y)))

    
    pos = np.where(Y == 1)
    neg = np.where(Y == 0)
    plt.figure()
    plt.scatter(X[pos,0], X[pos,1], marker='+', c='g', label='admitidos')
    plt.scatter(X[neg,0], X[neg,1], marker='o', c='k', label='no admitidos')
    pinta_frontera_recta(X, Y, theta)
    plt.xlabel("Nota examen 1")
    plt.ylabel("Nota examen 2")
    plt.legend(loc='upper right')
    
    xPorTheta = np.dot(X_vec, theta)
    
    X_pos = xPorTheta[pos]
    X_neg = xPorTheta[neg]
    #Calculamos la proporcion de aciertos del modelo
    aciertos_admitidos = np.sum(sigmoid(X_pos) >= 0.5)
    aciertos_no_admitidos = np.sum(sigmoid(X_neg) < 0.5)
    
    print("Proporcion de aciertos: " + str((aciertos_admitidos + aciertos_no_admitidos)/X.shape[0]))
    

def main():
    values = load_data("ex2data1.csv")
    
    X = values[:, :-1]
    Y = values[:,-1]
    
    parte1(X, Y)
    



if __name__ == "__main__":
    main()