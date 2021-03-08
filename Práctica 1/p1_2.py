# -*- coding: utf-8 -*-
"""
Created on Mon Mar  1 19:51:00 2021

@author: Majo
"""
import numpy as np
from pandas.io.parsers import read_csv
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm


def getExampleValues():
    feet = 1650
    nHab = 3
    return feet, nHab

def load_data(file):
    return read_csv(file, header=None).to_numpy().astype(float)

def normalize(X):
    mu = np.array([])
    sigma = np.array([])
    #For each attribute (= column)
    for i in range(np.shape(X)[1]):
        mu = np.append(mu, np.mean(X[:, i]))
        sigma = np.append(sigma, np.std(X[:, i]))
    
    X_norm = np.divide(np.subtract(X, mu), sigma)
    return X_norm, mu, sigma

    
def costFunction(X, Y, Theta, m):
    diferences = np.subtract(np.dot(X, Theta), Y)
    return 1/(2*m)*(np.sum(np.power(diferences, 2)))

def computeGradient(X, Y, Theta, m):
    return 1/m * np.dot(np.transpose(X), np.subtract(np.dot(X, Theta), Y))

def updateTheta(X, Y, Theta, m, alpha):
    return Theta - alpha*computeGradient(X, Y, Theta, m)

def costFunctionvolutionModifyingAlpha(X, Y, m, n):
    #Normalizo X
    X, mu, sigma = normalize(X)
    
    #Add 1s column to X
    X = np.hstack([np.ones([m,1]), X])
    
    #Alphas values
    alphas = [1.5, 1.0, 0.5, 0.3, 0.1, 0.03, 0.01, 0.001, 0.0001]
    
    #Init values
    loops = 800  
    
    for alpha in alphas:
        Theta = np.zeros(n+1)
        costs = []
        for i in range(loops):
            costs.append(costFunction(X, Y, Theta, m))
            Theta = updateTheta(X, Y, Theta, m, alpha)
        
        if alpha == 0.1:
            #It's a good alpha, save optimum theta for later prediction example
            Theta_opt = Theta
        plt.figure()
        plt.scatter(range(loops), costs, c='blue')
        plt.xlabel("Número de iteraciones")
        plt.ylabel("costFunction")
        plt.legend()
        plt.title("Evolución del costFunction con alpha=" + str(alpha))
        #plt.savefig("regresion_lineal_varias_variables_alpha_" + str(alpha) + ".png")
    
    
    print("-------------- GRADIENT DESCENDANT --------------")
    print("Optimum theta: " + str(Theta))
    
    feet, nHab = getExampleValues()
    x = np.array([[feet, nHab]])
    #Normalize x
    x = np.divide(np.subtract(x, mu), sigma)
    x = np.insert(x, 0, 1)
    
    pred = np.dot(x, Theta_opt)
    
    print("Home:" + str(feet) + " square foot and " + str(nHab) + " rooms")
    print("Price prediction: "+ str(int(pred)) + "$")
    
def normalEcuation(X, Y, m, n):
    #Add 1s column to X
    X = np.hstack([np.ones([m,1]), X])
    
    #Compute optimum theta
    Theta = np.dot(np.dot(np.linalg.pinv(np.dot(np.transpose(X), X)), np.transpose(X)), Y)

    print("-------------- NORMAL EQUATION --------------")
    print("Optimum theta: " + str(Theta))
    #Example
    feet, nHab = getExampleValues()
    x = [1, feet, nHab]
    pred = np.dot(x, Theta)
    print("Home:" + str(feet) + " square foot and " + str(nHab) + " rooms")
    print("Price prediction: "+ str(int(pred)) + "$")
    
def main():
    #Load data
    values = load_data("ex1data2.csv")
    
    X = values[:, :-1]
    Y = values[:,-1]
    
    m, n = np.shape(X)

    print(np.shape(X))
    print(np.shape(Y))
    
    #Parte 2.1
    costFunctionvolutionModifyingAlpha(X, Y, m, n)
        
    #Parte 2.2
    normalEcuation(X, Y, m, n)


if __name__ == "__main__":
    main()