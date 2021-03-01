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



def normalizar(X, n):
    mu = []
    sigma = []
    #Para cada atributo (= columna)
    for i in range(n):
        #Calculo su media
        mu.append(np.mean(X[:, i]))
        sigma.append(np.std(X[:, i]))
    
    X_norm = np.divide(np.subtract(X, mu), sigma)
    return X_norm, mu, sigma

    
def coste(X, Y, Theta, m):
    diferencias = np.dot(X, Theta) - Y
    return 1/(2*m)*(np.sum(np.power(diferencias, 2)))

def calculaGradiente(X, Y, Theta, m):
    return 1/m * np.dot(np.transpose(X), np.subtract(coste(X, Y, Theta, m), Y))

def actualizaTheta(X, Y, Theta, m, alpha):
    return Theta - alpha*calculaGradiente(X, Y, Theta, m)


def main():
    #Cargamos los valores de training
    datos = read_csv("ex1data2.csv", header=None).to_numpy().astype(float)
    
    X = datos[:, :-1]
    Y = datos[:,-1]
    
    m, n = np.shape(X)
    
    #Normalizo X
    X, mu, sigma = normalizar(X, n)
    
    #Añado una columna de 1s a X
    X = np.hstack([np.ones([m,1]), X])
    
    #Valroes de prueba de alpha
    alphas = [0.3, 0.1, 0.03, 0.01]
    
    #Valores fijos
    nIt = 1500
    Theta = np.zeros(n+1)
    
    # for alpha in alphas:
    #     costes = []
    #     for i in range(nIt):
    #         costes.append(coste(X, Y, Theta, m))
    #         Theta = actualizaTheta(X, Y, Theta, m, alpha)
           
    #     fig = plt.figure()
    #     plt.scatter(range(nIt), costes, c='blue')
    #     plt.xlabel("Número de iteraciones")
    #     plt.ylabel("Coste")
    #     plt.legend()
    #     plt.title("Evolución del coste con alpha=" + str(alpha))
    #     #plt.savefig('regresion_lineal_varias_variables.png')
    
    alpha = alphas[0]
    costes = []
    for i in range(nIt):
        costes.append(coste(X, Y, Theta, m))
        Theta = actualizaTheta(X, Y, Theta, m, alpha)
    
    print(costes)
    fig = plt.figure()
    plt.scatter(range(nIt), costes, c='blue')
    plt.xlabel("Número de iteraciones")
    plt.ylabel("Coste")
    plt.legend()
    plt.title("Evolución del coste con alpha=" + str(alpha))
    
    
    
  
    
    

    
        
    
    


main()