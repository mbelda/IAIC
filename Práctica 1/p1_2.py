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

feet = 1650
nHab = 3

def normalizar(X):
    mu = []
    sigma = []
    #Para cada atributo (= columna)
    for i in range(np.shape(X)[1]):
        #Calculo su media
        mu.append(np.mean(X[:, i]))
        sigma.append(np.std(X[:, i]))
    
    X_norm = np.divide(np.subtract(X, mu), sigma)
    return X_norm, mu, sigma

    
def coste(X, Y, Theta, m):
    diferencias = np.subtract(np.dot(X, Theta), Y)
    return 1/(2*m)*(np.sum(np.power(diferencias, 2)))

def calculaGradiente(X, Y, Theta, m):
    return 1/m * np.dot(np.transpose(X), np.subtract(np.dot(X, Theta), Y))

def actualizaTheta(X, Y, Theta, m, alpha):
    return Theta - alpha*calculaGradiente(X, Y, Theta, m)

def evolucionCostesModificandoAlpha(X, Y, m, n):
    #Normalizo X
    X, mu, sigma = normalizar(X)
    
    #Añado una columna de 1s a X
    X = np.hstack([np.ones([m,1]), X])
    
    #Valroes de prueba de alpha
    alphas = [1.5, 1.0, 0.5, 0.3, 0.1, 0.03, 0.01, 0.001, 0.0001]
    
    #Valores fijos
    nIt = 800  
    
    for alpha in alphas:
        Theta = np.zeros(n+1)
        costes = []
        for i in range(nIt):
            costes.append(coste(X, Y, Theta, m))
            Theta = actualizaTheta(X, Y, Theta, m, alpha)
        
        if alpha == 0.01:
            #Nos guardamos el theta optimo de un buen alpha
            Theta_opt = Theta
        fig = plt.figure()
        plt.scatter(range(nIt), costes, c='blue')
        plt.xlabel("Número de iteraciones")
        plt.ylabel("Coste")
        plt.legend()
        plt.title("Evolución del coste con alpha=" + str(alpha))
        #plt.savefig("regresion_lineal_varias_variables_alpha_" + str(alpha) + ".png")
    
    
    print("-------------- DESCENSO DE GRADIENTE --------------")
    print("Theta optimo: " + str(Theta_opt))
    
    x = np.array([[feet, nHab]])
    #Normalizo x
    x = np.divide(np.subtract(x, mu), sigma)
    x = np.insert(x, 0, 1)
    
    pred = np.dot(x, Theta_opt)
    
    print("Vivienda de " + str(feet) + " pies cuadrados y " + str(nHab) + " habitaciones")
    print("Predicción del precio: "+ str(int(pred)) + "$")
    
def ecuacionNormal(X, Y, m, n):
    #Añado una columna de 1s a X
    X = np.hstack([np.ones([m,1]), X])
    
    #Calculamos el theta optimo
    Theta = np.dot(np.dot(np.linalg.pinv(np.dot(np.transpose(X), X)), np.transpose(X)), Y)

    print("-------------- ECUACION NORMAL --------------")
    print("Theta optimo: " + str(Theta))
    #Predecimos el precio de una casa con 1650 pies cuadrados y 3 habitaciones
    x = [1, feet, nHab]
    pred = np.dot(x, Theta)
    print("Vivienda de " + str(feet) + " pies cuadrados y " + str(nHab) + " habitaciones")
    print("Predicción del precio: "+ str(int(pred)) + "$")
    
def main():
    #Cargamos los valores de training
    datos = read_csv("ex1data2.csv", header=None).to_numpy().astype(float)
    
    X = datos[:, :-1]
    Y = datos[:,-1]
    
    m, n = np.shape(X)
    
    #Parte 2.1
    evolucionCostesModificandoAlpha(X, Y, m, n)
        
    #Parte 2.2
    ecuacionNormal(X, Y, m, n)


main()