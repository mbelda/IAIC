# -*- coding: utf-8 -*-
"""
Created on Mon Feb 22 19:44:22 2021

@author: Majo
"""
import numpy as np
from pandas.io.parsers import read_csv
import matplotlib.pyplot as plt

#Cargamos los valores de training
valores = read_csv("ex1data1.csv", header=None).to_numpy()
m,n = valores.shape

#Decidimos los parametros alpha, theta inicial y el número de iteraciones
alpha = 0.01
theta = [0,0]
nIt = 1500


def valoresY():
    return valores[:,n-1]

def valoresX():
    return valores[:,n-2]

def aplicaFuncion(theta):
    #Una recta con theta = (theta_0, theta_1)
    return np.add(theta[0], np.dot(theta[1], valoresX()))

def funcion_coste(theta):
    #Calculo para cada punto su diferencia entre la imagen real y la imagen por la recta aproximada
    diferencias = np.abs(aplicaFuncion(theta) - valoresY()) 
    return 1/(2*m)*(np.sum(np.power(diferencias, 2))) 


def actualiza_theta(theta):
    diferencias = aplicaFuncion(theta) - valoresY()
    auxtheta0 = theta[0] - alpha/m*np.sum(diferencias)
    auxtheta1 = theta[1] - alpha/m*np.sum(np.dot(diferencias, valoresX()))
    return [auxtheta0, auxtheta1]



print("Coste con theta=[0,0]: " + str(funcion_coste(theta)))
    
#Itero hasta conseguir un coste mínimo
for i in range(nIt):
    theta = actualiza_theta(theta)

#Ahora tenemos un coste mínimo
print("Coste mejor: " + str(funcion_coste(theta)))
valor = 70000
print("Prediccion para 70000: " + str(theta[0] + theta[1]*valor))
print("Mejores theta: " + str(theta))

#Dibujamos los piuntos y la recta que los aproxima
plt.figure()
plt.scatter(valores[:,0], valores[:,1], c='blue')
plt.plot(valoresX(), aplicaFuncion(theta), c='red')
plt.xlabel("x")
plt.ylabel("y")
plt.legend()