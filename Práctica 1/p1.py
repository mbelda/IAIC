# -*- coding: utf-8 -*-
"""
Created on Mon Feb 22 19:44:22 2021

@author: Majo
"""
import numpy as np
from pandas.io.parsers import read_csv
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

#Cargamos los valores de training
valores = read_csv("ex1data1.csv", header=None).to_numpy()
m,n = valores.shape

def valoresY():
    return valores[:,n-1]

def valoresX():
    return valores[:,n-2]
    

def aplicaH(theta):
    #Una recta con theta = (theta_0, theta_1)
    return np.add(theta[0], np.dot(theta[1], valoresX()))

def funcion_coste(theta):
    #Calculo para cada punto su diferencia entre la imagen real y la imagen por la recta aproximada
    diferencias = np.abs(aplicaH(theta) - valoresY()) 
    return 1/(2*m)*(np.sum(np.power(diferencias, 2))) 


def actualiza_theta(theta, alpha):
    diferencias = aplicaFuncion(theta) - valoresY()
    auxtheta0 = theta[0] - alpha/m*np.sum(diferencias)
    auxtheta1 = theta[1] - alpha/m*np.sum(np.dot(diferencias, valoresX()))
    return [auxtheta0, auxtheta1]

def calculaThetaoptimoYGrafica_Parte1():
    #Decidimos los parametros fijos 
    alpha = 0.01 #Alpha: parámetro para avanzar en el descenso de gradiente
    theta = [0,0] #Theta inicial
    nIt = 1500 #Numero de iteraciones para encontrar el theta optimo
    
    print("Coste con theta=[0,0]: " + str(funcion_coste(theta)))
    #Itero hasta conseguir un coste mínimo
    for i in range(nIt):
        theta = actualiza_theta(theta, alpha)
    
    #Ahora tenemos un coste mínimo
    print("Coste mejor: " + str(funcion_coste(theta)))
    valor = 7
    print("Prediccion para 70.000 habitantes: " + str((theta[0] + theta[1]*valor)*10000) + "$")
    print("Mejores theta: " + str(theta))
    
    #Dibujamos los puntos y la recta que los aproxima
    plt.figure()
    plt.scatter(valores[:,0], valores[:,1], c='blue')
    plt.plot(valoresX(), aplicaFuncion(theta), c='red')
    plt.xlabel("Población de la ciudad en 10.000s")
    plt.ylabel("Ingresos en $10.000s")
    plt.legend()
    plt.savefig('regresion_lineal.png')
    


def make_data(t0_range, t1_range, X, Y):
    """
        Genera las matrices X,Y,Z para generar un plot en 3D
    """
    step = 0.1
    Theta0 = np.arange(t0_range[0], t0_range[1], step)
    Theta1 = np.arange(t1_range[0], t1_range[1], step)
    Theta0, Theta1 = np.meshgrid(Theta0, Theta1)
    # Theta0 y Theta1 tienen las misma dimensiones, de forma que
    # cogiendo un elemento de cada uno se generan las coordenadas x,y
    # de todos los puntos de la rejilla
    Coste = np.empty_like(Theta0)
    for ix, iy in np.ndindex(Theta0.shape):
        Coste[ix, iy] = funcion_coste([Theta0[ix, iy], Theta1[ix, iy]])
    return [Theta0, Theta1, Coste]
    
def visualizacionFuncionCoste_Parte1_1():
    t0_range = [-10, 10]
    t1_range = [-1, 4]
    
    #Genero las matrices para el plot 3D
    Theta0, Theta1, Coste = make_data(t0_range, t1_range, valoresX(), valoresY())
    
    #Dibujamos el contour
    plt.figure()
    plt.contour(Theta0, Theta1, Coste, np.logspace(-2, 3, 20), colors='blue')
    plt.xlabel("Theta_0")
    plt.ylabel("Theta_1")
    plt.legend()
    plt.savefig('ContourRegresionLineal.png')


    #Dibujamos el surface
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    surf = ax.plot_surface(Theta0, Theta1, Coste, cmap=cm.coolwarm)
    plt.savefig('Surface_regresion_lineal.png')
    
def main():
    #Cargamos los valores de training
    #valores = read_csv("ex1data1.csv", header=None).to_numpy()
    
    calculaThetaoptimoYGrafica_Parte1()
    
    visualizacionFuncionCoste_Parte1_1()
    
  
    
    

    
        
    
    


main()