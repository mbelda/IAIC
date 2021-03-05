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


def load_data(file):
    data = read_csv(file, header=None)
    return np.asarray(data)

def applyH(theta, X):
    #line y = theta0 + theta1*x
    return np.add(theta[0], np.dot(theta[1], X))

def compute_cost_function(theta, X, Y, m):
    #Diference between real and aproximate images
    diferences = np.abs(np.subtract(applyH(theta, X), Y)) 
    return 1/(2*m)*(np.sum(np.power(diferences, 2)))

def update_theta(theta, alpha, X, Y, m):
    diferences = np.subtract(applyH(theta, X), Y)
    auxtheta0 = theta[0] - alpha/m*np.sum(diferences)
    auxtheta1 = theta[1] - alpha/m*np.sum(np.dot(diferences, X))
    return [auxtheta0, auxtheta1]

def optimusThetaAndPlot_Part1(values, X, Y, m, n):
    # Init gradient descendant
    loops = 1500
    alpha = 0.01
    theta = np.zeros(m)

    cost = compute_cost_function(theta, X, Y, m)
    print("Initial cost:",cost)

    for i in range(loops):
        # Update theta values
        theta = update_theta(theta, alpha, X, Y, m)
    
    # Compute cost function
    cost = compute_cost_function(theta, X, Y, m)

    print("The computed line is y = ",theta[0],"+ ",theta[1],"x")
    print("Final cost:",cost)
    
    #Example
    citizens=7
    print("Example prediction for 70.000 citizens: " + str((theta[0] + theta[1]*citizens)*10000) + "$")
    
    #Plot data points and aproxximation line
    plt.figure()
    plt.scatter(values[:,0], values[:,1], c='blue')
    plt.plot(X, applyH(theta, X), c='red')
    plt.xlabel("Población de la ciudad en 10.000s")
    plt.ylabel("Ingresos en $10.000s")
    plt.legend()
    plt.savefig('regresion_lineal.png')
    
    return theta
    

def make_data(t0_range, t1_range, X, Y, m):
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
        Coste[ix, iy] = compute_cost_function([Theta0[ix, iy], Theta1[ix, iy]], X, Y, m)
    return [Theta0, Theta1, Coste]
    
def plotsCostFunction_Part1_1(X, Y, m, n, theta_opt):
    t0_range = [-10, 10]
    t1_range = [-1, 4]
    
    #Generate the matrix
    Theta0, Theta1, Coste = make_data(t0_range, t1_range, X, Y, m)
    
    #Plot contour
    plt.figure()
    plt.contour(Theta0, Theta1, Coste, np.logspace(-2, 3, 20), colors='blue')
    plt.scatter(theta_opt[0], theta_opt[1], c='red')
    plt.xlabel(r'$\theta_0$')
    plt.ylabel(r'$\theta_1$')
    plt.legend()
    plt.savefig('ContourRegresionLineal.png')

    #Plot surface
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.plot_surface(Theta0, Theta1, Coste, cmap=cm.coolwarm)
    plt.savefig('Surface_regresion_lineal.png')
    
def main():
    # Load values
    values = load_data("ex1data1.csv")
    m,n = values.shape
    
    X = values[:,n-2]
    Y = values[:,-1]
  
    theta_opt = optimusThetaAndPlot_Part1(values, X, Y, m, n)
    
    plotsCostFunction_Part1_1(X, Y, m, n, theta_opt)
    
    
if __name__ == "__main__":
    main()