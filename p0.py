# -*- coding: utf-8 -*-
import math
import random
import numpy as np
from scipy import integrate
import time
import matplotlib.pyplot as plt

def exampleFunction(x):
    return math.sin(x)

def aproxMaximo(fun, a, b):
    #Devuelve el máximo de la función en el intervalo a,b
    return 1

def integra_mc(fun, a, b, num_puntos):
    m = aproxMaximo(fun,a,b)
    
    tic = time.process_time()
    
    #Generamos los puntos en el rectángulo
    x = np.random.uniform(a,b,num_puntos)
    y = np.random.uniform(0,m,num_puntos)
    
    #Para cada punto calculamos el valor de la función en ese x
    im = np.vectorize(exampleFunction)(x)
    
    #Nos interesan los puntos con su y menor que la imagen de x
    area = np.sum(y < im)/num_puntos*(b-a)*m
    
    toc = time.process_time()

    print("Integral:" + str(area))
    print("Tiempo: " + str(1000*(toc - tic)))
    return 1000*(toc - tic)

def integra_mc_lenta(fun, a, b, num_puntos):
    cont = 0
    m = aproxMaximo(fun,a,b)
    
    tic = time.process_time()
    
    for i in range (0, num_puntos):
        #Generamos un punto aleatorio
        x = random.randrange(a, b)
        y = random.randrange(0, m)
        #Calculamos la imagen de la función en ese x
        im = exampleFunction(x)
        #Si queda por debajo lo sumamos
        if y < im:
            cont = cont + 1
    
    area = cont/num_puntos*(b-a)*m
    
    toc = time.process_time()
    
    print("Integral:" + str(area))
    print("Tiempo: " + str(1000*(toc - tic)))
    return 1000*(toc - tic)

def main():
    num_puntos = 10000000
    a = 0
    b = 3
    print("Comprobación con scipy: " + str(integrate.quad(exampleFunction, a,b)[0]))
    print("-------------------- VECTORIZADO --------------------")
    integra_mc(exampleFunction, a, b, num_puntos)

    
    print("-------------------- BUCLES --------------------")
    integra_mc_lenta(exampleFunction, a, b, num_puntos)
    

def compara_tiempos():
    sizes = np.linspace(100, 10000000, 20)
    times_fast = []
    times_slow = []
    a = 0
    b = 3
    for size in sizes:
        times_fast += [integra_mc(exampleFunction, a, b, int(size))]
        times_slow += [integra_mc_lenta(exampleFunction, a, b, int(size))]
    
    plt.figure()
    plt.scatter(sizes, times_fast, c='blue', label='vector')
    plt.scatter(sizes, times_slow, c='red', label='bucle')
    plt.xlabel("Número de puntos")
    plt.ylabel("Tiempo (ms)")
    plt.legend()
    plt.savefig('time.png')
    
    
    
compara_tiempos()