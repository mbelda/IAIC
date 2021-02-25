import math
import random
import numpy as np
from scipy import integrate
import time
import matplotlib.pyplot as plt

def example_function(x):
    return math.sin(x)

def max_vector(fun, a, b):
    puntos = np.linspace(a,b,(b-a)*100)
    im = np.vectorize(fun)(puntos)
    return np.amax(im)

def max_loops(fun, a, b):
    values = []
    for i in range((b-a)*100):
        x = a + i * 1.0/100.0
        values.append(fun(x))    

    return max(values)


#La tuya tal cual
def integra_mc_vector(fun, a, b, num_puntos):
    m = max_vector(fun,a,b)
    
    x = np.random.uniform(a,b,num_puntos)
    y = np.random.uniform(0,m,num_puntos)
    
    im = np.vectorize(fun)(x)
    area = np.sum(y < im)/num_puntos*(b-a)*m
    return area

#La tuya tal cual
def integra_mc_loop(fun, a, b, num_puntos):
    cont = 0
    m = max_loops(fun,a,b)
    
    for i in range (0, num_puntos):
        x = random.uniform(a, b)
        y = random.uniform(0, m)
        im = fun(x)
        if y < im:
            cont = cont + 1
    
    area = cont/num_puntos*(b-a)*m
    return area

def show_figure(original, points_slow, points_fast):
    plt.figure()
    plt.plot(original, points_fast,'o', c='blue', label='vector')
    plt.plot(original, points_slow,'o', c='red', label='bucle')
    plt.xlabel("Número de puntos")
    plt.ylabel("Tiempo (ms)")
    plt.legend()
    plt.savefig('p0-montecarlo.png')

def main():
    fun = example_function
    num_puntos = 10000000
    a = 0
    b = 3
    sizes = np.linspace(100, num_puntos, 20)
    points_fast = []
    points_slow = []
    
    for rectangle in sizes:
        tic = time.process_time()
        area = integra_mc_vector(fun, a, b, int(rectangle))
        toc = time.process_time()
        tm = 1000*(toc - tic)
        points_fast.append(tm)

        if (rectangle == num_puntos):
            print("Compute area with numpy-vector")
            print("Area  :", area)
            print("Time  : ", tm)


        tic = time.process_time()
        area = integra_mc_loop(fun, a, b, int(rectangle))
        toc = time.process_time()
        tm = 1000*(toc - tic)
        points_slow.append(tm)
        
        if (rectangle == num_puntos):
            print("Compute area with loops")
            print("Area  :", area)
            print("Time  : ", tm)

    
    show_figure(sizes, points_slow, points_fast)


if __name__ == "__main__":
    main()   