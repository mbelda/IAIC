import numpy as np
from pandas.io.parsers import  read_csv
import matplotlib.pyplot as plt

def compute_cost_function(values, theta0, theta1):
    computed_values = theta0 + theta1*values[:, 0]
    expected_values = values[:, 1]

    cost = 1/(2*len(values[:, 0]))*np.sum((computed_values - expected_values)**2)
    return cost

def compute_cost_function_der(X, Y, thetai, thetai_1):
    computed_values = thetai + thetai_1*values[:, 0]
    expected_values = Y

    cost = 1/(len(X))*np.sum((computed_values - expected_values)*X)
    return cost

def gradient_descent_vector_method(X, Y):
    # Init gradient descendant
    loops = 1500
    alpha = 0.01
    m = np.shape(X)[0]
    theta = np.zeros(m)
    theta_aux = np.zeros(m)

    cost = compute_cost_function(values, theta0, theta1)
    print("Initial cost:",cost)
    for i in range(loops):
        for j in range(m):
            # Update theta values
            theta_aux = theta[j] - alpha* compute_cost_function_der(X[:,j], Y, theta[j], theta[j+1])
    theta = theta_aux
    print(theta)

def gradient_descent_matrix_method(X, Y):
    # Init gradient descendant
    loops = 1500
    alpha = 0.01
    m = np.shape(X)[0]
    theta = np.zeros(m)

    cost = compute_cost_function(values, theta0, theta1)
    print("Initial cost:",cost)

    for i in range(loops):
        # Update theta values
        theta0_aux = theta0 - alpha* compute_cost_function_der_0(values, theta0, theta1)
        theta1_aux = theta1 - alpha* compute_cost_function_der_1(values, theta0, theta1)
        theta0 = theta0_aux
        theta1 = theta1_aux

        # Compute cost function
        cost = compute_cost_function(values, theta0, theta1)

    print("The computed line is y = ",theta0,"+ ",theta1,"x")
    print("Final cost:",cost)

def normal_method(X, Y):
    # Before starting algorythm, we need to normalice these data   
    m = np.shape(X)[1]
    print(m)
    print(np.shape(X))
    for i in range(m):
        if i != 0:
            X[:, i] = (X[:, i] - np.mean(X[:, i]))/np.std(X[:, i])
    
    Y = (Y - np.mean(Y))/np.std(Y)
    print(X)
    print(Y)

    X_T = np.transpose(X)
    theta = np.matmul(np.matmul(np.linalg.pinv(np.matmul(X_T,X)),X_T),Y)
    print(theta)


def main():
    # Load values
    data = np.asarray(read_csv("ex1data2.csv", header=None))
    X = data[: , :-1]
    Y = data[: , -1]
    m = np.shape(X)[0]
    X = np.hstack([np.ones([m, 1]), X])

    # gradient descent with vectors

    # gradient descent with matrix

    # normal equation
    normal_method(X, Y)

    # Mostrar gráfico
    # show_figure(values, theta0, theta1)

if __name__ == "__main__":
    main()
    


    #Los valores de theta por el metodo del gradiente y por el metodo de la normal,
    #seran distintos pq estamos normalizando los vectores por lo que necesitamos calcular el normal
    # de la x que tomamos(guardar la media y mediana)



    # Representamos el cuadrado de forma matricial multiplicando la transpuesta por la original