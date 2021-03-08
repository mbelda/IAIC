import numpy as np
from pandas.io.parsers import  read_csv
import matplotlib.pyplot as plt

def normalize(X):
    m = np.shape(X)[1]
    mean_X = np.ones(m)
    var_X = np.ones(m)
    for i in range(m):
        if i != 0:
            mean_X[i] = np.mean(X[:, i])
            var_X[i] = np.std(X[:, i])
            X[:, i] = (X[:, i] - mean_X[i])/var_X[i]
    return X, mean_X, var_X

def show_cost_function(loops, cost, title):
    plt.figure()
    plt.plot(loops, cost,'o', c='blue', label='vector')
    plt.xlabel("Iterations")
    plt.ylabel("Cost")
    plt.legend()
    plt.savefig('p1-relation-'+ loops +'.png')

def compute_cost_function(X, Y, theta):
    computed_values = np.dot(X, theta)
    expected_values = Y
    aux = (computed_values - expected_values)
    square = np.matmul(np.transpose(aux), aux)
    cost = np.sum(square)/(2*len(X))
    return cost

def compute_cost_function_der(X, Y, theta, m):
    computed_values = np.dot(X, theta)
    expected_values = Y
    aux = (computed_values - expected_values)
    return 1/m * np.matmul(np.transpose(X), aux)

def gradient_descent_matrix_method(X, Y):
    # Init gradient descendant
    loops = 1500
    alpha = 0.01
    m = np.shape(X)[0]
    theta = np.zeros(m)
    theta_aux = np.zeros(m)
    cost = []
    # Normalize data
    X, mean_X, var_X = normalize(X)

    # Update theta & compute cost
    cost.append(compute_cost_function(X, Y, theta))
    print("Initial cost:", cost[0])
    for i in range(loops):
        # Update theta values
        theta_aux = theta - alpha*compute_cost_function_der(X, Y, theta, m)
        cost.append(compute_cost_function(X, Y, theta_aux))
    theta = theta_aux
    print("Final cost:", cost[m-1])
    show_cost_function(loops, cost)
    return theta, mean_X, var_X

def normal_method(X, Y):
    X_T = np.transpose(X)
    return np.matmul(np.matmul(np.linalg.pinv(np.matmul(X_T,X)),X_T),Y)

def main():
    # Load values
    data = np.asarray(read_csv("ex1data2.csv", header=None))
    X = data[: , :-1]
    Y = data[: , -1]
    m = np.shape(X)[0]
    X = np.hstack([np.ones([m, 1]), X])

    # gradient descent with matrix
    theta, mean_X, var_X = gradient_descent_matrix_method(X, Y)
    print(theta)

    # normal equation
    theta = normal_method(X, Y)
    print(theta)

if __name__ == "__main__":
    main()
