import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt
from scipy.io import loadmat
from checkNNGradients import *
from displayData import *

def sigmoid(z):
    return np.divide(1, 1 + np.exp(-z))

def costFunction(X, y, theta1, theta2):
    a_1, a_2, H = forwardPropagation(X, y, theta1, theta2)    
    return -1 / len(X) * np.sum(y*np.log(H) + (1-y)*np.log(1-H))

def costFunctionRegularized(X, y, theta1, theta2, lamb):
    return costFunction(X, y, theta1, theta2) + lamb/(2*len(X))*(np.sum(np.power(theta1, 2)) + np.sum(np.power(theta2, 2)))

def gradient(X, y, theta1, theta2):
    # Init delta
    Delta1 = np.zeros(np.shape(theta1))
    Delta2 = np.zeros(np.shape(theta2))

    # Perform forward propagation
    a_1, a_2, a_3 = forwardPropagation(X, y, theta1, theta2)

    for i in range(len(X)):
        delta_3 = a_3[i, :] - y[i]
        delta_2 = np.dot(theta2.T, delta_3) * (a_2[i, :]*(1 - a_2[i, :]))

        Delta1 = Delta1 + np.dot(delta_2[1:, np.newaxis], a_1[i, :][np.newaxis, :])
        Delta2 = Delta2 + np.dot(delta_3[:, np.newaxis], a_2[i, :][np.newaxis, :])

    return np.concatenate((np.ravel(Delta1), np.ravel(Delta2)))/len(X)

def gradientRegularized(X, y, theta1, theta2, lamb):
    # Compute gradient
    grad = gradient(X, y, theta1, theta2)
    
    # Init delta
    m,n = np.shape(theta1)
    Delta1 = np.reshape(grad[:m*n], theta1.shape)
    Delta2 = np.reshape(grad[m*n:], theta2.shape)
    
    #Compute regularized gradient
    Delta1[:, 1:] = Delta1[:, 1:] + lamb/len(X)*theta1[:, 1:]
    Delta2[:, 1:] = Delta2[:, 1:] + lamb/len(X)*theta2[:, 1:]
    
    return np.concatenate((np.ravel(Delta1), np.ravel(Delta2)))

def forwardPropagation(X,  y, theta1, theta2):
    # Input layer
    a_1 = X
    # Add 1s column to a_1
    a_1 = np.hstack([np.ones([X.shape[0],1]), X])

    # Hidden layer
    z_2 = np.dot(a_1, np.transpose(theta1))
    a_2 = sigmoid(z_2)
    # Add 1s column to a_2
    a_2 = np.hstack([np.ones([a_2.shape[0],1]), a_2])

    # Output layer
    z_3 = np.dot(a_2, np.transpose(theta2))
    a_3 = sigmoid(z_3) 

    return a_1, a_2, a_3

def backwardPropagation(nn_params, input_layer_size, hidden_layer_size, num_labels , X, y , lamb):
    # Init theta
    Theta1 = np.reshape(nn_params[:hidden_layer_size*(input_layer_size + 1)], (hidden_layer_size, (input_layer_size + 1)))
    Theta2 = np.reshape(nn_params[hidden_layer_size*(input_layer_size + 1):], (num_labels, (hidden_layer_size + 1)))

    # Perform regularized cost
    grad = gradientRegularized(X, y, Theta1, Theta2, lamb)
    cost = costFunctionRegularized(X, y, Theta1, Theta2, lamb)

    return (cost, grad)

def accuracyPercentage(X, y, H):
    row = 0
    hits = 0
    for k in range(len(X)):
        if np.argmax(H[row, :]) == np.argmax(y[row, :]):
            hits = hits + 1    
        row = row + 1
        
    acc = hits/len(X)*100        
    print("Hit rate:", acc, "%")

def trainNetwork(X, y, num_iterations, lamb):
    # Init params for network
    num_layers = 2
    nodes_per_layer = np.array([400, 25, 10]) 

    # Init epsilon 
    eps = [-0.12, 0.12]
    print("Epsilon:", eps)

    # eps = np.zeros(num_layers)
    # for i in range(n_layers):
    #     eps[i] = np.sqrt(6)/np.sqrt(nodes_per_layer[i] + nodes_per_layer[i+1])

    #Init theta
    Theta1 = np.random.random((nodes_per_layer[1], nodes_per_layer[0] + 1)) * 2*eps[0] - eps[0]
    Theta2 = np.random.random((nodes_per_layer[2], nodes_per_layer[1] + 1)) * 2*eps[1] - eps[1]
    
    initialTheta = np.concatenate((np.ravel(Theta1), np.ravel(Theta2)))
    
    #Opt theta
    optTheta = initialTheta
    fmin = opt.minimize(fun = backwardPropagation, x0 = optTheta, args = (nodes_per_layer[0], nodes_per_layer[1], nodes_per_layer[2], X, y, lamb), method = 'TNC', jac = True, options = {'maxiter' : num_iterations})
    optTheta = fmin['x']

    d1 = nodes_per_layer[1] * (nodes_per_layer[0] + 1)
    theta1 = np.reshape(optTheta[:d1], (nodes_per_layer[1], nodes_per_layer[0] + 1))
    theta2 = np.reshape(optTheta[d1:], (nodes_per_layer[2], nodes_per_layer[1] + 1))
    
    a_1, a_2, a_3 = forwardPropagation(X, y, theta1, theta2)

    # Perform accuracy
    accuracyPercentage(X, y, a_3)        

def main():
    # Load data
    data = loadmat("ex4data1.mat")  
    y = data['y']
    X = data['X']
    lamb = 1
    num_iterations = 70

    # Update Y for labels
    labels = np.unique(y)
    y = (y - 1)
    Y = np.zeros((len(y), len(labels)))
    for i in range(len(y)):
        Y[i][y[i]] = 1

    # Load weights
    weights = loadmat("ex4weights.mat")  
    theta1 , theta2 = weights['Theta1'] ,weights['Theta2']
    
    # Initial cost
    print("Initial cost:", costFunction(X, Y, theta1, theta2))
    # Regularized cost
    print("Regularized cost:", costFunctionRegularized(X, Y, theta1, theta2, lamb))
    # Gradient
    print("Gradient:", gradient(X, Y, theta1, theta2))
    # Regularized gradient
    print("Regularized gradient:", gradientRegularized(X, Y, theta1, theta2, lamb))

    # Check NNG gradient
    # checkNNGradients(backwardPropagation, lamb)

    # Train network
    trainNetwork(X, Y, num_iterations, lamb)

if __name__ == "__main__":
    main()