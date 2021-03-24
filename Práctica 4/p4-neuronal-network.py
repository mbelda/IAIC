import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt
import scipy.optimize as opt

def sigmoid(z):
    return np.divide(1, 1 + np.exp(-z))

def costFunction(X, y, theta1, theta2):
    a_1, a_2, H = forwardPropagation(X, y, theta1, theta2)
    
    return -1 / len(X) * np.sum(np.dot(y.T, np.log(H)) + np.dot((1-y).T, np.log(1-H)))

def costFunctionRegularized(X, y, theta1, theta2, lamb):
    return costFunction(X, y, theta1, theta2) + lamb/(2*len(X))*(np.sum(np.power(theta1, 2)) + np.sum(np.power(theta2, 2)))

def accuracyPercentage(input, output, y):
    hits = 0
    row = 0
    fallos = np.zeros(len(np.unique(y)))
    for ex in input:
        if np.argmax(output[row]) + 1 == y[row]:
            hits += 1
        else:
            fallos[y[row]-1] += 1
        
        row += 1

def backwardPropagation(params_rn, num_entradas, num_ocultas, num_etiquetas , X, y , reg):
    cost = 0
    gradient = 0
    return (cost, gradient)

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

    # Hit rate
    accuracyPercentage(a_1, a_3, y)

    return a_1, a_2, a_3


def main():
    # Load data
    data = loadmat("ex4data1.mat")  
    y = data['y']
    X = data['X']
    # Load weights
    weights = loadmat("ex4weights.mat")  
    theta1 , theta2 = weights['Theta1'] ,weights['Theta2']
    
    labels = np.unique(y)
    print("Cost:", costFunction(X, y, theta1, theta2))

    lamb = 1
    print("Regularized cost:", costFunctionRegularized(X, y, theta1, theta2, lamb))
    

if __name__ == "__main__":
    main()