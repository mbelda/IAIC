import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt
import scipy.optimize as opt

def sigmoid(z):
    return np.divide(1, 1 + np.exp(-z))

def accuracyPercentage(input, output, y):
    hits = 0
    row = 0
    fallos = np.zeros(10)
    for ex in input:
        if np.argmax(output[row]) + 1 == y[row]:
            hits += 1
        else:
            fallos[y[row]-1] += 1
        
        row += 1

    print("Hit rate: " + str(hits/input.shape[0]*100) + "%")

def neuronalNetwork(X,  y, theta1, theta2):
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


def main():
    # Load data
    data = loadmat("ex3data1.mat")  
    y = data['y']
    X = data['X']
    # Load weights
    weights = loadmat("ex3weights.mat")  
    theta1 , theta2 = weights['Theta1'] ,weights['Theta2']

    print("----------------- NEURONAL NETWORK -----------------")
    neuronalNetwork(X, y, theta1, theta2)

if __name__ == "__main__":
    main()