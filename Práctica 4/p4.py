import numpy as np
from pandas.io.parsers import read_csv
import matplotlib.pyplot as plt
import scipy.optimize as opt
from scipy.io import loadmat
from sklearn.preprocessing import PolynomialFeatures

def sigmoid(z):
    return np.divide(1, 1 + np.exp(-z))

def gradient(theta, X, Y):
    H = sigmoid(np.dot(X, theta))
    
    return 1 / len(Y) * np.dot(X.T, H - Y)

def regularized_gradient(theta, lamb, X, Y):
    gr = gradient(theta, X, Y)
    reg = lamb/len(X)*theta
    reg[0] = 0 #Theta_0 doesn't have extra cost
    return np.add(gr, reg)


def displayData(X):
    num_plots = int(np.size(X, 0)**.5)
    fig, ax = plt.subplots(num_plots, num_plots, sharex=True, sharey=True)
    plt.subplots_adjust(left=0, wspace=0, hspace=0)
    img_num = 0
    for i in range(num_plots):
        for j in range(num_plots):
            # Convert column vector into 20x20 pixel matrix
            # transpose
            img = X[img_num, :].reshape(20, 20).T
            ax[i][j].imshow(img, cmap='Greys')
            ax[i][j].set_axis_off()
            img_num += 1

    return (fig, ax)

def displayImage(im):
    fig2, ax2 = plt.subplots()
    image = im.reshape(20, 20).T
    ax2.imshow(image, cmap='gray')
    return (fig2, ax2)

def neuronalNetwork(X, theta1, theta2):
    # Input layer
    a_1 = X
    a_1 = np.hstack([np.ones([X.shape[0],1]), X])

    # Hidden layer
    z_2 = np.dot(a_1, theta1.T)
    a_2 = sigmoid(z_2)
    a_2 = np.hstack([np.ones([a_2.shape[0],1]), a_2])

    # Output layer
    z_3 = np.dot(a_2, theta2.T)
    a_3 = sigmoid(z_3)
    
    return a_3

def cost(theta1, theta2, X, Y):
    H = neuronalNetwork(X, theta1, theta2)
    c1 = Y * np.log(H)
    c2 = (1 - Y) * np.log(1 - H)
    return -1 / X.shape[0] * np.sum(c1 + c2)

def regularizedCost(theta1, theta2, X, Y, reg):
    c = cost(theta1, theta2, X, Y)
    return c + reg / (2 * len(X)) * (np.sum(np.power(theta1[:,1:], 2)) + np.sum(np.power(theta2[:,1:], 2))) 


def backprop(params_rn, num_entradas, num_ocultas, num_etiquetas, X, Y, reg):
    d1 = num_ocultas * (num_entradas + 1)
    theta1 = np.reshape(params_rn[:d1], (num_ocultas, num_entradas + 1))
    theta2 =np.reshape(params_rn[d1:], (num_etiquetas, num_ocultas + 1))
    cost = regularizedCost(theta1, theta2, X, Y, reg)
    
    return cost
 

def main():
    #Load data
    data = loadmat('ex4data1.mat')
    y = data['y'].ravel()
    X = data['X']
    m = len(y)
    num_entradas = 400
    num_ocultas = 25
    num_labels = 10
    
    #Create label matrix Y
    y = (y - 1)
    Y = np.zeros((m, num_labels)) # (5000, 10)
    for i in range(m):
        Y[i][y[i]] = 1
    
    #Load network weights
    weights = loadmat ('ex4weights.mat')
    theta1, theta2 = weights['Theta1'], weights['Theta2']
    print(theta1.shape)
    print(theta2.shape)
    
    #Compute initial cost
    c = cost(theta1, theta2, X, Y)
    print("Initial cost: %.6f" % c)
    
    #Compute initial regularized cost
    reg = 1 #Regularization parameter
    c = regularizedCost(theta1, theta2, X, Y, reg)
    print("Initial regularized cost: %.6f" % c)
    
    #Concatenate neuronal network params
    thetaVec = np.concatenate((np.ravel(theta1), np.ravel(theta2)))
    
    c = backprop(thetaVec, num_entradas, num_ocultas, num_labels, X, Y, reg)
    print(c)
    
    
 
if __name__ == "__main__":
    main()
