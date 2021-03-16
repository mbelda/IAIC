import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt
import scipy.optimize as opt

def sigmoid(z):
    return np.divide(1, 1 + np.exp(-z))

def cost(theta, X, Y):
    H = sigmoid(np.dot(X, theta))
    
    return -1 / len(X) * (np.dot(Y, np.log(H)) + np.dot((1-Y), np.log(1-H+1e-6)))

def regularized_cost(theta, lamb, X, Y):
    return cost(theta, X, Y) + lamb/(2*len(X))*np.sum(np.power(theta, 2))

def gradient(theta, X, Y):
    H = sigmoid(np.dot(X, theta))
    
    return 1 / len(Y) * np.dot(X.T, H - Y)

def regularized_gradient(theta, lamb, X, Y):
    gr = gradient(theta, X, Y)
    reg = lamb/len(X)*theta
    reg[0] = 0 #Theta_0 doesn't have extra cost
    return np.add(gr, reg)

def regularizedLogisticregression(X, Y, reg):
    theta = np.zeros(X.shape[1])
    
    #Optimize theta with fmin_tnc from scipy
    result = opt.fmin_tnc(func=regularized_cost, x0=theta, \
                          fprime=regularized_gradient, args=(reg, X,Y), \
                          disp=False)
    return result[0]

def accuracyPercentage(X, y, labels, theta, reg):
    # No funciona
    num_labels = len(labels)
    hit_radio_partial = np.zeros(num_labels)
    for i in range(num_labels):
        aux = np.dot(X, theta[i])
        pos_hits = np.where(y == labels[i])[0]
        hits = np.sum(sigmoid(aux[pos_hits]) > 0.5)
        hit_radio_partial[i] = hits/len(pos_hits) * 100

    print(hit_radio_partial)
    hit_ratio = np.mean(hit_radio_partial)
    print('The accuracy of our method is:', hit_ratio, '%')

def oneVsAll(X,  y, labels, reg) :
    """oneVsAll entrena varios clasificadores por regresión logística con
     término de regularización ’reg’ y devuelve el resultado en una matriz,
     donde la fila i−ésima corresponde al clasificador de la etiqueta i−ésima"""
    #Initial variables
    num_labels = len(labels)
    theta = np.zeros(shape=(num_labels, X.shape[1]))
    
    for i in range(num_labels):
        y_i = np.zeros(y.shape[0])
        np.put(y_i, labels[i], 1)
        
        #Compute thetas
        theta[i] = regularizedLogisticregression(X, y_i, reg)

    return theta
    

def show_sample_data(X, y):
    sample= np.random.choice(X.shape[0], 10)
    plt.imshow(X[sample, :].reshape(-1, 20).T)
    plt.axis('off')
    plt.savefig("numbers-example.png")

def main():
    # Load data
    data = loadmat("ex3data1.mat")  
    y = data['y']
    X = data['X']
    # Show sample data
    show_sample_data(X, y)

    #Add 1s column to X
    X_vec = np.hstack([np.ones([X.shape[0],1]), X])

    labels = np.unique(y)
    reg = 0.1
    print("----------------- MULTICLASS LOGISTIC REGRESSION -----------------")
    theta = oneVsAll(X_vec, y, labels, reg)
    accuracyPercentage(X_vec, y, labels, theta, reg)

if __name__ == "__main__":
    main()