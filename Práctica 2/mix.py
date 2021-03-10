import numpy as np
import scipy.optimize as opt
import matplotlib.pyplot as plt
from pandas.io.parsers import read_csv
from sklearn.preprocessing import PolynomialFeatures

def load_data(file):
    return read_csv(file, header=None).to_numpy().astype(float)

def compute_sigmoid(Z):
    mat = 1/ (1 + np.exp(-Z))
    return mat

def compute_cost_function(theta, X, Y):
    sigmoid = compute_sigmoid(np.matmul(X, theta))
    aux_first = np.dot(Y, np.log(sigmoid))
    aux_second= np.dot((1 - Y), np.log(1 - sigmoid))
    return -1/len(X) * (aux_first + aux_second)

def regularized_cost(theta, lamb, X, Y):
    return compute_cost_function(theta, X, Y) + lamb/(2*len(X))*np.sum(np.power(theta, 2))

def compute_gradient(theta, X, Y):
    aux = np.subtract(compute_sigmoid(np.dot(X, theta)), Y)
    return np.dot(np.transpose(X), aux) / len(X)

def regularized_gradient(theta, lamb, X, Y):
    gr = compute_gradient(theta, X, Y)
    reg = lamb/len(X)*theta
    reg[0] = 0 #Theta_0 doesn't have extra cost
    return np.add(gr, reg)

def compute_optimal_params(theta, X, Y):
    result = opt.fmin_tnc(func=compute_cost_function, x0=theta, fprime=compute_gradient, args=(X,Y), disp=False)
    return result[0]

def show_border(X, Y, theta):
    x1_min, x1_max = np.min(X[:, 0]), np.max(X[:, 0])
    x2_min, x2_max = np.min(X[:, 1]), np.max(X[:, 1])
    xx1, xx2 = np.meshgrid(np.linspace(x1_min, x1_max), np.linspace(x2_min, x2_max))

    line = compute_sigmoid(np.c_[np.ones((xx1.ravel().shape[0], 1)), xx1.ravel(), xx2.ravel()].dot(theta))
    line = line.reshape(xx1.shape)

    plt.contour(xx1, xx2, line, [0.5], linewidths=2, colors='black')

def show_regression_log(theta, X, Y):
    # plot points where y=1
    pos_adm = np.where(Y == 1)
    plt.scatter(X[pos_adm, 0], X[pos_adm, 1], marker='+',c='green', label='Admitted')

    # plot points where y=0
    pos_no_adm = np.where(Y == 0)
    plt.scatter(X[pos_no_adm, 0], X[pos_no_adm, 1], marker='o',c='orange', label='Not admitted')

    # plot line
    show_border(X, Y, theta)

    plt.xlabel('Exam 1 score')
    plt.ylabel('Exam 2 score')
    plt.legend(loc='upper right')
    plt.savefig('p2-regression-log.png')
    return pos_adm[0], pos_no_adm[0]

def compute_accuracy(theta, X, pos_adm, pos_no_adm):
    aux = np.dot(X, theta)
    compute_adm = np.sum(compute_sigmoid(aux[pos_adm]) > 0.5)
    compute_no_adm = np.sum(compute_sigmoid(aux[pos_no_adm]) < 0.5)

    print('Total admitted', len(pos_adm))
    print('Total hit admitted', compute_adm)
    print('Total not admitted', len(pos_no_adm))
    print('Total hit not admitted', compute_no_adm)

    hit_ratio_adm = compute_adm/len(pos_adm) * 100
    hit_ratio_no_adm = compute_no_adm/len(pos_no_adm) * 100
    percentage = np.mean([hit_ratio_adm , hit_ratio_no_adm])
    print('The accuracy of our method is:', percentage, '%')

def plot_decisionboundary(X, Y, theta, poly):
    x1_min, x1_max = X[:, 0].min(), X[:, 0].max()
    x2_min, x2_max = X[:, 1].min(), X[:, 1].max()
    xx1, xx2 = np.meshgrid(np.linspace(x1_min, x1_max),
    np.linspace(x2_min, x2_max))
    h = compute_sigmoid(poly.fit_transform(np.c_[xx1.ravel(),
    xx2.ravel()]).dot(theta))
    h = h.reshape(xx1.shape)
    plt.contour(xx1, xx2, h, [0.5], linewidths=1, colors='g')

def logisticRegression(X, Y):
    #Add 1s column to X
    matrix_X = np.hstack([np.ones([X.shape[0], 1]), X])
    
    # Init thetas 
    theta = np.zeros(matrix_X.shape[1])

    # Compute initial cost funtion
    cost = compute_cost_function(theta, matrix_X, Y)
    print('Initial cost: ', cost)

    # Compute gradient cost funtion
    grad = compute_gradient(theta, matrix_X, Y)
    print('Gradient vector: ', grad)

    # Compute optimal params 
    theta_opt = compute_optimal_params(theta, matrix_X, Y)
    print('Optimal theta vector: ', theta_opt)
    cost_opt = compute_cost_function(theta_opt, matrix_X, Y)
    print('Optimal cost: ', cost_opt)

    pos_adm, pos_no_adm = show_regression_log(theta_opt, X, Y)
    compute_accuracy(theta_opt, matrix_X, pos_adm, pos_no_adm)

def regularizedLogisticregression(X, Y):
    #Transform X to 6 degree polynomial
    poly = PolynomialFeatures(6)
    X_pol = poly.fit_transform(X)
    
    lambdas = [0.0, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.]
   
    for lamb in lambdas:
        #Initial theta
        theta = np.zeros(X_pol.shape[1])
        
        #Initial cost
        print("----------- LAMBDA = " + str(lamb) + " -----------")
        print("Initial cost: " + str(regularized_cost(theta, lamb, X_pol, Y)))
        
        #Optimize theta with fmin_tnc from scipy
        result = opt.fmin_tnc(func=regularized_cost, x0=theta, fprime=regularized_gradient, args=(lamb, X_pol,Y), disp=False)
        theta = result[0]
        
        #Cost with optimized theta
        print("Optimized cost: " + str(regularized_cost(theta, lamb, X_pol, Y)))
        
        #Plot points and border
        pos = np.where(Y == 1)
        neg = np.where(Y == 0)
        plt.figure()
        plt.scatter(X[pos,0], X[pos,1], marker='+', c='g', label='accepted')
        plt.scatter(X[neg,0], X[neg,1], marker='o', c='orange', label='not accepted')
        plot_decisionboundary(X, Y, theta, poly)
        plt.xlabel("Microchip test 1")
        plt.ylabel("Microchip test 2")
        plt.legend(loc='upper right')
        plt.title("Border with lambda=%.3f" % lamb)
        plt.savefig("logistic_regression_circle_with_lambda_%.3f.png" % lamb)    

def main():
    # Load values
    data = load_data("ex2data1.csv")
    
    X = data[:,:-1]
    Y = data[:,-1]
    
    print("----------------- LOGISTIC REGRESSION -----------------")
    logisticRegression(X, Y)
    
    values = load_data("ex2data2.csv")
    
    X = values[:, :-1]
    Y = values[:,-1]
    print("----------------- REGULARIZED LOGISTIC REGRESSION -----------------")
    regularizedLogisticregression(X, Y)
    

    
if __name__ == "__main__":
    main()
    