import numpy as np
import scipy.optimize as opt
import matplotlib.pyplot as plt
from pandas.io.parsers import read_csv

def compute_sigmoid(Z):
    mat = 1/ (1 + np.exp(-Z))
    return mat

def compute_cost_function(theta, X, Y):
    sigmoid = compute_sigmoid(np.matmul(X, theta))
    aux_first = np.dot(Y, np.log(sigmoid))
    aux_second= np.dot((1 - Y), np.log(1 - sigmoid))
    return -1/len(X) * (aux_first + aux_second)

def compute_gradient(theta, X, Y):
    aux = np.subtract(compute_sigmoid(np.dot(X, theta)), Y)
    return np.dot(np.transpose(X), aux) / len(X)

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

def main():
    # Load values
    data = read_csv("ex2data1.csv", header=None).to_numpy().astype(float)
    m,n = data.shape
    
    X = data[:,:-1]
    Y = data[:,-1]
    m = np.shape(X)[0]
    matrix_X = np.hstack([np.ones([m, 1]), X])

    # Init thetas 
    theta = np.zeros(n)

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

if __name__ == "__main__":
    main()
    import numpy as np
import scipy.optimize as opt
import matplotlib.pyplot as plt
from pandas.io.parsers import read_csv

def compute_sigmoid(Z):
    mat = 1/ (1 + np.exp(-Z))
    return mat

def compute_cost_function(theta, X, Y):
    sigmoid = compute_sigmoid(np.matmul(X, theta))
    aux_first = np.dot(Y, np.log(sigmoid))
    aux_second= np.dot((1 - Y), np.log(1 - sigmoid))
    return -1/len(X) * (aux_first + aux_second)

def compute_gradient(theta, X, Y):
    aux = np.subtract(compute_sigmoid(np.dot(X, theta)), Y)
    return np.dot(np.transpose(X), aux) / len(X)
    
def compute_regularized_cost_function(theta, lamb, X, Y):
    return compute_cost_function(theta, X, Y) + lamb/(2*len(X))*np.sum(np.power(theta, 2))

def compute_regularized_gradient(theta, lamb, X, Y):
    reg = lamb/len(X)*theta
    reg[0] = 0
    return np.add(compute_gradient(theta, X, Y), reg)

def compute_optimal_params(theta, X, Y):
    result = opt.fmin_tnc(func=compute_cost_function, x0=theta, fprime=compute_gradient, args=(X,Y), disp=False)
    return result[0]

def show_boundary(X, Y, theta):
    x1_min, x1_max = np.min(X[:, 0]), np.max(X[:, 0])
    x2_min, x2_max = np.min(X[:, 1]), np.max(X[:, 1])
    xx1, xx2 = np.meshgrid(np.linspace(x1_min, x1_max), np.linspace(x2_min, x2_max))

    line = compute_sigmoid(np.c_[np.ones((xx1.ravel().shape[0], 1)), xx1.ravel(), xx2.ravel()].dot(theta))
    line = line.reshape(xx1.shape)

    plt.contour(xx1, xx2, line, [0.5], linewidths=2, colors='black')
    
def show_reg_boundary(X, Y, theta, pol):
    x1_min, x1_max = np.min(X[:, 0]), np.max(X[:, 0])
    x2_min, x2_max = np.min(X[:, 1]), np.max(X[:, 1])
    xx1, xx2 = np.meshgrid(np.linspace(x1_min, x1_max), np.linspace(x2_min, x2_max))
    
    line = compute_sigmoid(pol.fit_transform(np.c_[xx1.ravel(), xx2.ravel()]).dot(theta))
    line = line.reshape(xx1.shape)
    plt.contour(xx1, xx2, h, [0.5], linewidths=2, colors='black')

def show_regression_log(theta, X, Y):
    # plot points where y=1
    pos_adm = np.where(Y == 1)
    plt.scatter(X[pos_adm, 0], X[pos_adm, 1], marker='+',c='green', label='Admitted')

    # plot points where y=0
    pos_no_adm = np.where(Y == 0)
    plt.scatter(X[pos_no_adm, 0], X[pos_no_adm, 1], marker='o',c='orange', label='Not admitted')

    # plot line
    show_boundary(X, Y, theta)

    plt.xlabel('Exam 1 score')
    plt.ylabel('Exam 2 score')
    plt.legend(loc='upper right')
    plt.savefig('p2-regression-log.png')
    return pos_adm[0], pos_no_adm[0]

def show_regularized_regression_log(theta, X, Y, pol):
    # plot points where y=1
    pos_adm = np.where(Y == 1)
    plt.scatter(X[pos_adm, 0], X[pos_adm, 1], marker='+',c='green', label='Admitted')

    # plot points where y=0
    pos_no_adm = np.where(Y == 0)
    plt.scatter(X[pos_no_adm, 0], X[pos_no_adm, 1], marker='o',c='orange', label='Not admitted')

    # plot line
    show_reg_boundary(X, Y, theta, pol)
    
    plt.xlabel("Microchip test 1")
    plt.ylabel("Microchip test 2")
    plt.legend(loc='upper right')
    plt.title("Border with lambda=%.3f" % lamb)
    plt.savefig("p2-regression-log-lambda_%.3f.png" % lamb)
    
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
    
def logistic_regression(X, Y):
    m = np.shape(X)[0]
    matrix_X = np.hstack([np.ones([m, 1]), X])
    
    # Init thetas 
    theta = np.zeros(n)

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

def regularized_logistion_regression(X, Y):
    #Transform X to 6 degree polynomial
    pol = PolynomialFeatures(6)
    X_pol = pol.fit_transform(X)
    
    lambdas = [0.0, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1., 2., 50.]
   
    for lamb in lambdas:
        # Init theta
        theta = np.zeros(X_pol.shape[1])
        
        # Compute initial cost funtion
        cost = compute_regularized_cost_function(theta, lamb, X_pol, Y)
        print("Initial cost: ", cost)
        
        # Compute optimal params
        result = opt.fmin_tnc(func=compute_regularized_cost_function, x0=theta, fprime=compute_regularized_gradient, args=(lamb, X_pol,Y), disp=False)
        theta_opt = result[0]
        print('Optimal theta vector: ', theta_opt)
        cost = compute_regularized_cost_function(theta_opt, lamb, X_pol, Y)
        print("Optimal cost: ", cost)
        
        show_regularized_regression_log(X, Y, theta_opt, pol)

def main():
    # Load values
    data = read_csv("ex2data1.csv", header=None).to_numpy().astype(float)
    m,n = data.shape
    
    X = data[:,:-1]
    Y = data[:,-1]
    
    print("---------- LOGISTIC REGRESSION ----------")
    logistic_regression(X, Y)

    print("---------- REGULARIZED LOGISTIC REGRESSION ----------")
    regularized_logistion_regression

if __name__ == "__main__":
    main()
