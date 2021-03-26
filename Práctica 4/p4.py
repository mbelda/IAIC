import numpy as np
import scipy.optimize as opt
from scipy.io import loadmat
import matplotlib.pyplot as plt

def sigmoid(z):
    return np.divide(1, 1 + np.exp(-z))

def forwardPropagation(X, theta1, theta2):
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
    
    return a_1, a_2, a_3

def cost(theta1, theta2, X, Y):
    A1, A2, H = forwardPropagation(X, theta1, theta2)
    c1 = Y * np.log(H)
    c2 = (1 - Y) * np.log(1 - H)
    return -1 / X.shape[0] * np.sum(c1 + c2)

def regularized_cost(theta1, theta2, X, Y, reg):
    c = cost(theta1, theta2, X, Y)
    return c + reg / (2 * len(X)) * (np.sum(np.power(theta1[:,1:], 2))  \
                                     + np.sum(np.power(theta2[:,1:], 2))) 

def gradient(theta1, theta2, X, Y):
    #Compute neuronal network layers outputs
    A1, A2, H = forwardPropagation(X, theta1, theta2)
    
    #Compute deltas
    Delta1 = np.zeros(theta1.shape)
    Delta2 = np.zeros(theta2.shape)
    for t in range(len(X)):
        a1t = A1[t, :] # (401,)
        a2t = A2[t, :] # (26,)
        ht = H[t, :] # (10,)
        yt = Y[t] # (10,)
        d3t = ht - yt # (10,)
        d2t = np.dot(theta2.T, d3t) * (a2t * (1 - a2t)) # (26,)
        Delta1 = Delta1 + np.dot(d2t[1:, np.newaxis], a1t[np.newaxis, :])
        Delta2 = Delta2 + np.dot(d3t[:, np.newaxis], a2t[np.newaxis, :])
    
    Delta1 = Delta1/len(X)
    Delta2 = Delta2/len(X)
    
    return np.concatenate((np.ravel(Delta1), np.ravel(Delta2)))
   

def regularized_gradient(theta1, theta2, X, Y, reg_param):
    #Compute gradient
    grad = gradient(theta1, theta2, X, Y)
    
    #Unroll deltas
    m,n = theta1.shape
    Delta1 = np.reshape(grad[:m*n], theta1.shape)
    Delta2 = np.reshape(grad[m*n:], theta2.shape)
    
    #Compute regularized gradient
    Delta1[:, 1:] = Delta1[:, 1:] + reg_param/ len(X) * theta1[:, 1:]
    Delta2[:, 1:] = Delta2[:, 1:] + reg_param/ len(X) * theta2[:, 1:]
    
    return np.concatenate((np.ravel(Delta1), np.ravel(Delta2)))

def backPropagation(nn_params, input_layer_size, hidden_layer_size, num_labels, \
             X, Y, reg_param):
    #Unroll theta's
    d1 = hidden_layer_size * (input_layer_size + 1)
    theta1 = np.reshape(nn_params[:d1], (hidden_layer_size, input_layer_size + 1))
    theta2 = np.reshape(nn_params[d1:], (num_labels, hidden_layer_size + 1))
    
    #Compute regularized cost
    A1, A2, H = forwardPropagation(X, theta1, theta2)
    c1 = Y * np.log(H)
    c2 = (1 - Y) * np.log(1 - H)
    cost = -1 / X.shape[0] * np.sum(c1 + c2)
    reg_cost = cost + reg_param / (2 * len(X)) * (np.sum(np.power(theta1[:,1:], 2))  \
                                     + np.sum(np.power(theta2[:,1:], 2)))
    
    #Compute regularized gradient
    Delta1 = np.zeros(theta1.shape)
    Delta2 = np.zeros(theta2.shape)
    for t in range(len(X)):
        a1t = A1[t, :] # (401,)
        a2t = A2[t, :] # (26,)
        ht = H[t, :] # (10,)
        yt = Y[t] # (10,)
        d3t = ht - yt # (10,)
        d2t = np.dot(theta2.T, d3t) * (a2t * (1 - a2t)) # (26,)
        Delta1 = Delta1 + np.dot(d2t[1:, np.newaxis], a1t[np.newaxis, :])
        Delta2 = Delta2 + np.dot(d3t[:, np.newaxis], a2t[np.newaxis, :])

    Delta1 = Delta1/len(X)
    Delta2 = Delta2/len(X)
    
    Delta1[:, 1:] = Delta1[:, 1:] + reg_param / len(X) * theta1[:, 1:]
    Delta2[:, 1:] = Delta2[:, 1:] + reg_param / len(X) * theta2[:, 1:]
    
    DeltaVec = np.concatenate((np.ravel(Delta1), np.ravel(Delta2)))
    
    return reg_cost, DeltaVec

#-----------------------------------------------------------------------------
#-----------------------------------------------------------------------------
#-------------------------------- TEST CODE ----------------------------------
#-----------------------------------------------------------------------------
#-----------------------------------------------------------------------------
def debugInitializeWeights(fan_in, fan_out):
    """
    Initializes the weights of a layer with fan_in incoming connections and
    fan_out outgoing connections using a fixed set of values.
    """

    # Set W to zero matrix
    W = np.zeros((fan_out, fan_in + 1))

    # Initialize W using "sin". This ensures that W is always of the same
    # values and will be useful in debugging.
    W = np.array([np.sin(w) for w in
                  range(np.size(W))]).reshape((np.size(W, 0), np.size(W, 1)))

    return W


def computeNumericalGradient(J, theta):
    """
    Computes the gradient of J around theta using finite differences and
    yields a numerical estimate of the gradient.
    """

    numgrad = np.zeros_like(theta)
    perturb = np.zeros_like(theta)
    tol = 1e-4

    for p in range(len(theta)):
        # Set perturbation vector
        perturb[p] = tol
        loss1 = J(theta - perturb)
        loss2 = J(theta + perturb)

        # Compute numerical gradient
        numgrad[p] = (loss2 - loss1) / (2 * tol)
        perturb[p] = 0

    return numgrad


def checkNNGradients(costNN, reg_param):
    """
    Creates a small neural network to check the back propogation gradients.
    Outputs the analytical gradients produced by the back prop code and the
    numerical gradients computed using the computeNumericalGradient function.
    These should result in very similar values.
    """
    # Set up small NN
    input_layer_size = 3
    hidden_layer_size = 5
    num_labels = 3
    m = 5

    # Generate some random test data
    Theta1 = debugInitializeWeights(hidden_layer_size, input_layer_size)
    Theta2 = debugInitializeWeights(num_labels, hidden_layer_size)

    # Reusing debugInitializeWeights to get random X
    X = debugInitializeWeights(input_layer_size - 1, m)

    # Set each element of y to be in [0,num_labels]
    y = [(i % num_labels) for i in range(m)]

    ys = np.zeros((m, num_labels))
    for i in range(m):
        ys[i, y[i]] = 1

    # Unroll parameters
    nn_params = np.append(Theta1, Theta2).reshape(-1)

    # Compute Cost
    cost, grad = costNN(nn_params,
                        input_layer_size,
                        hidden_layer_size,
                        num_labels,
                        X, ys, reg_param)

    def reduced_cost_func(p):
        """ Cheaply decorated nnCostFunction """
        return costNN(p, input_layer_size, hidden_layer_size, num_labels,
                      X, ys, reg_param)[0]

    numgrad = computeNumericalGradient(reduced_cost_func, nn_params)

    # Check two gradients
    print('grad shape: ', grad.shape)
    print('num grad shape: ', numgrad.shape)
    np.testing.assert_almost_equal(grad, numgrad)
    print("Gradient OK")
    return (grad - numgrad)


#-----------------------------------------------------------------------------
#-----------------------------------------------------------------------------
#-------------------------------- TEST CODE ----------------------------------
#-----------------------------------------------------------------------------
#-----------------------------------------------------------------------------


def trainNetwork(X, Y):
    #Compute epsilon for each layer
    n_layers = 2
    nodes_per_layer = np.array([400, 25, 10]) 
    eps = np.zeros(n_layers)
    for i in range(n_layers):
        eps[i] = np.sqrt(6)/np.sqrt(nodes_per_layer[i] + nodes_per_layer[i+1])
    print("Epsilon: " + str(eps))
     
    #Fixed parameters
    num_iters = np.array([40, 70, 100, 200])
    reg_params = np.array([0.1, 0.5, 1, 1.5, 2])
    
    acc = np.zeros(len(num_iters)*len(reg_params))
    i = 0
    for num_iter in num_iters:
        print("------------------- NUM ITERS = %d -------------------" % num_iter)
        j = 0
        for reg_param in reg_params:
            print("------------------- REG PARAM = %0.1f -------------------" % reg_param)
            #Initialize theta
            Theta1 = np.random.random((nodes_per_layer[1], nodes_per_layer[0] + 1)) \
                    * 2*eps[0] - eps[0]
            Theta2 = np.random.random((nodes_per_layer[2], nodes_per_layer[1] + 1)) \
                    * 2*eps[1] - eps[1]
            
            initialTheta = np.concatenate((np.ravel(Theta1), np.ravel(Theta2)))
            
            #Optimize theta
            optTheta = initialTheta
            fmin = opt.minimize(fun = backPropagation, \
                                    x0 = optTheta, \
                                    args = (nodes_per_layer[0], nodes_per_layer[1], \
                                          nodes_per_layer[2], X, Y, reg_param), \
                                    method = 'TNC', \
                                    jac = True, \
                                    options = {'maxiter' : num_iter})
            
            optTheta = fmin['x']
        
            #Unroll theta
            d1 = nodes_per_layer[1] * (nodes_per_layer[0] + 1)
            theta1 = np.reshape(optTheta[:d1], (nodes_per_layer[1], nodes_per_layer[0] + 1))
            theta2 = np.reshape(optTheta[d1:], (nodes_per_layer[2], nodes_per_layer[1] + 1))
            
            #Prediction results
            A1, A2, H = forwardPropagation(X, theta1, theta2)
            
            #Compute accuracy
            row = 0
            hits = 0
            for k in range(X.shape[0]):
                if np.argmax(H[row, :]) == np.argmax(Y[row, :]):
                    hits = hits + 1    
                row = row + 1
            acc[i*len(reg_params) + j] = hits/X.shape[0]*100
                
            print("Hit rate: %.2f%%" % acc[i*len(reg_params) + j])
            
            j = j + 1
        
        #Plot difference for reg_param
        plt.figure()
        plt.plot(reg_params, acc[i*len(reg_params):(i + 1)*len(reg_params)], \
                  marker='x', c='orange')
        plt.xlabel("Regularization parameter")
        plt.ylabel("Accuracy (%)")
        plt.ylim([80, 100])
        plt.title("Accuracy with %d iterations" % num_iter)
        plt.legend()
        #plt.savefig("accuracy_with_%d_iterations.png" % num_iter)
        plt.show()
        
        i = i + 1
        
    
    #Plot difference for num_iter
    for i in range(len(reg_params)):
        plt.figure()
        indexes = np.linspace(i, len(acc) - 1, num=len(num_iters)).astype(int)
        plt.plot(num_iters, acc[indexes], marker='x', c='blue')
        plt.xlabel("Number of iterations")
        plt.ylabel("Accuracy (%)")
        plt.ylim([80, 100])
        plt.title("Accuracy with lambda=%0.2f" % reg_params[i])
        plt.legend()
        #plt.savefig("accuracy_with_lambda_%.2f.png" % reg_params[i])
        plt.show()


def main():
    #Load data
    data = loadmat('ex4data1.mat')
    y = data['y'].ravel()
    X = data['X']
    m = len(y)
    num_labels = 10
    
    #Create label matrix Y
    y = (y - 1)
    Y = np.zeros((m, num_labels)) # (5000, 10)
    for i in range(m):
        Y[i][y[i]] = 1
    
    #Load network weights
    weights = loadmat ('ex4weights.mat')
    theta1, theta2 = weights['Theta1'], weights['Theta2']
    
    #Compute initial cost
    c = cost(theta1, theta2, X, Y)
    print("Initial cost: %.6f" % c)
    
    #Compute initial regularized cost
    reg = 1
    c = regularized_cost(theta1, theta2, X, Y, reg)
    print("Initial regularized cost: %.6f" % c)
    
    #Check gradient
    checkNNGradients(backPropagation, reg)
    
    #Train network
    trainNetwork(X, Y)
    
 
if __name__ == "__main__":
    main()
