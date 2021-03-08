import numpy as np
from pandas.io.parsers import  read_csv
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from mpl_toolkits.mplot3d import Axes3D


def compute_cost_function(X, Y, theta):
    computed_values = np.dot(X, theta)
    expected_values = Y
    aux = (computed_values - expected_values)
    square = np.matmul(np.transpose(aux), aux)

    cost = np.sum(square)/(2*len(X))
    return cost

def show_3D_cost(cost, T0, T1):
    fig = plt.figure()
    ax = Axes3D(fig)

    surf = ax.plot_surface(T0, T1, cost, cmap = cm.coolwarm, linewidth = 0, antialiased=False)
    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
    ax.set_xlabel(r'$\theta_0$')
    ax.set_ylabel(r'$\theta_1$')
    fig.colorbar(surf, shrink=0.5, aspect=5)

    plt.savefig('p1-regression-cost-3D.png')

def show_contour_cost(cost, T0, T1):
    fig = plt.figure()

    plt.contour(T0, T1, cost, np.logspace(-2, 3, 20),colors='blue')
    plt.xlabel(r'$\theta_0$')
    plt.ylabel(r'$\theta_1$') 

    plt.savefig('p1-regression-cost-contour.png')


def main():
    # Load values
    data = np.asarray(read_csv( "ex1data1.csv", header=None))
    X = data[: , :-1]
    Y = data[: , -1]
    m = np.shape(X)[0]
    X = np.hstack([np.ones([m, 1]), X])
    
    # Init thetas and range for cost
    theta0_range = [-10, 10]
    theta1_range = [-1, 4]

    step = 0.1
    T0 = np.arange(theta0_range[0], theta0_range[1], step)
    T1 = np.arange(theta1_range[0], theta1_range[1], step)
    T0, T1 = np.meshgrid(T0, T1)

    # Observe dimensions
    print(T0.shape)
    print(T1.shape)

    # Create an array with the same shape and type as T0
    cost = np.empty_like(T0)

    # Compute cost for every pair of coordinates
    for ix, iy in np.ndindex(T0.shape):
        cost[ix, iy] = compute_cost_function(X, Y, [T0[ix, iy], T1[ix, iy]])

    # Show 3D-figure cost
    show_3D_cost(cost, T0, T1)

    # Show contour cost
    show_contour_cost(cost, T0, T1)
    

if __name__ == "__main__":
    main()
    