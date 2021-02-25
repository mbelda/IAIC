import numpy as np
from pandas.io.parsers import  read_csv
import matplotlib.pyplot as plt

def compute_cost_function(values, theta0, theta1):
    computed_values = theta0 + theta1*values[:, 0]
    expected_values = values[:, 1]

    cost = 1/(2*len(values[:, 0]))*np.sum((computed_values - expected_values)**2)
    return cost

def compute_cost_function_der_0(values, theta0, theta1):
    computed_values = theta0 + theta1*values[:, 0]
    expected_values = values[:, 1]

    cost = 1/(len(values[:, 0]))*np.sum((computed_values - expected_values))
    return cost

def compute_cost_function_der_1(values, theta0, theta1):
    computed_values = theta0 + theta1*values[:, 0]
    expected_values = values[:, 1]

    cost = 1/(len(values[:, 0]))*np.sum((computed_values - expected_values)*values[:, 0])
    return cost

def load_data():
    data = read_csv( "ex1data1.csv", header=None)
    return np.asarray(data)

def show_figure(values, theta0, theta1):
    plt.plot(values[:, 0], values[:, 1], 'o', color='black', label="Data")
    plt.plot(values[:, 0], theta0 + theta1*values[:, 0], label="Regression line")
    plt.xlabel("Población de la ciudad en 10.000s")
    plt.ylabel("Ingresos en $10.000s") 
    plt.legend()
    plt.savefig('p1-regression.png')


def main():
    # Load values
    values = load_data()

    # Init gradient descendant
    loops = 1500
    alpha = 0.01
    theta0 = 0
    theta1 = 0

    cost = compute_cost_function(values, theta0, theta1)
    print("Initial cost:",cost)

    for i in range(loops):
        # Update theta values
        theta0_aux = theta0 - alpha* compute_cost_function_der_0(values, theta0, theta1)
        theta1_aux = theta1 - alpha* compute_cost_function_der_1(values, theta0, theta1)
        theta0 = theta0_aux
        theta1 = theta1_aux

        # Compute cost function
        cost = compute_cost_function(values, theta0, theta1)

    print("The computed line is y = ",theta0,"+ ",theta1,"x")
    print("Final cost:",cost)
    # Mostrar gráfico
    show_figure(values, theta0, theta1)

if __name__ == "__main__":
    main()
    