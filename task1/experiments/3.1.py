import numpy as np
import optimization
import oracles
from plot_trajectory_2d import plot_levels, plot_trajectory
from matplotlib import pyplot as plt


def save_comparison_plot(oracle, x_init, xrange=None, yrange=None, levels=None, plot_name='plot'):
    """
    Save trajectory plot for three different optimization methods (Wolfe, Armijo and Constant)
    :param oracle: Oracle model
    :param x_init: Initial point for optimization
    :param xrange: Range of x values for plot_levels()
    :param yrange: Range of y values for plot_levels()
    :param levels: Range of level values for plot_levels()
    :return: None
    """
    [_, _, history_wolf] = optimization.gradient_descent(oracle, x_init,
                                                         line_search_options={'method': 'Wolfe'},
                                                         trace=True)

    [_, _, history_armijo] = optimization.gradient_descent(oracle, x_init,
                                                           line_search_options={'method': 'Armijo'},
                                                           trace=True)
    [_, _, history_constant] = optimization.gradient_descent(oracle, x_init,
                                                             line_search_options={'method': 'Constant', 'c': 0.01},
                                                             trace=True)

    plot_levels(oracle.func, xrange=xrange, yrange=yrange, levels=levels)
    plot_trajectory(oracle.func, history_constant['x'], color='red')
    plot_trajectory(oracle.func, history_armijo['x'], color='orange')
    plot_trajectory(oracle.func, history_wolf['x'], color='blue')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend(['Constant, iterations number = {}'.format(len(history_constant['x']) - 1),
                'Armije, iterations number = {}'.format(len(history_armijo['x']) - 1),
                'Wolfe, iterations number = {}'.format(len(history_wolf['x']) - 1)])
    plt.savefig('./3.1/{}'.format(plot_name))
    plt.show()


if __name__ == '__main__':
    A = np.array([[1, 0],
                [0, 50]])
    b = np.array([0, 0])
    x_init = np.array([10.0, 4.0])
    oracle = oracles.QuadraticOracle(A, b)
    xrange = [-6, 20]
    yrange = None
    levels = [0, 16, 64, 128, 256]
    save_comparison_plot(oracle, x_init, xrange, yrange, levels, plot_name='plot1')

    A = np.array([[1, 0],
                  [0, 50]])
    b = np.array([0, 0])
    x_init = np.array([15.0, 0.0])
    oracle = oracles.QuadraticOracle(A, b)
    xrange = [-6, 20]
    yrange = None
    levels = [0, 16, 64, 128, 256]
    save_comparison_plot(oracle, x_init, xrange, yrange, levels, plot_name='plot2')

    A = np.array([[5, 0],
                  [0, 5]])
    b = np.array([10, -5])
    x_init = np.array([5.0, 3.0])
    oracle = oracles.QuadraticOracle(A, b)
    xrange = [-6, 10]
    yrange = None
    levels = [0, 16, 64, 128]
    save_comparison_plot(oracle, x_init, xrange, yrange, levels, plot_name='plot3')

    A = np.array([[15, 10],
                  [10, 25]])
    b = np.array([-10, -5])
    x_init = np.array([5.0, 3.0])
    oracle = oracles.QuadraticOracle(A, b)
    xrange = [-6, 10]
    yrange = None
    levels = [0, 16, 64, 128, 256, 512]
    save_comparison_plot(oracle, x_init, xrange, yrange, levels, plot_name='plot4')

    A = np.array([[15, 10],
                  [10, 25]])
    b = np.array([-10, -5])
    x_init = np.array([5.0, 3.0])
    oracle = oracles.QuadraticOracle(A, b)
    xrange = [-6, 10]
    yrange = None
    levels = [0, 16, 64, 128, 256, 512]
    save_comparison_plot(oracle, x_init, xrange, yrange, levels, plot_name='plot5')
