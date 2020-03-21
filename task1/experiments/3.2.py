import numpy as np
import scipy
import optimization
import oracles
from matplotlib import pyplot as plt


def generate_matrix(condition_num, n):
    """
    Generates positively defined diagonal matrix A with min value = 1 and
    max_value = condition num

    :param condition_num: condition number of matrix A
    :param n: dimensionality of matrix A
    :return: matrix in scipy.sparse.diags format
    """

    if n < 2:
        raise ValueError("n should be greater than 1")

    values = np.random.uniform(low=1, high=condition_num, size=(n-2))
    values = np.concatenate([[1, condition_num], values])
    np.random.shuffle(values)  # Shuffle values just in case
    # A = np.diag(values)
    A = scipy.sparse.diags(values)
    return A


def generate_task(condition_num, n, b_min=-1, b_max=1):
    """
    Generates random quadratic problem of size n with given condition number
    1/2 x^TAx - b^Tx.
    :param condition_num: condition number of matrix A
    :param n: number of features
    :param b_min: min value in b
    :param b_max: max value in b
    :return: A, b
    """
    A = generate_matrix(condition_num, n)
    b = np.random.uniform(low=b_min, high=b_max, size=n) * condition_num

    return A, b


def get_iteration_num_for_fixed_n(condition_num_values, n, x_init, method_name, b_min=-1, b_max=1):
    """
    Returns an array of iterations number for gradient descent
                    for each value in condition_num_values for given n
    :param condition_num_values: array of condition numbers
    :param n: number of features
    :param x_init: initial point for gradient descent
    :param b_min: min value in b
    :param b_max: max value in b
    :param method_name: 'Wolfe', 'Armijo' or 'Constant'
    :return: an array with number of iterations for each value in condition_num_values
    """
    iteration_num_values = []
    for condition_num in condition_num_values:
        A, b = generate_task(condition_num, n, b_min, b_max)
        oracle = oracles.QuadraticOracle(A, b)
        [x_star, _, history] = optimization.gradient_descent(oracle, x_init, \
                                                               line_search_options={'method': method_name, 'c': 0.001}, \
                                                               trace=True)
        # print(x_star)
        # print(history['grad_norm'])
        # print("====================================")
        iteration_num_values.append(len(history['grad_norm']))

    return iteration_num_values


def plot_iterations_dependency(condition_num_values, n_values, colors, method_name, iter_num=5):
    """
    Plots dependency of number of iterations on condition_num_values and number of features
    :param condition_num_values: array of condition numbers
    :param n_values: number of features array
    :param colors: colors array, corresponding
    :param iter_num:
    :return:
    """
    plt.title(method_name)
    for i, n in enumerate(n_values):
        x_init = np.zeros(n)
        for it in range(iter_num):
            iteration_num_values = get_iteration_num_for_fixed_n(condition_num_values, n, x_init, method_name)

            if it == 0:
                plt.plot(condition_num_values, iteration_num_values, color=colors[i], label='n = {}'.format(n))
            else:
                plt.plot(condition_num_values, iteration_num_values, color=colors[i])
    plt.xlabel("Condition number")
    plt.ylabel("Iterations number")
    plt.legend()
    plt.savefig('./3.2/iterations_dependency_{}.png'.format(method_name))
    plt.show()


if __name__ == '__main__':
    np.random.seed(32)
    condition_num_values = [10, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
    n_values = [10, 100, 1000, 10000]
    colors = ['red', 'blue', 'green', 'yellow', 'orange']

    plot_iterations_dependency(condition_num_values, n_values, colors, 'Wolfe')
    plot_iterations_dependency(condition_num_values, n_values, colors, 'Armijo')
    plot_iterations_dependency(condition_num_values, n_values, colors, 'Constant')


