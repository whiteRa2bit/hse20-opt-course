import numpy as np
import optimization
import oracles
from matplotlib import pyplot as plt
from sklearn.datasets import load_svmlight_file


def plot_function_values_on_time(dataset, history_grad, history_newton):
    """
    :param dataset: Name of dataset
    :param history_grad: History of updates for gradient descent
    :param history_newton: History of updates for Newton
    :return:
    """
    plt.plot(history_grad['time'], history_grad['func'])
    plt.plot(history_newton['time'], history_newton['func'])
    plt.xlabel("Time [sec]", fontsize=14)
    plt.ylabel(r"$f(x_{k})$", fontsize=20)
    plt.legend(["Gradient descent", "Newton"])
    plt.savefig("./3.3/{}_function_values_on_time.png".format(dataset))
    plt.show()


def plot_grad_norm_values_on_time(dataset, history_grad, history_newton):
    """
        :param dataset: Name of dataset
        :param history_grad: History of updates for gradient descent
        :param history_newton: History of updates for Newton
        :return:
    """
    descent_grad_norm0 = history_grad['grad_norm'][0]
    newton_grad_norm0 = history_newton['grad_norm'][0]
    plt.plot(history_grad['time'], np.log(np.array(history_grad['grad_norm'])**2/descent_grad_norm0**2))
    plt.plot(history_newton['time'], np.log(np.array(history_newton['grad_norm'])**2/newton_grad_norm0**2))
    plt.xlabel("Time [sec]", fontsize=14)
    plt.ylabel(r'$log\frac{\|\nabla f(x_{k})\|_{2}^{2}} {\| \nabla f(x_0)\|_{2}^{2}}$', fontsize=20)
    plt.tight_layout()
    plt.subplots_adjust(top=0.88)
    plt.legend(["Gradient descent", "Newton"])
    plt.savefig("./3.3/{}_grad_norm_values_on_time.png".format(dataset))
    plt.show()


def plot_results(dataset):
    """
    Plots function values dependency and squared norm of gradient on time for
    gradient descent and newton optimization methods
    :param dataset: One of 'w8a', 'gissete', 'real-sim'
    :return:
    """
    available_datasets = ['w8a', 'gissete', 'real-sim']
    if dataset not in available_datasets:
        raise ValueError("Dataset {0} currently is not supported. Available datasets are: {1}".format(dataset,
                                                                                    ' '.join(available_datasets)))

    A, b = load_svmlight_file('./data/{}'.format(dataset))
    oracle = oracles.create_log_reg_oracle(A, b, 1/len(b))
    x_init = np.zeros(A.shape[1])

    [_, _, history_grad] = optimization.gradient_descent(oracle, x_init,
                                                         line_search_options={'method': 'Wolfe', 'c': 1},
                                                         trace=True)
    [_, _, history_newton] = optimization.newton(oracle, x_init,
                                                 line_search_options={'method': 'Wolfe', 'c': 1},
                                                 trace=True)
    plot_function_values_on_time(dataset, history_grad, history_newton)
    plot_grad_norm_values_on_time(dataset, history_grad, history_newton)


if __name__ == '__main__':
    plot_results('w8a')
    plot_results('gissete')
    plot_results('real-sim')
