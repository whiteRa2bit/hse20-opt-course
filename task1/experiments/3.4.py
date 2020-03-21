import numpy as np
import optimization
import oracles
from matplotlib import pyplot as plt


def generate_data(m=10000, n=8000, seed=31415):
    """
    Generates syntethic data
    :param m: number of training samples
    :param n: number of features
    :param seed: random_seed
    :return:
    """
    np.random.seed(seed)
    A = np.random.randn(m, n)
    b = np.sign(np.random.randn(m))
    return A, b


def plot_function_values_vs_iteration(history, history_optimized):
    """
    :param history:
    :param history_optimized:
    :return:
    """
    plt.plot(history['func'])
    plt.plot(history_optimized['func'])
    plt.xlabel("Iteration")
    plt.ylabel("Function value")
    plt.legend(['Usual', 'Optimized'])
    plt.savefig("./3.4/function_values_vs_iteration.png")
    plt.show()


def plot_function_values_vs_time(history, history_optimized):
    """
    :param history:
    :param history_optimized:
    :return:
    """
    plt.plot(history['time'], history['func'])
    plt.plot(history_optimized['time'], history_optimized['func'])
    plt.xlabel("Time [sec]")
    plt.ylabel("Function value")
    plt.legend(['Usual', 'Optimized'])
    plt.savefig("./3.4/function_values_vs_time.png")
    plt.show()


def plot_grad_norm_vs_time(history, history_optimized):
    """
    :param history:
    :param history_optimized:
    :return:
    """
    history_norm0 = history['grad_norm'][0]
    history_optimized_norm0 = history_optimized['grad_norm'][0]
    plt.plot(history['time'], np.log(np.array(history['grad_norm']) ** 2 / history_norm0 ** 2))
    plt.plot(history_optimized['time'], np.log(np.array(history_optimized['grad_norm']) ** 2 / history_optimized_norm0 ** 2))
    plt.xlabel("Time [sec]", fontsize=14)
    plt.ylabel(r'$log\frac{\|\nabla f(x_{k})\|_{2}^{2}} {\| \nabla f(x_0)\|_{2}^{2}}$', fontsize=20)
    plt.tight_layout()
    plt.subplots_adjust(top=0.88)
    plt.legend(["Usual", "Optimized"])
    plt.savefig("./3.4/grad_norm_vs_time.png")
    plt.show()


def compare_oracles(oracle, oracle_optimized, x_init):
    """
    :param oracle: Oracle model for LogReg
    :param oracle_optimized: Oracle model for optimized LogReg
    :param x_init: Initial point for optimization
    :return:
    """
    [_, _, history] = optimization.gradient_descent(oracle, x_init,
                                                         line_search_options={'method': 'Wolfe', 'c': 1},
                                                         trace=True)
    [_, _, history_optimized] = optimization.gradient_descent(oracle_optimized, x_init,
                                                    line_search_options={'method': 'Wolfe', 'c': 1},
                                                    trace=True)

    plot_function_values_vs_iteration(history, history_optimized)
    plot_function_values_vs_time(history, history_optimized)
    plot_grad_norm_vs_time(history, history_optimized)


if __name__ == '__main__':
    A, b = generate_data()
    oracle = oracles.create_log_reg_oracle(A, b, 1 / len(b))
    oracle_optimized = oracles.create_log_reg_oracle(A, b, 1/len(b), oracle_type='optimized')
    x_init = np.zeros(A.shape[1])
    compare_oracles(oracle, oracle_optimized, x_init)
