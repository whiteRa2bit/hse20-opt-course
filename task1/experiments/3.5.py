import numpy as np
import optimization
import oracles
from matplotlib import pyplot as plt


def generate_data(m=1000, n=800, seed=31415):
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


def plot_grad_norm_vs_time(history, color, label):
    """
    :param history:
    :param history_optimized:
    :param color:
    :return:
    """
    history_norm0 = history['grad_norm'][0]
    plt.plot(history['time'], np.log(np.array(history['grad_norm']) ** 2 / history_norm0 ** 2), color=color,\
                                                                                                    label=label)
    plt.xlabel("Time [sec]", fontsize=14)
    plt.ylabel(r'$log\frac{\|\nabla f(x_{k})\|_{2}^{2}} {\| \nabla f(x_0)\|_{2}^{2}}$', fontsize=20)
    plt.tight_layout()
    plt.subplots_adjust(top=0.88)


def analyze_constant(oracle, x_init, c_values, colors):
    for i, c in enumerate(c_values):
        [_, _, history] = optimization.gradient_descent(oracle, x_init,
                                                        line_search_options={'method': 'Constant', 'c': c},
                                                        trace=True)
        plot_grad_norm_vs_time(history, colors[i], 'Constant, c={}'.format(c))


def analyze_armijo(oracle, x_init, c1_values, colors):
    for i, c1 in enumerate(c1_values):
        [_, _, history] = optimization.gradient_descent(oracle, x_init,
                                                        line_search_options={'method': 'Armijo', 'c1': c1},
                                                        trace=True)
        plot_grad_norm_vs_time(history, colors[i], "Armijo, c1={}".format(c1))


def analyze_wolfe(oracle, x_init, c2_values, colors):
    for i, c2 in enumerate(c2_values):
        [_, _, history] = optimization.gradient_descent(oracle, x_init,
                                                        line_search_options={'method': 'Wolfe', 'c2': c2},
                                                        trace=True)
        plot_grad_norm_vs_time(history, colors[i], 'Wolfe, c2={}'.format(c2))


if __name__ == '__main__':
    ### Logistic regression
    A, b = generate_data()
    oracle = oracles.create_log_reg_oracle(A, b, 1 / len(b), oracle_type='optimized')
    x_init = np.zeros(A.shape[1])

    c_values = [0.001, 0.01, 0.1]
    colors_constant = ['lime', 'green', 'darkgreen']
    analyze_constant(oracle, x_init, c_values, colors_constant)

    c1_values = [0.1, 0.25, 0.4]
    colors_armijo = ['red', 'darkred', 'lightcoral']
    analyze_armijo(oracle, x_init, c1_values, colors_armijo)

    c2_values = [0.6, 0.75, 0.9]
    colors_wolfe = ['blue', 'midnightblue', 'cornflowerblue']
    analyze_wolfe(oracle, x_init, c2_values, colors_wolfe)
    plt.legend()
    plt.savefig("./3.5/log_reg.png")
    plt.show()

