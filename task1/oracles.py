import numpy as np
import scipy
from scipy.special import expit


class BaseSmoothOracle(object):
    """
    Base class for implementation of oracles.
    """
    def func(self, x):
        """
        Computes the value of function at point x.
        """
        raise NotImplementedError('Func oracle is not implemented.')

    def grad(self, x):
        """
        Computes the gradient at point x.
        """
        raise NotImplementedError('Grad oracle is not implemented.')

    def hess(self, x):
        """
        Computes the Hessian matrix at point x.
        """
        raise NotImplementedError('Hessian oracle is not implemented.')

    def func_directional(self, x, d, alpha):
        """
        Computes phi(alpha) = f(x + alpha*d).
        """
        return np.squeeze(self.func(x + alpha * d))

    def grad_directional(self, x, d, alpha):
        """
        Computes phi'(alpha) = (f(x + alpha*d))'_{alpha}
        """
        return np.squeeze(self.grad(x + alpha * d).dot(d))


class QuadraticOracle(BaseSmoothOracle):
    """
    Oracle for quadratic function:
       func(x) = 1/2 x^TAx - b^Tx.
    """

    def __init__(self, A, b):
        if not scipy.sparse.isspmatrix_dia(A) and not np.allclose(A, A.T):
            raise ValueError('A should be a symmetric matrix.')
        self.A = A
        self.b = b

    def func(self, x):
        return 0.5 * np.dot(self.A.dot(x), x) - self.b.dot(x)

    def grad(self, x):
        return self.A.dot(x) - self.b

    def hess(self, x):
        return self.A


class LogRegL2Oracle(BaseSmoothOracle):
    """
    Oracle for logistic regression with l2 regularization:
         func(x) = 1/m sum_i log(1 + exp(-b_i * a_i^T x)) + regcoef / 2 ||x||_2^2.

    Let A and b be parameters of the logistic regression (feature matrix
    and labels vector respectively).
    For user-friendly interface use create_log_reg_oracle()

    Parameters
    ----------
        matvec_Ax : function
            Computes matrix-vector product Ax, where x is a vector of size n.
        matvec_ATx : function of x
            Computes matrix-vector product A^Tx, where x is a vector of size m.
        matmat_ATsA : function
            Computes matrix-matrix-matrix product A^T * Diag(s) * A,
    """
    def __init__(self, matvec_Ax, matvec_ATx, matmat_ATsA, b, regcoef):
        self.matvec_Ax = matvec_Ax
        self.matvec_ATx = matvec_ATx
        self.matmat_ATsA = matmat_ATsA
        self.b = b
        self.regcoef = regcoef

    def func(self, x):
        # TODO: Implement
        m = len(self.b)
        n = len(x)
        output = -self.b * self.matvec_Ax(x)
        losses = np.logaddexp(0, output)
        reg_term = (self.regcoef/2) * np.linalg.norm(x)**2

        return (losses @ np.ones(m) / m) + reg_term

    def grad(self, x):
        # TODO: Implement
        m = len(self.b)
        residuals = scipy.special.expit(self.matvec_Ax(x)) - (self.b + 1)/2  # shape = (m, )
        reg_term = self.regcoef * x

        return (self.matvec_ATx(residuals) / m) + reg_term

    def hess(self, x):
        # TODO: Implement
        n = len(x)
        m = len(self.b)
        output = self.matvec_Ax(x)
        diag_values = scipy.special.expit(output) * (1 - scipy.special.expit(output))
        reg_term = self.regcoef * np.eye(n)

        return self.matmat_ATsA(diag_values)/m + reg_term


class LogRegL2OptimizedOracle(LogRegL2Oracle):
    """
    Oracle for logistic regression with l2 regularization
    with optimized *_directional methods (are used in line_search).

    For explanation see LogRegL2Oracle.
    """
    def __init__(self, matvec_Ax, matvec_ATx, matmat_ATsA, b, regcoef):
        super().__init__(matvec_Ax, matvec_ATx, matmat_ATsA, b, regcoef)
        self.x = None
        self.d = None
        self.x_hat = None
        self.Ax = None
        self.Ad = None
        self.Ax_hat = None

    def _update_ax(self, x: np.array):
        if np.all(x == self.x):
            return
        else:
            self.x = x
            self.Ax = self.matvec_Ax(x)

    def _update_ad(self, d: np.array):
        if np.all(d == self.d):
            return
        else:
            self.d = d
            self.Ad = self.matvec_Ax(self.d)

    def _update_xhat(self, x: np.array, d: np.array, alpha: float):
        if np.all(x + alpha * d == self.x_hat):
            return
        else:
            self.x_hat = x + alpha * d
            self.Ax_hat = self.Ax + alpha * self.Ad

    def func(self, x: np.array):
        # TODO: Implement
        m = len(self.b)
        n = len(x)

        if np.all(self.x_hat == x):
            output = -self.b * self.Ax_hat
            losses = np.logaddexp(0, output)
            reg_term = (self.regcoef / 2) * np.linalg.norm(x) ** 2

            return (losses @ np.ones(m) / m) + reg_term

        self._update_ax(x)

        output = -self.b * self.Ax
        losses = np.logaddexp(0, output)
        reg_term = (self.regcoef/2) * np.linalg.norm(x)**2

        return (losses @ np.ones(m) / m) + reg_term

    def grad(self, x: np.array):
        # TODO: Implement
        m = len(self.b)

        if np.all(self.x_hat == x):
            residuals = scipy.special.expit(self.Ax_hat) - (self.b + 1) / 2  # shape = (m, )
            reg_term = self.regcoef * self.x_hat

            return (self.matvec_ATx(residuals) / m) + reg_term
        else:
            self._update_ax(x)

            residuals = scipy.special.expit(self.Ax) - (self.b + 1)/2  # shape = (m, )
            reg_term = self.regcoef * x

            return (self.matvec_ATx(residuals) / m) + reg_term

    def hess(self, x: np.array):
        # TODO: Implement
        n = len(x)
        m = len(self.b)

        if np.all(self.x_hat == x):
            output = self.Ax_hat
            diag_values = scipy.special.expit(output) * (1 - scipy.special.expit(output))
            reg_term = self.regcoef * np.eye(n)

            return self.matmat_ATsA(diag_values) / m + reg_term

        else:
            self._update_ax(x)
            output = self.Ax
            diag_values = scipy.special.expit(output) * (1 - scipy.special.expit(output))
            reg_term = self.regcoef * np.eye(n)

            return self.matmat_ATsA(diag_values)/m + reg_term

    def func_directional(self, x, d, alpha):
        m = len(self.b)
        n = len(x)

        if alpha == 0:
            return self.func(x)

        if np.all(self.x_hat == x):
            self.Ax = self.Ax_hat
            self._update_ad(d)
            self._update_xhat(x, d, alpha)
        else:
            self._update_ax(x)
            self._update_ad(d)
            self._update_xhat(x, d, alpha)

        output = -self.b * (self.Ax + alpha*self.Ad)
        losses = np.logaddexp(0, output)
        reg_term = (self.regcoef/2) * np.linalg.norm(x+alpha*d)**2

        return np.squeeze((losses @ np.ones(m) / m) + reg_term)

    def grad_directional(self, x, d, alpha):
        m = len(self.b)

        if alpha == 0:
            return self.grad(x) @ d

        if np.all(self.x_hat == x):
            self.Ax = self.Ax_hat
            self._update_ad(d)
            self._update_xhat(x, d, alpha)
        else:
            self._update_ax(x)
            self._update_ad(d)
            self._update_xhat(x, d, alpha)

        residuals = scipy.special.expit(self.Ax + alpha * self.Ad) - (self.b + 1) / 2
        reg_term = self.regcoef * (x + alpha * d) @ d

        return (np.transpose(residuals)@self.Ad / m) + reg_term


def create_log_reg_oracle(A, b, regcoef, oracle_type='usual'):
    """
    Auxiliary function for creating logistic regression oracles.
        `oracle_type` must be either 'usual' or 'optimized'
    """
    def matvec_Ax(x):
        # TODO: implement proper matrix-vector multiplication
        return A @ x

    def matvec_ATx(x):
        # TODO: implement proper martix-vector multiplication
        return np.transpose(A) @ x

    def matmat_ATsA(s):
        # TODO: Implement
        return np.transpose(A) @ scipy.sparse.diags(s) @ A

    if oracle_type == 'usual':
        oracle = LogRegL2Oracle
    elif oracle_type == 'optimized':
        oracle = LogRegL2OptimizedOracle
    else:
        raise 'Unknown oracle_type=%s' % oracle_type
    return oracle(matvec_Ax, matvec_ATx, matmat_ATsA, b, regcoef)


def grad_finite_diff(func, x, eps=1e-8):
    """
    Returns approximation of the gradient using finite differences:
        result_i := (f(x + eps * e_i) - f(x)) / eps,
        where e_i are coordinate vectors:
        e_i = (0, 0, ..., 0, 1, 0, ..., 0)
                          >> i <<
    """
    # TODO: Implement numerical estimation of the gradient
    result = np.zeros(len(x))

    for i in range(len(x)):
        e = np.zeros(len(x))
        e[i] = 1
        result[i] = (func(x + eps * e) - func(x)) / eps

    return result


def hess_finite_diff(func, x, eps=1e-5):
    """
    Returns approximation of the Hessian using finite differences:
        result_{ij} := (f(x + eps * e_i + eps * e_j)
                               - f(x + eps * e_i)
                               - f(x + eps * e_j)
                               + f(x)) / eps^2,
        where e_i are coordinate vectors:
        e_i = (0, 0, ..., 0, 1, 0, ..., 0)
                          >> i <<
    """
    # TODO: Implement numerical estimation of the Hessian
    result = np.zeros((len(x), len(x)))

    for i in range(len(x)):
        for j in range(len(x)):
            ei = np.zeros(len(x))
            ej = np.zeros(len(x))
            ei[i] = 1
            ej[j] = 1
            result[i][j] = (func(x + eps * ei + eps * ej) - func(x + eps * ei) - func(x + eps * ej) + func(x)) / eps**2

    return result
