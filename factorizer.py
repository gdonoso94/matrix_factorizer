from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.exceptions import NotFittedError
import numpy as np
import os


class MatrixFactorizer(BaseEstimator, ClassifierMixin):
    def __init__(self, p=None, alpha=None, beta=None, verbose=True):
        """Constructor for matrix factorization model

        :param p: latent space dimension
        :param alpha: regularization factor for matrix U
        :param beta: regularization factor for matrix V
        :param verbose: verbosity level
        """
        self.p = p
        self.alpha = alpha
        self.beta = beta
        self.u = None
        self.v = None
        self.r_hat = None
        self._verbose = verbose
        self.err = []

    def _loss_dense(self, r):
        """Computes loss function for dense matrix r with proposed formula

        :param r: ratings matrix
        :return: loss
        """
        r_hat = np.matmul(self.u, self.v.T)
        squared_err = (np.square(r - r_hat)).sum()
        x_reg = self._reg_err(self.u, self.alpha)
        y_reg = self._reg_err(self.v, self.beta)

        return squared_err + x_reg + y_reg

    @staticmethod
    def _reg_err(mat, l):
        """Calculates regularization error for proposed loss

        :param mat: input matrix
        :param l: lambda or regularization factor
        :return: regularization error
        """
        return l * np.multiply(mat, mat).sum()

    def fit(self, X=None, n_iter=100, tol=0.0001):
        """Fits U and V to X, and computes prediction matrix in every iteration

        :param X: matrix to be fitted
        :param n_iter: number of iterations
        :param tol: tolerance for early stopping
        :return: MatrixFactorizer model
        """
        if self.u is None and self.v is None:
            if X is not None:
                n, m = X.shape

                self.u = np.random.rand(n, self.p)
                self.v = np.random.rand(m, self.p)

                for i in range(n_iter):

                    self.u = np.matmul(X, np.matmul(self.v, np.linalg.inv(
                        np.matmul(self.v.T, self.v) + self.alpha * np.eye(self.p))))
                    self.v = np.matmul(X.T, np.matmul(self.u, np.linalg.inv(
                        np.matmul(self.u.T, self.u) + self.beta * np.eye(self.p))))
                    loss = self._loss_dense(X)
                    if i > 0:
                        if np.abs(self.err[-1] - loss) < tol:
                            break

                    if self._verbose:
                        if i % 20 == 0:
                            print(f"({i}/{n_iter}): loss --> {np.round(loss, 2)}")
                        else:
                            continue

                    self.err.append(loss)

                self.r_hat = np.matmul(self.u, self.v.T)
                return self
            else:
                raise Exception("No data to fit.")
        else:
            try:
                self.r_hat = np.matmul(self.u, self.v.T)
                return self
            except AttributeError:
                raise (Exception("U or V (or both) is none. Model cannot be fitted."))

    def load(self, path):
        """Load and fits MatrixFactorizer model from previous trainings

        :param path: path find model
        """
        if not os.path.exists(path):
            raise Exception("Model not found. Please, check model id.")

        self.u, self.v = np.load(f"{path}/u.npy", allow_pickle=False), np.load(f"{path}/v.npy", allow_pickle=False)
        self.fit()

    def predict(self, client_id, n_items=30):
        """Makes prediction of first n_items for a particular client_id

        :param client_id: row number in ratings matrix (client)
        :param n_items: number of items to return
        :return: sorted list of most appropriate items
        """
        if self.r_hat is None:
            raise NotFittedError("Not fitted model. Please, call fit method")
        else:
            return np.argsort(self.r_hat)[client_id, :][::-1][:n_items]
