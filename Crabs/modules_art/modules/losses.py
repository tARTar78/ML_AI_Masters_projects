import numpy as np
import scipy
from scipy.special import expit
from scipy.special import logsumexp


class BaseLoss:
    """
    Base class for loss function.
    """

    def func(self, X, y, w):
        """
        Get loss function value at w.
        """
        raise NotImplementedError('Func oracle is not implemented.')

    def grad(self, X, y, w):
        """
        Get loss function gradient value at w.
        """
        raise NotImplementedError('Grad oracle is not implemented.')


class BinaryLogisticLoss(BaseLoss):
    """
    Loss function for binary logistic regression.
    It should support l2 regularization.
    """

    def __init__(self, l2_coef):
        """
        Parameters
        ----------
        l2_coef - l2 regularization coefficient
        """
        self.l2_coef = l2_coef

    def func(self, X, y, w):
        """
        Get loss function value for data X, target y and coefficient w; w = [bias, weights].

        Parameters
        ----------
        X : scipy.sparse.csr_matrix or numpy.ndarray
        y : 1d numpy.ndarray
        w : 1d numpy.ndarray

        Returns
        -------
        : float
        """
        X_n = np.hstack((np.ones((X.shape[0],1)),X))
        res = -y*(X_n @ w.T).T
        res = np.logaddexp(np.zeros((X.shape[0])),res)        
        return np.mean(res) + self.l2_coef * w[1:] @ w[1:].T

    def grad(self, X, y, w):
        """
        Get loss function gradient for data X, target y and coefficient w; w = [bias, weights].

        Parameters
        ----------
        X : scipy.sparse.csr_matrix or numpy.ndarray
        y : 1d numpy.ndarray
        w : 1d numpy.ndarray

        Returns
        -------
        : 1d numpy.ndarray
        """
        X_n = np.hstack((np.ones((X.shape[0],1)),X))
        res = -y*(X_n @ w.T).T
        res = scipy.special.expit(res)
        res = -y * res
        wb =w.copy()
        wb[0] = 0     
        X_n = (X_n * np.array(res)[:,np.newaxis]) + 2*self.l2_coef*wb
        np.mean(X_n,axis = 0)
        return np.mean(X_n,axis = 0)

