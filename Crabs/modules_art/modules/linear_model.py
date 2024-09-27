import numpy as np
from scipy.special import expit
import time


class LinearModel:
    def __init__(
        self,
        loss_function,
        batch_size=100,
        step_alpha=1,
        step_beta=0, 
        tolerance=1e-5,
        max_iter=1000,
        random_seed=153,
        w = None,
        **kwargs
    ):
        """
        Parameters
        ----------
        loss_function : BaseLoss inherited instance
            Loss function to use
        batch_size : int
        step_alpha : float
        step_beta : float
            step_alpha and step_beta define the learning rate behaviour
        tolerance : float
            Tolerace for stop criterio.
        max_iter : int
            Max amount of epoches in method.
        """
        self.loss_function = loss_function
        self.batch_size = batch_size
        self.step_alpha = step_alpha
        self.step_beta = step_beta
        self.tolerance = tolerance
        self.max_iter = max_iter
        self.random_seed = random_seed

    def fit(self, X, y, w_0=None, trace=False, X_val=None, y_val=None):
        """

        Parameters
        ----------
        X : numpy.ndarray or scipy.sparse.csr_matrix
            2d matrix, training set.
        y : numpy.ndarray
            1d vector, target values.
        w_0 : numpy.ndarray
            1d vector in binary classification.
            2d matrix in multiclass classification.
            Initial approximation for SGD method.
        trace : bool
            If True need to calculate metrics on each iteration.
        X_val : numpy.ndarray or scipy.sparse.csr_matrix
            2d matrix, validation set.
        y_val: numpy.ndarrays
            1d vector, target values for validation set.

        Returns
        -------
        : dict
            Keys are 'time', 'func', 'func_val'.
            Each key correspond to list of metric values after each training epoch.
        """
        w = w_0
        k = 1
        history = {'time': [], 'func': [], 'func_val': []}
        
        if(w_0 is None):
        	w = np.ones((X.shape[1]+1))
        ind = X.shape[0]
        if(self.batch_size != None):
        	ind = self.batch_size
        start_time = time.time()
        loss = self.loss_function.func(X,y,w)
        while(loss > self.tolerance and k <= self.max_iter):
        	indices = np.arange(X.shape[0])
        	np.random.shuffle(indices)
        	inds = indices[:ind]
        	batch = [indices[i : i+self.batch_size] for i in range(0, X.shape[0], self.batch_size)]
        	for b in batch:
        		w = w - (self.step_alpha/(k**self.step_beta))*self.loss_function.grad(X[b,:],y[b],w)
        		
        	k+=1
        	loss = self.loss_function.func(X,y,w)
        	if(trace):
        		iteration_time = time.time() - start_time
        		history['time'].append(iteration_time)
        		history['func'].append(loss)
        		if(X_val is None or y_val is None):
        			history['func_val'].append(np.nan) 
        		else:
        			history['func_val'].append(self.loss_function.func(X_val,y_val,w))  

        self.w = w
        return history
    def predict(self, X, threshold=0):
        """

        Parameters
        ----------
        X : numpy.ndarray or scipy.sparse.csr_matrix
            2d matrix, test set.
        threshold : float
            Chosen target binarization threshold.

        Returns
        -------
        : numpy.ndarray
            answers on a test set
        """

        X_n = np.hstack((np.ones((X.shape[0],1)),X))
        l = X_n @ self.w.T
        l = np.sign(l.T - threshold)
        return l.ravel()

    def get_weights(self):
        """
        Get model weights

        Returns
        -------
        : numpy.ndarray
            1d vector in binary classification.
            2d matrix in multiclass classification.
            Initial approximation for SGD method.
        """
        return self.w[1:]

    def get_objective(self, X, y):
        """
        Get objective.

        Parameters
        ----------
        X : numpy.ndarray or scipy.sparse.csr_matrix
            2d matrix.
        y : numpy.ndarray
            1d vector, target values for X.

        Returns
        -------
        : float
        """
        return self.loss_function.func(X,y,self.w)

    def get_bias(self):
        """
        Get model bias

        Returns
        -------
        : float
            model bias
        """
        return self.w[0]
