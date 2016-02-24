import numpy as np


class ExtendedKalmanFilter(object):

    def __init__(self, P0, x0, Q, R, Afun, Ffun, Cfun, Gfun, f, g, param=None):
        '''
            Creates an extended Kalman filter.

            Parameters
            ----------
            P0 : numpy array
                Initial covariance matrix of the states.

            Q : numpy array
                Initial covariance matrix of the observation error.
                If all states have the same error variance, Q can be a scalar.
                Higher Q implies that the model is not reliable.

            R : numpy array
                Initial covariance matrix of the measurement error.
                If all outputs have the same error variance, Q can be a scalar.
                Higher R implies that the observations are not reliable.

            Afun : function
                Function that returns the linearization matrix A = df(x,u,w)/dx.
                Must be a function of the type: fun(x,u)

            Ffun : function
                Function that returns the linearization matrix F = df(x,u,w)/dw.

            Cfun : function
                Function that returns the linearization matrix C = dg(x,v)/dx.

            Gfun : function
                Function that returns the linearization matrix G = df(x,v)/dv.

            f: numpy array of functions
                Array of functions to predict the states.
                xk|k-1 = f(xk-1|k-1, u, w)

            g: numpy array of functions
                Array of functions to approximate the output.
                yk = f(x,v)

            param: optional, extra parameter used in the functions. 

            Returns
            -------
            Instance of the ekf class. with the following features:

            See Also
            --------

            Examples
            --------
        '''
        self.n = P0.shape[0]  # number of states
        self.P = P0
        self.x = x0

        if type(Q).__module__ == np.__name__:
            self.Q = Q
        else:
            self.Q = Q * np.eye(self.n)

        self.R = R

        self.Afun = Afun
        self.A = np.zeros((self.n, self.n))

        self.Ffun = Ffun
        self.F = np.zeros((self.n, self.n))

        self.Cfun = Cfun
        self.C = np.zeros(self.n)

        self.Gfun = Gfun
        self.G = 0.0

        self.f = f
        self.g = g

        self.param = param

    def predict(self, x=None, u=None):
        '''
            Predict stage.
        '''
        if x is None:
            x = self.x

        if self.param is None:
            self.x = self.f(x, u)
            self.__calculate_linearization_matrixes(self.x, u)
        else:
            self.x = self.f(x, u, self.param)
            self.__calculate_linearization_matrixes(self.x, u, self.param)

        self.P = np.dot(np.dot(self.A, self.P), np.transpose(
            self.A)) + np.dot(np.dot(self.F, self.Q), np.transpose(self.F))

        if self.param is None:
            return self.g(x, u)
        else:
            return self.g(x, u, self.param)

    def update(self, y, u):
        '''
            Update stage.
        '''
        if self.param is None:
            e = y - self.g(self.x, u)
        else:
            e = y - self.g(self.x, u, self.param)
        S = np.dot(np.dot(self.C, self.P), np.transpose(
            self.C)) + np.dot(np.dot(self.G, self.R), np.transpose(self.G))

        S = np.reshape(S, (1, 1))
        K = np.dot(self.P, np.transpose(self.C)) * np.linalg.inv(S)

        self.x = self.x + K * e
        self.P = self.P - np.dot(np.dot(K, self.C), self.P)

        if self.param is None:
            return self.g(self.x, u)
        else:
            return self.g(self.x, u, self.param)

    def __calculate_linearization_matrixes(self, x, u, param=None):
        '''
        Calculate the linearization matrixes to be used in EKF and EKS.
        '''
        if param is None:
            self.A = self.Afun(x, u)
            self.F = self.Ffun(x, u)
            self.C = self.Cfun(x, u)
            self.G = self.Gfun(x, u)
        else:
            self.A = self.Afun(x, u, param)
            self.F = self.Ffun(x, u, param)
            self.C = self.Cfun(x, u, param)
            self.G = self.Gfun(x, u, param)
