import numpy as np


class MultiLayerPerceptron(object):

    def __init__(self, act_fun='logsig'):

        if act_fun == 'tansig':
            self.fun = np.tanh
        elif act_fun == 'logsig':
            self.fun = logsig

    def train(self, x, y, n_hl, eta=0.01, alpha=0.5, MAX_EPOCH=1000, MIN_ERROR=0.1):

        self.n_ex = x.shape[0]  # number of examples
        self.m = x.shape[1]  # number of attributes.
        self.nc = y.shape[1]  # number of classes.
        self.n_hl = n_hl  # number of hidden layers.

        self.hl = []
        for i in range(0, self.n_hl):
            self.hl.append(Perceptron(self.m))

        self.ol = []
        for i in range(0, self.nc):
            self.ol.append(Perceptron(self.n_hl))

        self.x = []
        self.x.append(np.zeros((self.m, 1)))
        self.x.append(np.zeros((self.n_hl, 1)))
        self.x.append(np.zeros((self.nc, 1)))

        delta = []
        delta.append(np.zeros(self.n_hl))
        delta.append(np.zeros(self.nc))

        ''' w_t will be used in momentum trainning '''
        w_t = [[], [], []]  # each position is the w delayed.
        w_ant = [[], []]  # each position is the w of a layer.
        for i in range(0, self.n_hl):
            w_ant[0].append([])
            w_ant[0][i].append(np.zeros(self.m + 1))
            w_ant[0][i] = self.hl[i].w

        for i in range(0, self.nc):
            w_ant[1].append([])
            w_ant[1][i].append(np.zeros(self.n_hl + 1))
            w_ant[1][i] = self.ol[i].w

        w_t[0] = w_ant
        w_t[1] = w_ant
        w_t[2] = w_ant
        del w_ant

        erro_total = np.arange(100.0, 95.0, -1)
        for i in range(0, MAX_EPOCH):

            if erro_total[-1] < MIN_ERROR:
                print 'Minimum error reached.'
                break
            elif np.mean(np.diff(erro_total)) >= 0.01 * erro_total[-1]:
                print 'Error raising. Stopping trainning.'
                break
            erro_total[0] = erro_total[1]
            erro_total[1] = erro_total[2]
            erro_total[2] = erro_total[3]
            erro_total[3] = erro_total[4]

            erro_total[-1] = 0
            ind_rand = np.arange(0, self.n_ex)
            np.random.shuffle(ind_rand)  # indices em ordem aleatoria.

            for ex in ind_rand:

                ''' Update the delayed weights '''
                w_t[2] = w_t[1]
                w_t[1] = w_t[0]
                for n in range(0, self.n_hl):
                    w_t[0][0][n] = self.hl[n].w
                for n in range(0, self.nc):
                    w_t[0][1][n] = self.ol[n].w

                self.x[0] = x[ex, :]  # new example.

                ''' Calculate the output'''
                for n in range(0, self.n_hl):
                    self.x[1][n] = self.hl[n].activate(self.x[0])

                for n in range(0, self.nc):
                    self.x[2][n] = self.ol[n].activate(self.x[1])

                ''' Calculate the error'''
                erro = np.reshape(y[ex, :], (-1, 1)) - self.x[2]

                ''' Update the weights of the output layer'''
                for n in range(0, self.nc):
                    delta[1][n] = self.x[
                        2][n] * (1 - self.x[2][n]) * (y[ex, n] - self.x[2][n])
                    self.ol[n].w = self.update_weight(
                        self.ol[n].w, w_t[1][1][n], self.x[1], delta[1][n], eta, alpha)

                ''' Update the weights of the hidden layer'''
                for n in range(0, self.n_hl):
                    sum_err = 0
                    for n2 in range(0, self.nc):
                        sum_err = sum_err + self.ol[n2].w[n + 1] * delta[1][n2]

                    delta[0][n] = self.x[1][n] * (1 - self.x[1][n]) * sum_err
                    self.hl[n].w = self.update_weight(
                        self.hl[n].w, w_t[1][0][n], self.x[0], delta[0][n], eta, alpha)

                erro_total[-1] = erro_total[-1] + np.sum(np.power(erro, 2))
            # rmse
            erro_total[-
                       1] = np.sqrt(erro_total[-1] / (self.nc * len(ind_rand)))

#             print '\nEpoca {:d}.'.format(i)
#             print 'Erro = '
#             print erro_total[-1]

    def update_weight(self, w, w_ant, x, delta, eta, alpha):
        x = np.reshape(x, (1, -1))
        x = np.hstack(([[1.0]], x))
        return w + np.reshape(eta * x * delta, (-1, 1)) + alpha * (w - w_ant)

    def activate(self, x):

        if x.size == self.m:
            x = np.reshape(x, (1, -1))

        y = np.zeros((x.shape[0], self.nc))

        for ex in range(0, x.shape[0]):
            self.x[0] = x[ex, :]

            for n in range(0, self.n_hl):
                self.x[1][n] = self.hl[n].activate(self.x[0])

            for n in range(0, self.nc):
                self.x[2][n] = self.ol[n].activate(self.x[1])

            y[ex, :] = np.transpose(self.x[2])

        return y


class Perceptron(object):

    def __init__(self, m=None, act_fun='logsig'):

        if act_fun == 'tansig':
            self.fun = np.tanh
        elif act_fun == 'logsig':
            self.fun = logsig

        if m != None:
            self.m = m
            self.w = np.random.rand(self.m + 1, 1)

    def train(self, x, y, eta=0.01, MAX_EPOCH=100):
        n = x.shape[0]

        self.m = x.shape[1]

        b = np.ones((n, 1))
        x = np.hstack((b, x))  # adding the bias input

        self.w = np.random.rand(self.m + 1, 1)

        for i in range(0, MAX_EPOCH):
            acum_error = 0
            for j in range(0, n):
                yhat = self.activate(x[j, :])
                e = y[j, 0] - yhat[0]
                if e != 0:
                    self.w = self.w + np.reshape(eta * e * x[j, :], (-1, 1))
                    acum_error = acum_error + e
            if acum_error == 0:
                break

    def activate(self, x):

        if x.size == self.m:
            x = np.reshape(x, (1, -1))
            x = np.hstack(([[1.0]], x))  # adding the bias input
            return self.fun(np.dot(x, self.w))
        elif x.size == self.m + 1:
            return self.fun(np.dot(x, self.w))
        else:
            b = np.ones((x.shape[0], 1))
            x = np.hstack((b, x))  # adding the bias input
            return self.fun(np.dot(x, self.w))


def logsig(x):
    return 1.0 / (1 + np.exp(-x))
