# 4) Para a base de dados Iris (disponivel em http://archive.ics.uci.edu/ml/),
# embaralhe os dados. Use 75% dos dados para treinar uma MLP e 25% para teste.
# Calcule a medida de acuracia sobre o conjunto de teste. Repita esse
# procedimento 4 vezes e retorne a acuracia obtida em cada teste.

import utils
from artificial_neural_networks import MultiLayerPerceptron
import numpy as np
import numpy.matlib

x, y = utils.abrir_dados_iris('./bases/iris/iris.data')

for k in range(0, 4):
    print '\nInitializing test {:d}'.format(k+1)
    ytypes = np.union1d(y, y)

    new_y = np.zeros((x.shape[0], ytypes.size))
    for i in range(0, x.shape[0]):
        new_y[i, np.where(np.reshape(
            numpy.matlib.repmat(y[i], 1, ytypes.size), (1, -1)) == ytypes)[1][0]] = 1

    ind_rand = np.arange(0, y.shape[0])
    np.random.shuffle(ind_rand)  # indices em ordem aleatoria.
    ind_train = ind_rand[0:.75 * y.shape[0]]
    ind_test = ind_rand[.75 * y.shape[0]:]

    mlp = MultiLayerPerceptron()

    mlp.train(x[ind_train, :], new_y[ind_train, :],
              5, eta=0.01, alpha=0.7, MAX_EPOCH=1000)

    yhat = mlp.activate(x[ind_test, :])

    yhat = ytypes[np.argmax(yhat, 1)]

    acc = utils.accuracy(y[ind_test, :], yhat)

    print 'Acuracia = {:3.3f}'.format(100 * acc)
