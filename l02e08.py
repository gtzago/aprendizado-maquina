# 8) Para a base Servo (disponivel em http://archive.ics.uci.edu/ml/):
import utils
import numpy as np
from classif_regres.regression_tree import RegressionTree
import matplotlib.pyplot as plt


x, y = utils.abrir_dados_servo('./bases/servo/servo.data')
# a) Construa uma arvore de regressao usando a medida de reducao de desvio
# padrao e um criterio de pre-poda (pode ser um valor minimo de reducao de
# desvio padrao). Selecione aleatoriamente 75% dos dados para treinamento.
# Retorne a estrutura da arvore construida.
nclasses = np.union1d(y, y).size
n = len(y)
randind = np.arange(0, n)
np.random.shuffle(randind)
ind_train = randind[0:0.75 * n]
ind_test = randind[0.75 * n:n]

tree = RegressionTree(nclasses)
tree.train(x[ind_train, :], y[ind_train], SDRMIN=0.1, NMIN=3)

g, pos = tree.gerar_grafo()
utils.draw_graph(g, pos)

# b) Use os restantes 25% dos dados para avaliacao. Retorne as medidas MAPE e
# RMSE.

yhat = tree.estimate(x[ind_test, :])

rmse = utils.rmse(y[ind_test], yhat)
mape = utils.mape(y[ind_test], yhat)
print 'RMSE encontrado: {:3.2f}\nMAPE encontrado: {:3.2f}'.format(rmse,mape)

plt.plot(y[ind_test])
plt.hold(True)
plt.plot(yhat)
plt.legend(['real','estimado'])
plt.show()

# c) Tente obter as regras de decisao a partir da arvore construida.