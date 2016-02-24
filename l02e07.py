# 7) Para a base Car Evaluation (disponivel em http://archive.ics.uci.edu/ml/):
from classif_regres.decision_tree import DecisionTree

import numpy as np
import utils

x, y = utils.abrir_dados_car_str('./bases/car/car.data')
# a) Construa uma arvore de decisao usando a medida de Ganho de Informacao e um
# criterio de pre-poda (pode ser um valor minimo de Ganho de Informacao).
# Selecione aleatoriamente 75% dos dados para treinamento. Retorne a estrutura
# da arvore construida.
nclasses = np.union1d(y, y).size
n = len(y)
randind = np.arange(0, n)
np.random.shuffle(randind)
ind_train = randind[0:0.75 * n]
ind_test = randind[0.75 * n:n]

tree = DecisionTree(nclasses)
tree.train(x[ind_train, :], y[ind_train], IGMIN=0.01, NMIN=30)


g, pos = tree.gerar_grafo()
utils.draw_graph(g, pos)

# b) Use os restantes 25% dos dados para avaliacao. Retorne a acuracia obtida.

yhat = tree.classify(x[ind_test, :])

acc = utils.accuracy(y[ind_test], yhat)
print 'Acuracia encontrada: {:3.2f} %'.format(100 * acc)

# c) Tente obter as regras de decisao a partir da arvore construida.
