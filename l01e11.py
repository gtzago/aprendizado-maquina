# 11) Usando os metodos apresentados em aula para estimar o numero de amostras
# de treinamento e de teste, determine para a base de dados Wine (disponivel em
# http://archive.ics.uci.edu/ml/):
import utils
import numpy as np
import reduc_dim
import classif_regres
import matplotlib.pyplot as plt


x, y = utils.abrir_dados_wine('./bases/wine/wine.data')

n = x.shape[0]
m = x.shape[1]
# a) O numero de amostras de treinamento tal que a diferenca entre o erro
# assintotico e a taxa de erro seja menor ou igual a 2%. Use as estimativas do
# Vizinho Mais Proximo e a metrica de acuracia.
print '\nLETRA A'
acc, contingencia, pr = classif_regres.kfold_cross_validation(
    x, y, n, utils.accuracy, classif_regres.knn, 1)
pc = 1 - acc

yhat = classif_regres.knn(x, y, x, 1)
acc = utils.accuracy(y, yhat)
pr = 1 - acc

pinf = (pc + pr) / 2

# pinf = 0.1151

print 'Taxa de erro assimptotica = ' + str(pinf)

print 'Diferenca entre erro assimptotico e taxa de erro usando todas as amostras eh de {:3.3f}%'.format(100*(pc-pr/(2*pinf)))

# deltaf = np.zeros(n)
# rpt = 10
# for k in range(0, rpt):
#     delta = np.array([])
#     for tam in range(1, n + 1):
#         randind = np.random.randint(0, n, tam)
#         x_train = np.reshape(x[randind, :], (-1, m))
#         y_train = np.reshape(y[randind], (-1, 1))
# #         x_test = np.delete(x, randind, 0)
# #         y_test = np.delete(y, randind, 0)
#         x_test = x
#         y_test = y
#         yhat = classif_regres.knn(x_train, y_train, x_test, 1)
#         acc = utils.accuracy(y_test, yhat)
#         pn = 1 - acc
#     #     print '\ntamanho = ' + str(tam)
#     #     print 'Pn = ' + str(pn)
#     #     print 'delta = ' + str(pn - pinf)
#         delta = np.append(delta, pn - pinf)
#     deltaf = deltaf + delta
# deltaf = 100.0 * deltaf / rpt
# 
# ntreino = np.where(deltaf < 2)[0][0]
# 
# print 'Numero de amostras para treino = ' + str(ntreino)

# plt.plot(deltaf)
# plt.grid(True)
# plt.show()


# b) O numero de amostras de teste, tal que a taxa de erro real seja P < 0,1 com
# um desvio aceitavel de 20%. Use as estimativas do Vizinho Mais Proximo e a
# metrica de acuracia.

p = 0.1
k = 20.0
ntest = 4 * (1 - p) / (p * np.power(k/100, 2))

print 'Numero de amostras necessarias para o teste: ' + str(ntest)
