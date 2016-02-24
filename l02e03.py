# 3) Aplique o Naive Bayes sobre a base de dados Balance Scale (disponivel em
# http://archive.ics.uci.edu/ml/) utilizando o procedimento de Hold-Out dez
# vezes, na proporcao de 75% de amostras de treino e 25% de teste. Obtenha a
# acuracia media e o desvio padrao da acuracia. Realize os experimentos:

import utils
import numpy as np
import classif_regres

x, y = utils.abrir_dados_balance_scale(
    './bases/balance-scale/balance-scale.data')

# a) Considerando uma distribuicao Gaussiana dos atributos;
k = 10
acc = np.zeros(k)
for i in range(0, k):
    ind_rand = np.arange(0, y.size)
    np.random.shuffle(ind_rand)  # indices em ordem aleatoria.
    ind_train = ind_rand[0:.75 * y.size]
    ind_test = ind_rand[.75 * y.size:]
    yhat = classif_regres.naive_bayes(
        x[ind_train, :], y[ind_train], x[ind_test, :], probmod='gauss')
    acc[i] = utils.accuracy(y[ind_test], yhat)
print 'LETRA A - Utilizando probabilidade gaussiana.'
print 'A acuracia media eh de {:3.3f} % e seu d.p. eh de {:3.3f} %\n'.format(
    100 * np.mean(acc), 100 * np.std(acc))
# b) Discretizando os valores (em 5 partes cada atributo);

k = 10
acc = np.zeros(k)
for i in range(0, k):
    ind_rand = np.arange(0, y.size)
    np.random.shuffle(ind_rand)  # indices em ordem aleatoria.
    ind_train = ind_rand[0:.75 * y.size]
    ind_test = ind_rand[.75 * y.size:]
    yhat = classif_regres.naive_bayes(
        x[ind_train, :], y[ind_train], x[ind_test, :], probmod='freq')
    acc[i] = utils.accuracy(y[ind_test], yhat)
print 'LETRA B - Utilizando probabilidade baseada em frequencia.'
print 'A acuracia media eh de {:3.3f} % e seu d.p. eh de {:3.3f} %\n'.format(
    100 * np.mean(acc), 100 * np.std(acc))


# c) Discretize os valore da mesma forma que em b) usando a suavizacao de
# Laplace.

k = 10
acc = np.zeros(k)
for i in range(0, k):
    ind_rand = np.arange(0, y.size)
    np.random.shuffle(ind_rand)  # indices em ordem aleatoria.
    ind_train = ind_rand[0:.75 * y.size]
    ind_test = ind_rand[.75 * y.size:]
    yhat = classif_regres.naive_bayes(
        x[ind_train, :], y[ind_train], x[ind_test, :], laplace_smooth=True)
    acc[i] = utils.accuracy(y[ind_test], yhat)
print 'LETRA C - Utilizando probabilidade baseada em frequencia com suavizacao de laplace.'    
print 'A acuracia media eh de {:3.3f} % e seu d.p. eh de {:3.3f} %\n'.format(
    100 * np.mean(acc), 100 * np.std(acc))