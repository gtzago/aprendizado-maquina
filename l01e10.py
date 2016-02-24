# 10) Usando as tecnicas de selecao de caracteristicas SFS e SBE sobre a base de
# dados Wine (disponivel em http://archive.ics.uci.edu/ml/), faca:
import utils
import numpy as np
import reduc_dim
import classif_regres

x, y = utils.abrir_dados_wine('./bases/wine/wine.data')


print '\nUtilizando SFS\n'
# a) Divida a base de dados em tres partes de forma estratificada. Selecione 5
# atributos usando uma parte da base de dados e valide os atributos sobre uma
# outra parte usando a metrica acuracia. Apos determinar os 5 atributos, obtenha
# a acuracia sobre a terceira parte, usando as duas partes como treinamento. Use
# o classificador Vizinho mais Proximo nesta tarefa. Quais foram os atributos
# selecionados?

# divide os dados.
k = 3
classes = np.union1d(y, y)
m = x.shape[1]
fold = {}
for f in range(0, k):
    fold[f] = np.array([])

for cls in classes:
    i = 0  # amostra
    f = 0  # fold em que a amostra sera guardada
    ind = np.where(y == cls)[0]

    while i < len(ind):
        if fold[f].size == 0:
            fold[f] = ind[i]
        else:
            fold[f] = np.vstack((fold[f], ind[i]))
        i = i + 1
        f = f + 1
        if f == k:
            f = 0


x_train = np.reshape(x[fold[0], :], (-1, m))
y_train = np.reshape(y[fold[0]], (-1, 1))
x_test = np.reshape(x[fold[1], :], (-1, m))
y_test = np.reshape(y[fold[1]], (-1, 1))


u = reduc_dim.sequential_forward_selection(
    x_train, y_train, x_test, y_test, utils.accuracy, 5, classif_regres.knn, 1)


x_train = np.reshape(x[fold[0], u], (-1, u.size))
y_train = np.reshape(y[fold[0]], (-1, 1))
x_train = np.vstack((x_train, np.reshape(x[fold[1], u], (-1, u.size))))
y_train = np.vstack((y_train, np.reshape(y[fold[1]], (-1, 1))))

x_test = np.reshape(x[fold[2], u], (-1, u.size))
y_test = np.reshape(y[fold[2]], (-1, 1))

yhat = classif_regres.knn(x_train, y_train, x_test, 1)
acc = utils.accuracy(y_test, yhat)
print '\nLETRA A\n'
print 'Atributos utilizados: ' + str(u)
print 'Acuracia = ' + str(acc)

# b) Realize o mesmo procedimento, mas agora selecionando 10 atributos;

x_train = np.reshape(x[fold[0], :], (-1, m))
y_train = np.reshape(y[fold[0]], (-1, 1))
x_test = np.reshape(x[fold[1], :], (-1, m))
y_test = np.reshape(y[fold[1]], (-1, 1))


u = reduc_dim.sequential_forward_selection(
    x_train, y_train, x_test, y_test, utils.accuracy, 10, classif_regres.knn, 1)


x_train = np.reshape(x[fold[0], u], (-1, u.size))
y_train = np.reshape(y[fold[0]], (-1, 1))
x_train = np.vstack((x_train, np.reshape(x[fold[1], u], (-1, u.size))))
y_train = np.vstack((y_train, np.reshape(y[fold[1]], (-1, 1))))

x_test = np.reshape(x[fold[2], u], (-1, u.size))
y_test = np.reshape(y[fold[2]], (-1, 1))

yhat = classif_regres.knn(x_train, y_train, x_test, 1)
acc = utils.accuracy(y_test, yhat)
print '\nLETRA B\n'
print 'Atributos utilizados: ' + str(u)
print 'Acuracia = ' + str(acc)


# c) Realize o mesmo procedimento de a) e b), mas agora selecionando os
# atributos usando duas partes e validando sobre as mesmas duas partes. Apos
# determinar os atributos, obtenha a acuracia sobre a terceira parte. A acuracia
# sobre a terceira parte foi melhor, igual ou pior do que as obtidas nas letras
# a) e b). Por que?

# divide os dados.
k = 3
classes = np.union1d(y, y)
m = x.shape[1]
fold = {}
for f in range(0, k):
    fold[f] = np.array([])

for cls in classes:
    i = 0  # amostra
    f = 0  # fold em que a amostra sera guardada
    ind = np.where(y == cls)[0]

    while i < len(ind):
        if fold[f].size == 0:
            fold[f] = ind[i]
        else:
            fold[f] = np.vstack((fold[f], ind[i]))
        i = i + 1
        f = f + 1
        if f == k:
            f = 0


x_train = np.reshape(x[fold[0], :], (-1, m))
y_train = np.reshape(y[fold[0]], (-1, 1))
x_test = np.reshape(x[fold[1], :], (-1, m))
y_test = np.reshape(y[fold[1]], (-1, 1))


u = reduc_dim.sequential_forward_selection(
    x_train, y_train, x_test, y_test, utils.accuracy, 5, classif_regres.knn, 1)


x_train = np.reshape(x[fold[0], u], (-1, u.size))
y_train = np.reshape(y[fold[0]], (-1, 1))


x_test = np.reshape(x[fold[1], u], (-1, u.size))
y_test = np.reshape(y[fold[1]], (-1, 1))

yhat = classif_regres.knn(x_train, y_train, x_test, 1)
acc = utils.accuracy(y_test, yhat)
print '\nLETRA C1\n'
print 'Atributos utilizados: ' + str(u)
print 'Acuracia = ' + str(acc)


################################################
x_train = np.reshape(x[fold[0], :], (-1, m))
y_train = np.reshape(y[fold[0]], (-1, 1))
x_test = np.reshape(x[fold[1], :], (-1, m))
y_test = np.reshape(y[fold[1]], (-1, 1))


u = reduc_dim.sequential_forward_selection(
    x_train, y_train, x_test, y_test, utils.accuracy, 10, classif_regres.knn, 1)


x_train = np.reshape(x[fold[0], u], (-1, u.size))
y_train = np.reshape(y[fold[0]], (-1, 1))


x_test = np.reshape(x[fold[1], u], (-1, u.size))
y_test = np.reshape(y[fold[1]], (-1, 1))

yhat = classif_regres.knn(x_train, y_train, x_test, 1)
acc = utils.accuracy(y_test, yhat)
print '\nLETRA C2\n'
print 'Atributos utilizados: ' + str(u)
print 'Acuracia = ' + str(acc)


print '\nQuando se treina e valida sobre o mesmo conjunto, escohem-se os atributos que maximizam a acuracia para aquele determinado conjunto, que nao necessariamente vale para o todo (overfitting)'




print '\nUtilizando SBE\n'
# a) Divida a base de dados em tres partes de forma estratificada. Selecione 5
# atributos usando uma parte da base de dados e valide os atributos sobre uma
# outra parte usando a metrica acuracia. Apos determinar os 5 atributos, obtenha
# a acuracia sobre a terceira parte, usando as duas partes como treinamento. Use
# o classificador Vizinho mais Proximo nesta tarefa. Quais foram os atributos
# selecionados?

# divide os dados.
k = 3
classes = np.union1d(y, y)
m = x.shape[1]
fold = {}
for f in range(0, k):
    fold[f] = np.array([])

for cls in classes:
    i = 0  # amostra
    f = 0  # fold em que a amostra sera guardada
    ind = np.where(y == cls)[0]

    while i < len(ind):
        if fold[f].size == 0:
            fold[f] = ind[i]
        else:
            fold[f] = np.vstack((fold[f], ind[i]))
        i = i + 1
        f = f + 1
        if f == k:
            f = 0


x_train = np.reshape(x[fold[0], :], (-1, m))
y_train = np.reshape(y[fold[0]], (-1, 1))
x_test = np.reshape(x[fold[1], :], (-1, m))
y_test = np.reshape(y[fold[1]], (-1, 1))


u = reduc_dim.sequential_backward_elimination(
    x_train, y_train, x_test, y_test, utils.accuracy, 5, classif_regres.knn, 1)


x_train = np.reshape(x[fold[0], u], (-1, u.size))
y_train = np.reshape(y[fold[0]], (-1, 1))
x_train = np.vstack((x_train, np.reshape(x[fold[1], u], (-1, u.size))))
y_train = np.vstack((y_train, np.reshape(y[fold[1]], (-1, 1))))

x_test = np.reshape(x[fold[2], u], (-1, u.size))
y_test = np.reshape(y[fold[2]], (-1, 1))

yhat = classif_regres.knn(x_train, y_train, x_test, 1)
acc = utils.accuracy(y_test, yhat)
print '\nLETRA A\n'
print 'Atributos utilizados: ' + str(u)
print 'Acuracia = ' + str(acc)

# b) Realize o mesmo procedimento, mas agora selecionando 10 atributos;

x_train = np.reshape(x[fold[0], :], (-1, m))
y_train = np.reshape(y[fold[0]], (-1, 1))
x_test = np.reshape(x[fold[1], :], (-1, m))
y_test = np.reshape(y[fold[1]], (-1, 1))


u = reduc_dim.sequential_backward_elimination(
    x_train, y_train, x_test, y_test, utils.accuracy, 10, classif_regres.knn, 1)


x_train = np.reshape(x[fold[0], u], (-1, u.size))
y_train = np.reshape(y[fold[0]], (-1, 1))
x_train = np.vstack((x_train, np.reshape(x[fold[1], u], (-1, u.size))))
y_train = np.vstack((y_train, np.reshape(y[fold[1]], (-1, 1))))

x_test = np.reshape(x[fold[2], u], (-1, u.size))
y_test = np.reshape(y[fold[2]], (-1, 1))

yhat = classif_regres.knn(x_train, y_train, x_test, 1)
acc = utils.accuracy(y_test, yhat)
print '\nLETRA B\n'
print 'Atributos utilizados: ' + str(u)
print 'Acuracia = ' + str(acc)


# c) Realize o mesmo procedimento de a) e b), mas agora selecionando os
# atributos usando duas partes e validando sobre as mesmas duas partes. Apos
# determinar os atributos, obtenha a acuracia sobre a terceira parte. A acuracia
# sobre a terceira parte foi melhor, igual ou pior do que as obtidas nas letras
# a) e b). Por que?

# divide os dados.
k = 3
classes = np.union1d(y, y)
m = x.shape[1]
fold = {}
for f in range(0, k):
    fold[f] = np.array([])

for cls in classes:
    i = 0  # amostra
    f = 0  # fold em que a amostra sera guardada
    ind = np.where(y == cls)[0]

    while i < len(ind):
        if fold[f].size == 0:
            fold[f] = ind[i]
        else:
            fold[f] = np.vstack((fold[f], ind[i]))
        i = i + 1
        f = f + 1
        if f == k:
            f = 0


x_train = np.reshape(x[fold[0], :], (-1, m))
y_train = np.reshape(y[fold[0]], (-1, 1))
x_test = np.reshape(x[fold[1], :], (-1, m))
y_test = np.reshape(y[fold[1]], (-1, 1))


u = reduc_dim.sequential_backward_elimination(
    x_train, y_train, x_test, y_test, utils.accuracy, 5, classif_regres.knn, 1)


x_train = np.reshape(x[fold[0], u], (-1, u.size))
y_train = np.reshape(y[fold[0]], (-1, 1))


x_test = np.reshape(x[fold[1], u], (-1, u.size))
y_test = np.reshape(y[fold[1]], (-1, 1))

yhat = classif_regres.knn(x_train, y_train, x_test, 1)
acc = utils.accuracy(y_test, yhat)
print '\nLETRA C1\n'
print 'Atributos utilizados: ' + str(u)
print 'Acuracia = ' + str(acc)


################################################
x_train = np.reshape(x[fold[0], :], (-1, m))
y_train = np.reshape(y[fold[0]], (-1, 1))
x_test = np.reshape(x[fold[1], :], (-1, m))
y_test = np.reshape(y[fold[1]], (-1, 1))


u = reduc_dim.sequential_backward_elimination(
    x_train, y_train, x_test, y_test, utils.accuracy, 10, classif_regres.knn, 1)


x_train = np.reshape(x[fold[0], u], (-1, u.size))
y_train = np.reshape(y[fold[0]], (-1, 1))


x_test = np.reshape(x[fold[1], u], (-1, u.size))
y_test = np.reshape(y[fold[1]], (-1, 1))

yhat = classif_regres.knn(x_train, y_train, x_test, 1)
acc = utils.accuracy(y_test, yhat)
print '\nLETRA C2\n'
print 'Atributos utilizados: ' + str(u)
print 'Acuracia = ' + str(acc)


print '\nQuando se treina e valida sobre o mesmo conjunto, escohem-se os atributos que maximizam a acuracia para aquele determinado conjunto, que nao necessariamente vale para o todo (overfitting)'
