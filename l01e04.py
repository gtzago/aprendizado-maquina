import utils
import numpy as np
import matplotlib.pyplot as plt
import classif_regres
import reduc_dim
# 4) A base de dados Nebulosa (disponibilizada em anexo) esta contaminada com
# ruidos, redundancias, dados incompletos, inconsistencias e outliers. Para esta
# base:
# A base de dados nebulosa e composta por 8 atributos:
# 1 - Numero de identificacao do usuario (um por usuario)
# 2 - Nome do usuario representado em numero (cada usuario tem um nome)
# 3 - Porte fisico da pessoa na escala entre 1 e 3, mas contaminado com ruido
# 4 - Nivel educacional da pessoa na escala entre 1 e 4, mas contaminado com ruido
# 5 - Nivel socio-economico da pessoa na escala entre 1 e 4, mas contaminado com ruido
# 6 - epoca de nascimento da pessoa na escala entre 1 e 4, mas contaminado com ruido
# 7 - Idade da pessoa na escala entre 1 e 4, mas contaminado com ruido
# 8 - Classificacao da pessoa na escala entre 1 e 3

attributes = ['Porte fisico', 'Nivel educacional',
              'nivel socio-econ', 'epoca de nascimento', 'idade']
x_train, target_train, x_test, target_test = utils.abrir_dados_nebulosa(
    './bases')

limites = np.array([[1, 3], [1, 4], [1, 4], [1, 4], [1, 4], [1, 3]])

# for i in range(0, x_train.shape[0]):
#     plt.boxplot(x_train[i, :])
#     plt.title(attributes[i])
#     plt.show()


# a) Obtenha os resultados da classificacao (metrica acuracia) usando a
# tecnica do vizinho mais proximo (NN) e Rocchio. Utilize a distancia Euclidiana e
# a base de dados crua, sem pre-processamento. Use o conjunto de 143 amostras para
# treino e o de 28 amostras para teste. Faca um tratamento para os dados
# incompletos.

# substituirei os dados incompletos pelas medianas daqueles atributos.
for i in range(0, x_train.shape[0]):
    for j in range(0, x_train.shape[1]):
        if x_train[i, j] == -1:
            x_train[i, j] = np.median(x_train[:, j])

# retiro os exemplos cuja classe e indefinida
ind = np.where(np.array(target_train) == '?')[0]
target_train = np.delete(target_train, ind, 0)
x_train = np.delete(x_train, ind, 0)

ind = np.where(np.array(target_test) == '?')[0]
target_test = np.delete(target_test, ind, 0)
x_test = np.delete(x_test, ind, 0)

cls_knn = classif_regres.knn(x_train, target_train, x_test, 5)

cls_rocchio = classif_regres.rocchio(x_train, target_train, x_test)

print 'Sem pre-processamento'

print 'Acuracia do knn (k=5) = ' + str(100 * utils.accuracy(target_test, cls_knn)) + '%'

print 'Acuracia do rocchio = ' + str(100 * utils.accuracy(target_test, cls_rocchio)) + '%'

# b) Realize um pre-processamento sobre os dados de forma a reduzir
# os ruidos, as redundancias, inconsistencias, outliers e a interferencia dos
# dados incompletos. Obtenha os resultados da classificacao usando a tecnica do
# vizinho mais proximo (NN) e Rocchio usando a distancia Euclidiana e a mesma
# divisao dos dados.

x_train, target_train, x_test, target_test = utils.abrir_dados_nebulosa(
    './bases')

x_train = x_train[:, 2:]
x_test = x_test[:, 2:]

# retiro os exemplos cuja classe e indefinida
ind = np.where(np.array(target_train) == '?')[0]
target_train = np.delete(target_train, ind, 0)
x_train = np.delete(x_train, ind, 0)

ind = np.where(np.array(target_test) == '?')[0]
target_test = np.delete(target_test, ind, 0)
x_test = np.delete(x_test, ind, 0)

# retiro os exemplos com valores indefinidos ou outliers
ind = np.where(np.array(x_train) == -1)[0]
target_train = np.delete(target_train, ind, 0)
x_train = np.delete(x_train, ind, 0)

ind = np.where(np.array(x_test) == -1)[0]
target_test = np.delete(target_test, ind, 0)
x_test = np.delete(x_test, ind, 0)

ind = np.where(np.array(x_train) < 0)[0]
target_train = np.delete(target_train, ind, 0)
x_train = np.delete(x_train, ind, 0)

ind = np.where(np.array(x_test) < 0)[0]
target_test = np.delete(target_test, ind, 0)
x_test = np.delete(x_test, ind, 0)

ind = np.where(np.array(x_train) > 5)[0]
target_train = np.delete(target_train, ind, 0)
x_train = np.delete(x_train, ind, 0)

ind = np.where(np.array(x_test) > 5)[0]
target_test = np.delete(target_test, ind, 0)
x_test = np.delete(x_test, ind, 0)
# boxplot
# for att in range(0, x_train.shape[1]):
#     plt.hold(False)
#     plt.boxplot(x_train[:, att])
#     plt.grid(True)
#
#     plt.title(str(att))
#     plt.savefig('./' + '/academiaboxplot'+str(att))


# os elementos das colunas 4 e 5 sao redundantes e podem ser eliminados
# (epoca do nascimento e idade) -- melhora o knn, nao muda o rocchio
# for ex in range(0, x_train.shape[0]):
#     if x_train[ex, 3] < limites[3][0] or x_train[ex, 3] > limites[3][1]:
#         x_train[ex, 3] = x_train[ex, 4]
# x_train = np.delete(x_train, 4, 1)
#
# for ex in range(0, x_test.shape[0]):
#     if x_test[ex, 3] < limites[3][0] or x_test[ex, 3] > limites[3][1]:
#         x_test[ex, 3] = x_test[ex, 4]
# x_test = np.delete(x_test, 4, 1)

# substituirei os dados incompletos pelas medianas daqueles atributos na
# classe.
# classes = list(set(target_train))
# for cls in classes:
#     indexes = np.where(target_train == cls)
#     for ex in indexes[0]:
#         for att in range(0, x_train.shape[1]):
#             if x_train[ex, att] == - 1:
#                 x_train[ex, att] = np.median(x_train[indexes, att])
# # sem olhar a classe no conjunto de teste
# for i in range(0, x_test.shape[0]):
#     for j in range(0, x_test.shape[1]):
#         if x_test[i, j] == -1:
#             x_test[i, j] = np.median(x_test[:, j])
#
# # normaliza dados de acordo com limites dados.
# for att in range(0, x_train.shape[1]):
#     a = 2.0 / (limites[att, 1] - limites[att, 0])
#     b = -1.0 * ((limites[att, 1] + limites[att, 0]) /
#                 (limites[att, 1] - limites[att, 0]))
#     x_train[:, att] = a * x_train[:, att] + b
#
# for att in range(0, x_test.shape[1]):
#     a = 2.0 / (limites[att, 1] - limites[att, 0])
#     b = -1.0 * ((limites[att, 1] + limites[att, 0]) /
#                 (limites[att, 1] - limites[att, 0]))
#     x_test[:, att] = a * x_test[:, att] + b


# plt.scatter(x_train[np.where(target_train == '1'), 0], x_train[
#             np.where(target_train == '1'), 1], s=70, c='b', alpha=0.5)
# plt.hold(True)
# plt.scatter(x_train[np.where(target_train == '2'), 0], x_train[
#             np.where(target_train == '2'), 1], s=70, c='g', alpha=0.5)
# plt.scatter(x_train[np.where(target_train == '3'), 0], x_train[
#             np.where(target_train == '3'), 1], s=70, c='r', alpha=0.5)
# plt.show()

# substituirei os valores outliers pelas medianas dos atributos da classe.
# classes = list(set(target_train))
# for cls in classes:
#     indexes = np.where(target_train == cls)
#     for ex in indexes[0]:
#         for att in range(0, x_train.shape[1]):
#             if x_train[ex, att] < - 2 or x_train[ex, att] > 2:
#                 x_train[ex, att] = np.median(x_train[indexes, att])

# saturo os valores dentro dos limites dados.
# for att in range(0, x_train.shape[1]):
#     ind = np.where(
#         x_train[:, att] < limites[att][0])[0]
#     x_train[ind, att] = limites[att][0]
#
#     ind = np.where(
#         x_train[:, att] > limites[att][1])[0]
#     x_train[ind, att] = limites[att][1]


# plt.scatter(x_train[np.where(target_train == '1'), 0], x_train[
#             np.where(target_train == '1'), 1], s=70, c='b', alpha=0.5)
# plt.hold(True)
# plt.scatter(x_train[np.where(target_train == '2'), 0], x_train[
#             np.where(target_train == '2'), 1], s=70, c='g', alpha=0.5)
# plt.scatter(x_train[np.where(target_train == '3'), 0], x_train[
#             np.where(target_train == '3'), 1], s=70, c='r', alpha=0.5)
# plt.show()

# utilizando PCA, so piora.
mu, var = utils.calcula_media_variancia(x_train)
x_trainhat, wl, vl = reduc_dim.pca(x_train, k=4)
x_testhat = np.real(np.dot((x_test - mu) / np.sqrt(var), vl))
cls_knn = classif_regres.knn(x_trainhat, target_train, x_testhat, 5)
cls_rocchio = classif_regres.rocchio(x_trainhat, target_train, x_testhat)

cls_knn = classif_regres.knn(x_train, target_train, x_test, 5)
cls_rocchio = classif_regres.rocchio(x_train, target_train, x_test)

print 'Com pre-processamento'

print 'Acuracia do knn (k=5) = ' + str(100 * utils.accuracy(target_test, cls_knn)) + '%'

print 'Acuracia do rocchio = ' + str(100 * utils.accuracy(target_test, cls_rocchio)) + '%'

# Comente os resultados obtidos.
