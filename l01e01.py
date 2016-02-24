import matplotlib.pyplot as plt
import numpy as np
import utils
import reduc_dim


iris_path = './bases/iris/'

x, target = utils.abrir_dados_iris(iris_path + 'iris.data')
attributes = ['sepal length in cm', 'sepal width in cm',
              'petal length in cm', 'petal width in cm']
n = x.shape[0]  # numero de amostras
'''
a) A media e variancia de cada um dos atributos
'''
print '\nLETRA A'
mu, var = utils.calcula_media_variancia(x)
s = ['Atributo', 'Media', 'Variancia']
for i in range(0, 4):
    s.append('{:}'.format(attributes[i]))
    s.append('{:2.2f}'.format(mu[i]))
    s.append('{:2.2f}'.format(var[i]))
s = np.reshape(s, (-1, 3))
print s


'''
b) A media e variancia de cada um dos atributos para cada uma das classes
'''
print '\nLETRA B'
x_dict = {}
mu_dict = {}
var_dict = {}
iris_classes = np.union1d(target, target)
for cls in iris_classes:
    index = np.nonzero(target == cls)
    x_cls = x[index[0], :]
    # dicionario com as amostras separadas por classe.
    x_dict[cls] = np.copy(x_cls)
    print '\n\nClasse - ' + str(cls)
    mu_dict[cls], var_dict[cls] = utils.calcula_media_variancia(x_cls)
    s = ['Atributo', 'Media', 'Variancia']
    for i in range(0, 4):
        s.append('{:}'.format(attributes[i]))
        s.append('{:2.2f}'.format(mu_dict[cls][i]))
        s.append('{:2.2f}'.format(var_dict[cls][i]))
    s = np.reshape(s, (-1, 3))
    print s

'''
c) O histograma com 16 bins de cada um dos atributos para cada uma das classes
(gere um unico grafico de histograma de 16 bins, ou seja, dividido em
16 segmentos, para cada atributo diferenciando as classes);
'''
print '\nLETRA C'
for att in range(0, 4):
    plt.Figure
    legenda = []
    for cls in x_dict.keys():
        n2, bins, patches = plt.hist(x_dict[cls][:, att], bins=16)
        legenda.append(cls)
    plt.title('Histogram')
    plt.ylabel('number of occurrences')
    plt.xlabel(attributes[att])
    plt.legend(legenda)
    plt.savefig(iris_path + 'results' + '/histogram' + str(att))
    plt.clf()

'''
d) Gere um grafico 2D com os dois componentes principais (uso de PCA)
das amostras, identificando cada classe. Pode usar a funcao eig do Matlab.
'''
print '\nLETRA D'
xhat, wl, vl = reduc_dim.pca(x, 2)  # dados na nova base

xhat_dict = {}  # dicionario que ira manter os novos dados por classe
colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w', ]  # cores para a plotagem
i = 0
plt.Figure
plt.hold(True)
h = []
legenda = []
for cls in x_dict.keys():
    x_cls = x_dict[cls]
    xhat_dict[cls] = np.real(np.dot((x_cls - mu) / np.sqrt(var), vl))
    h.append(plt.scatter(xhat_dict[cls][:, 0], xhat_dict[cls]
                         [:, 1], s=70, c=colors[i], alpha=0.5))
    legenda.append(cls)
    i = i + 1
plt.legend(h, legenda)
plt.grid(True)
plt.title('Principal components')
plt.xlabel('First principal component')
plt.ylabel('Second principal component')
plt.savefig(iris_path + 'results' + '/scatter')
plt.clf()
