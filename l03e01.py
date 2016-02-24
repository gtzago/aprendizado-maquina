# 1) Para as bases de dados Spiral e Jain (disponiveis em
# http://cs.joensuu.fi/sipu/datasets/), agrupe os dados em 3 e 2 grupos,
# respectivamente, usando kmeans e clusterizacao hierarquica. Avalie o resultado
# com a metrica de acuracia com o seguinte procedimento: para cada cluster
# verifique qual foi a classe predominante, amostras pertencentes a outras
# classes estao no grupo errado. Faca os experimentos com a distancia
# Euclidiana. Gere graficos com os grupos formados pelo kmeans e clusterizacao
# hierarquica. Comente os resultados. Lembre-se de nao usar o atributo da classe
# para agrupar os dados.
import matplotlib.pyplot as plt

from clustering import KMeans
from clustering import Hierarchical
import utils


print 'Base Spiral\n'
x, y = utils.abrir_dados_spiral('./bases/spiral.csv')
print 'Kmeans'
kmeans = KMeans(x)
for k in [2, 3]:
    kmeans.train(k)
    print 'k = {:d}. Acuracia = {:1.3f} '.format(k, 100 * kmeans.calculate_accuracy(y))
    fig = kmeans.scatter_plot()
    plt.savefig('./bases/l03/spiral_kmeans_k_{:d}'.format(k))
    plt.clf()

print 'Hierarchical'
hier = Hierarchical(x)
for k in [2, 3]:
    hier.agnes(k)
    print 'k = {:d}. Acuracia = {:1.3f} '.format(k, 100 * hier.calculate_accuracy(y))
    fig = hier.scatter_plot()
    plt.savefig('./bases/l03/spiral_hier_k_{:d}'.format(k))
    plt.clf()


print '\nBase Jain'
x, y = utils.abrir_dados_jain('./bases/jain.csv')

print 'Kmeans'
kmeans = KMeans(x)
for k in [2, 3]:
    kmeans.train(k)
    print 'k = {:d}. Acuracia = {:1.3f} '.format(k, 100 * kmeans.calculate_accuracy(y))
    fig = kmeans.scatter_plot()
    plt.savefig('./bases/l03/jain_kmeans_k_{:d}'.format(k))
    plt.clf()

print 'Hierarchical'
hier = Hierarchical(x)
for k in [2, 3]:
    hier.agnes(k)
    print 'k = {:d}. Acuracia = {:1.3f} '.format(k, 100 * hier.calculate_accuracy(y))
    fig = hier.scatter_plot()
    plt.savefig('./bases/l03/jain_hier_k_{:d}'.format(k))
    plt.clf()
