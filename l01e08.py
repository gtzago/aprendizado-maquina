import utils
import matplotlib.pyplot as plt
import numpy as np
import classif_regres

centroides = np.array([[60, 150], [50, 50], [160, 40]])

x_train = centroides
# cria as classes dadas.
y = np.arange(0, centroides.shape[0])
x_test = []

# Cria os dados para preencher o espaco amostral.
n = 0
for x1 in range(1, 201, 2):
    for x2 in range(1, 201, 2):
        x_test = np.append(x_test, [x1, x2])
        n = n + 1
x_test = np.reshape(x_test, (n, -1))

# classifica os dados criados de acordo os exemplos passados (centroides).
yhat = classif_regres.rocchio(x_train, y, x_test)

# desenha o diagrama de voronoi.
cor = ['gray', 'b', 'r']
h = []
legenda = []
for cls in range(0, centroides.shape[0]):
    h.append(plt.scatter(x_test[np.where(yhat == cls), 0], x_test[
             np.where(yhat == cls), 1], s=70, c=cor[cls], alpha=0.5))
    legenda.append(cls)

plt.legend(h, legenda)
plt.grid(True)
plt.title('Voronoi - Distancia Euclidiana')
plt.xlabel('x1')
plt.ylabel('x2')

plt.plot(190, 130, 'go')

plt.savefig('./bases/results/E08_voronoi_rocchio')
plt.clf()


# classifica os dados criados de acordo os exemplos passados (centroides).
yhat = classif_regres.similaridade_cosseno(x_train, y, x_test)

# desenha o diagrama de voronoi.
cor = ['gray', 'b', 'r']
h = []
legenda = []
for cls in range(0, centroides.shape[0]):
    h.append(plt.scatter(x_test[np.where(yhat == cls), 0], x_test[
             np.where(yhat == cls), 1], s=70, c=cor[cls], alpha=0.5))
    legenda.append(cls)

plt.legend(h, legenda)
plt.grid(True)
plt.title('Voronoi - Similaridade Cosseno')
plt.xlabel('x1')
plt.ylabel('x2')

plt.plot(190, 130, 'go')

plt.savefig('./bases/results/E08_voronoi_similaridade_cosseno')
plt.clf()
