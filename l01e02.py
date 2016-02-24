import utils
import numpy as np
import matplotlib.pyplot as plt
import reduc_dim


# 2) Dada a base de dados CNAE-9_reduzido (em anexo):
cnae9_path = './bases/'

x, target = utils.abrir_dados_cnae9_reduzido(
    cnae9_path + 'CNAE-9_reduzido.txt')

# separa as amostras por classe
x_dict = {}
mu_dict = {}
var_dict = {}
cnae9_classes = list(set(target))
for cls in cnae9_classes:
    index = np.nonzero(target == cls)
    x_cls = x[index[0], :]
    # dicionario com as amostras separadas por classe.
    x_dict[cls] = np.copy(x_cls)

mu, var = utils.calcula_media_variancia(x)

# a) gere um grafico 2D com os dois componentes principais (uso de PCA) das
# amostras, identificando cada classe (a base possui 5 classes. O rotulo das
# amostras esta na primeira coluna. Essa coluna nao deve ser usada no PCA).

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
plt.title('CNAE9 - Principal components')
plt.xlabel('First principal component')
plt.ylabel('Second principal component')
plt.xlim([-5, 5])
plt.ylim([-5, 5])
plt.savefig(cnae9_path + 'results' + '/scatter')
plt.clf()

# b) gere um grafico 2D com os dois componentes principais (uso de PCA) das
# amostras, identificando cada classe (a base possui 5 classes). Para este
# grafico realize o branqueamento dos dados (isto e, apos a aplicacao do PCA
# garantir que a matriz de correlacao dos dados seja uma matriz identidade).
omega, vl, m = reduc_dim.pca_whitening(x, 2)  # dados na nova base

xhat_dict = {}  # dicionario que ira manter os novos dados por classe
colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w', ]  # cores para a plotagem
i = 0
plt.Figure
plt.hold(True)
h = []
legenda = []
for cls in x_dict.keys():
    x_cls = x_dict[cls]
    xhat_dict[cls] = np.real(
        np.dot(np.dot((x_cls - mu) / np.sqrt(var), m), vl))
    h.append(plt.scatter(xhat_dict[cls][:, 0], xhat_dict[cls]
                         [:, 1], s=70, c=colors[i], alpha=0.5))
    legenda.append(cls)
    i = i + 1
plt.legend(h, legenda)
plt.grid(True)
plt.title('CNAE9 - Principal components with whitening')
plt.xlabel('First principal component')
plt.ylabel('Second principal component')
plt.xlim([-5, 5])
plt.ylim([-5, 5])
plt.savefig(cnae9_path + 'results' + '/scatter_white')
plt.clf()


# c) Apos a visualizacao dos graficos, e possivel identificar ao menos uma
# classe que possa ser facilmente separada das outras classes? Seria possivel
# separar essa classe (sem grandes perdas) se fosse usada somente uma
# componente principal? Gere o grafico para explicar.

plt.Figure
colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w', ]  # cores para a plotagem
plt.hold(True)
h = []
legenda = []
i = 0
for cls in xhat_dict.keys():
    h.append(plt.scatter(
        xhat_dict[cls][:, 1], np.zeros((120, 1)), s=70, c=colors[i], alpha=0.5))
    legenda.append(cls)
    i = i + 1
plt.legend(h, legenda)
plt.grid(True)
plt.title('CNAE9 - Only one Principal components with whitening')
plt.xlabel('First principal component')
plt.xlim([-5, 5])
plt.ylim([-5, 5])
plt.savefig(cnae9_path + 'results' + '/one_component')
plt.clf()
