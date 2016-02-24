# A partir da base de dados pista.txt, obtenha os valores das probabilidades
# condicionais e a priori de cada evento (ou seja, obter os valores necessarios
# para poder fazer as estimativas a partir da rede montada na questao a)). Na
# base de dados os valores 0 indicam que o evento nao aconteceu, enquanto 1
# aconteceu.

import utils
import numpy as np

x = utils.abrir_dados_pista('./bases/pista.txt')

att = ['C', 'F', 'E', 'M', 'A']

p_priori = np.zeros(x.shape[1])

for i in range(0, x.shape[1]):
    p_priori[i] = utils.simple_probability(x, i, np.equal, 1)
    print 'P({:s}=1)={:3.2f}%\n'.format(att[i], 100.0 * p_priori[i])


# matriz de emissao, indica que o no i eh pai do no j.
a = np.zeros((x.shape[1], x.shape[1]))

a[0, 2:] = 1
a[1, 2] = 1
a[1, 4] = 1
a[3, 2] = 1
a[3, 4] = 1
a[4, 2] = 1

# for i in range(0, x.shape[1]):
#     for j in range(0, x.shape[1]):
#         if a[i, j] == 1:
#             for t1 in range(0, 2):
#                 p = utils.conditional_probability(
#                     x, j, np.equal, 1, i, np.equal, t1)
#                 print 'P({:s}=1|{:s}={:d})={:3.2f}%'.format(att[j], att[i], t1, 100.0 * p)
#                 print 'P({:s}=0|{:s}={:d})={:3.2f}%'.format(att[j], att[i], t1, 100.0 * (1 - p))
#             print '\n'


c = []
for j in [2]:  # observando apenas os pais de E
    for i in range(0, x.shape[1]):
        if a[i, j] == 1:
            c.append([0, 1])  # para cada pai de E, adiciono um vetor em c.

comb = utils.cartesian(c)

for i in range(0, len(comb)):
    ind = np.logical_and(
        np.equal(x[:, 0], comb[i, 0]), np.equal(x[:, 1], comb[i, 1]))
    ind = np.logical_and(ind, np.equal(x[:, 3], comb[i, 2]))
    ind = np.logical_and(ind, np.equal(x[:, 4], comb[i, 3]))
    xl = x[ind, :]
    p = utils.simple_probability(xl, 2, np.equal, 1)
    if p == -1:
        print 'P(C={:d},F={:d},M={:d},A={:d})=0%\n'.format(comb[i, 0], comb[i, 1], comb[i, 2], comb[i, 3])
    else:
        print 'P(E=1|C={:d},F={:d},M={:d},A={:d})={:3.2f}%'.format(comb[i, 0], comb[i, 1], comb[i, 2], comb[i, 3], 100.0 * p)
        print 'P(E=0|C={:d},F={:d},M={:d},A={:d})={:3.2f}%\n'.format(comb[i, 0], comb[i, 1], comb[i, 2], comb[i, 3], 100.0 * (1 - p))
