# 3) Use o perceptron para fazer a classificacao dos dados abaixo. Indique os
# pesos obtidos e mostre graficamente o plano de separacao entre as amostras.

# Entradas
# X1  X2      Saida
# 0   0       0
# 0   1       1
# 1   0       1
# 1   1       1
import matplotlib.pyplot as plt

import numpy as np
from artificial_neural_networks import Perceptron
x = np.zeros((4, 2))
x[1, 1] = 1
x[2, 0] = 1
x[3, 0] = 1
x[3, 1] = 1

y = np.ones((4, 1))
y[0, 0] = 0

perc = Perceptron()

perc.train(x, y, eta=0.1, MAX_EPOCH=1000)

yhat = perc.activate(x)

print 'w='
print perc.w

plt.Figure
plt.hold(True)
x = np.array([])
y = np.array([])
for i in np.arange(-1, 2, 0.1):
    for j in np.arange(-1, 2, 0.1):
        xi = np.array([i, j])
        x = np.append(x, xi)
        y = np.append(y, perc.activate(xi))
        if y[-1] > 0.5:
            plt.plot(i, j, 'ro')
        else:
            plt.plot(i, j, 'go')
plt.show()
