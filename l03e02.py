# 2) Para a base de dados dados.txt em anexo, use RDE para estimar os outliers e
# a transicao nos dados analisados. Mostre os resultados graficamente e comente
# os resultados.
import matplotlib.pyplot as plt
import utils
from rde import RecursiveDensityEstimation
import numpy as np
x = utils.abrir_dados_dados('./bases/dados.txt')

rde = RecursiveDensityEstimation()
outlier = np.array([])
density = np.array([])
dp = np.array([])
mean_density = np.array([])
for k in range(0, x.size):
    outlier = np.append(outlier, rde.insert(x[k]))
    density = np.append(density, rde.density)
    mean_density = np.append(mean_density, rde.mean_density)
    dp = np.append(dp, np.sqrt(rde.var))

plt.plot(x)
plt.hold(True)
ind = np.where(outlier)[0]
plt.plot(ind, x[ind], 'ro')
plt.grid(True)
plt.plot(density)
plt.plot(mean_density + rde.thr * dp)
plt.plot(mean_density - rde.thr * dp)
plt.legend(['data','outliers','density','higher threshold','lower threshold'])
plt.savefig('./bases/l03/rde')