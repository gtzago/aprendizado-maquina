import utils
import matplotlib.pyplot as plt
import numpy as np
import classif_regres
# 7) Para a base de dados Polinomio (disponibilizada em anexo), faca:


x, y = utils.abrir_dados_polinomio('./bases/Polinomio.txt')
n = x.shape[0]
plt.plot(x, y)
plt.grid(True)
plt.title('Polinomio original')
plt.ylabel('y')
plt.xlabel('x')
plt.hold(True)
plt.plot(x, y, 'bo')
plt.savefig('./bases/results/polinomio_original')
plt.clf()

# a) Divida a base de dados ao meio. Use a primeira parte (as primeiras amostras)
# para estimar o modelo que melhor se ajusta aos dados. Informe os parametros do
# modelo encontrado.

what = classif_regres.linear_regression(
    x[0:(n / 2), :], y[0:(n / 2)], degree=2)
yhat = classif_regres.linear_regression_estimate_output(
    x[n / 2:, :], what, degree=2)

plt.plot(x, y)
plt.grid(True)
plt.title('Polinomio original')
plt.ylabel('y')
plt.xlabel('x')
plt.hold(True)
plt.plot(x[n / 2:, :], yhat)
plt.savefig('./bases/results/polinomio_estimado')
plt.clf()

print 'Letra A'
print 'Polinomio encontrado: '
print 'y = {:3.3f} + {:3.3f}x {: 3.3f}x^2\n'.format(what[0][0], what[1][0], what[2][0])

# b) Obtenha o RMSE e MAPE do modelo obtido sobre os dados da segunda metade dos
# dados;
print 'Letra B'
rmse = utils.rmse(y[n / 2:, :], yhat)
print 'RMSE = ' + str(rmse) + '\n'
mape = utils.mape(y[n / 2:, :], yhat)
print 'MAPE = ' + str(mape) + '\n'

# c) Estimar o modelo que melhor se ajusta aos dados usando todos os dados.
# Informe os parametros do modelo encontrado. Use os fatores de determinacao de
# complexidade do modelo para auxiliar a encontrar o modelo. Obtenha o RMSE e MAPE
# do modelo obtido sobre os dados.
print 'Letra C'
MAXDEGREE = 5
plt.Figure
plt.hold(True)
plt.grid(True)
plt.plot(x, y)
plt.title('Ajuste Polinomial')
plt.ylabel('y')
plt.xlabel('x')

legenda = ['real']
for d in range(1, MAXDEGREE + 1):
    what = classif_regres.linear_regression(x, y, degree=d)
    yhat = classif_regres.linear_regression_estimate_output(x, what, degree=d)
    rmse = utils.rmse(y, yhat)
    mape = utils.mape(y, yhat)
    r2 = utils.coef_determinacao_ajustado(y, yhat, degree=d)
    plt.plot(x, yhat)
    legenda.append('grau: ' + str(d))
    print '\nEstimacao utilizando polinomio de grau: ' + str(d)
    print 'Parametros: '
    print what
    print '\n'
    print 'RMSE = ' + str(rmse)
    print 'MAPE = ' + str(mape)
    print 'Coeficiente de determinacao ajustado = ' + str(r2)
plt.legend(legenda, loc=2)
plt.savefig('./bases/results/polinomio_estimado_LETRA_C')
plt.clf()


# d) Utilize o metodo de Ransac sobre todos os dados para remover os outliers e
# obter o modelo. Informe os parametros do modelo encontrado. Obtenha o RMSE e
# MAPE do modelo obtido sobre os dados.
print 'LETRA D'
plt.plot(x, y)
plt.title('Ajuste Polinomial')
plt.ylabel('y')
plt.xlabel('x')
plt.hold(True)
plt.grid(True)

legenda = ['real']

d = 3

what = classif_regres.linear_regression(x, y, degree=d)
yhat = classif_regres.linear_regression_estimate_output(x, what, degree=d)
plt.plot(x, yhat)
rmse = utils.rmse(y, yhat)
mape = utils.mape(y, yhat)
r2 = utils.coef_determinacao_ajustado(y, yhat, degree=d)
legenda.append('No RANSAC - MAPE = ' + str(mape) + ' - RMSE = ' + str(rmse))
print '\nEstimacao utilizando polinomio de grau: ' + str(d)
print '\n'
print 'RMSE = ' + str(rmse)
print 'MAPE = ' + str(mape)
print 'Coeficiente de determinacao ajustado = ' + str(r2)


s = 10
TAL = 2
what = classif_regres.ransac(x, y, degree=d, s=s, TAL=TAL)
yhat = classif_regres.linear_regression_estimate_output(x, what, degree=d)
plt.plot(x, yhat)

rmse = utils.rmse(y, yhat)
mape = utils.mape(y, yhat)
r2 = utils.coef_determinacao_ajustado(y, yhat, degree=d)
legenda.append('RANSAC - ' + ' s = ' + str(s) + ' - tal = ' +
               str(TAL))
print '\nEstimacao utilizando polinomio de grau: ' + str(d)
print '\n'
print 'RMSE = ' + str(rmse)
print 'MAPE = ' + str(mape)
print 'Coeficiente de determinacao ajustado = ' + str(r2)

plt.legend(legenda, loc=2)

plt.savefig('./bases/results/polinomio_estimado_LETRA_D')
plt.clf()
