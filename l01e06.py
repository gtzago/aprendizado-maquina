import utils
import matplotlib.pyplot as plt
import numpy as np
import classif_regres


# 6) Para a base de dados Runner (disponibilizada em anexo) obtenha:

x, target = utils.abrir_dados_runner('./bases/Runner_num.txt')
plt.plot(x, target)
plt.grid(True)
plt.title('100 m olympics time')
plt.ylabel('time (s)')
plt.xlabel('year')
plt.hold(True)
plt.plot(x, target, 'bo')
plt.savefig('./bases/results/runner')
plt.clf()
# a) A equacao linear que se ajusta aos dados e a RMSE;

what = classif_regres.linear_regression(x, target)
yhat = classif_regres.linear_regression_estimate_output(x, what)

plt.plot(x, target)
plt.grid(True)
plt.title('100 m olympics time')
plt.ylabel('time (s)')
plt.xlabel('year')
plt.hold(True)
plt.plot(x, yhat)
plt.legend(['real values', 'estimated values'])
plt.plot(x, target, 'bo')
plt.plot(x, yhat, 'go')
plt.savefig('./bases/results/runner_estimated')
plt.clf()
print 'Equacao linear: '
print 'tempo(s) = {:3.2f} {:3.3f} x year'.format(what[0], what[1])
print 'RMSE = ' + str(utils.rmse(target, yhat))

# b) Predizer o resultado para o ano de 2016;
print 'Tempo previsto para a prova de 100 m para o ano de 2016:'
print 'tempo(s) = ' + str(classif_regres.linear_regression_estimate_output([2016.0], what)[0]) + 's'


# c) Utilize o teste de hipotese de Kendall para verificar se existe
# dependencia entre os atributos. Realize o teste para 5% e 1% de nivel
# de significancia;

tal, hip = utils.kendall_coef(target, yhat, alfa=0.05)

print 'Coeficiente de kendall = ' + str(tal)
print 'Resultado do teste de hipotese para alfa = 5%: ' + str(hip)

tal, hip = utils.kendall_coef(target, yhat, alfa=0.01)

print 'Coeficiente de kendall = ' + str(tal)
print 'Resultado do teste de hipotese para alfa = 1%: ' + str(hip)

# d) Calcule a correlacao entre os dados. Se o modulo da correlacao for
# acima de 0,85 realize o teste de hipotese de Pearson para 5% e 1% de nivel
# de significancia (teste bilateral).

p, hip = utils.pearson_coef(target, yhat, alfa=0.05)
print 'Coeficiente de Pearson = ' + str(p)
print 'Resultado do teste de hipotese para alfa = 5%: ' + str(hip)

p, hip = utils.pearson_coef(target, yhat, alfa=0.01)
print 'Coeficiente de Pearson = ' + str(p)
print 'Resultado do teste de hipotese para alfa = 1%: ' + str(hip)
