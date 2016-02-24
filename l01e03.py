import numpy as np
import utils
x = np.reshape(
    np.array([8, 11, 5, 15, 20, 11, 13, 15, 8, 9, 11, 18, 16, 8, 10]), (-1, 1))

mu, var = utils.calcula_media_variancia(x)
dp = np.sqrt(var)

n0 = x.size

print 'LETRA A'
e = 2
alfa = 0.05
t = utils.t_student(n0 - 1, alfa)
n = np.ceil(np.power(t * dp / e, 2))[0].astype(int)
print 'Numero de amostras necessario: ' + str(n)


print 'LETRA B'
e = 1
alfa = 0.05
t = utils.t_student(n0 - 1, alfa)
n = np.ceil(np.power(t * dp / e, 2))[0].astype(int)
print 'Numero de amostras necessario: ' + str(n)
