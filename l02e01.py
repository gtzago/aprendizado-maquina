import utils
import numpy as np
# 1) Para a base Car Evaluation (disponivel em http://archive.ics.uci.edu/ml/),
# considerando que o primeiro atributo e x1, o segundo e x2 e assim por diante,
# estime as probabilidades:

x, y = utils.abrir_dados_car('./bases/car/car.data')

n = x.shape[0]
m = x.shape[1]

# conversion = [{'vhigh': 1, 'high': 2, 'med': 3, 'low': 4},
#               {'vhigh': 1, 'high': 2, 'med': 3, 'low': 4},
#               {'2': 2, '3': 3, '4': 4, '5more': 5},
#               {'2': 2, '4': 4, 'more': 5},
#               {'small': 1, 'med': 2, 'big': 3},
#               {'low': 1, 'med': 2, 'high': 3}]

# a) P(x1 =med) e P(x2 = low)
# ou seja P(x1 = 3) e P(x2 = 4)
p1 = 100.0 * utils.simple_probability(x, 0, np.equal, 3)
p2 = 100.0 * utils.simple_probability(x, 1, np.equal, 4)
print 'P ( x1 = med ) = {:3.2f}% e P ( x2 = low ) = {:3.2f}%\n'.format(p1, p2)

# b) P(x6=high|x3=2) e P(x2=low|x4=4)
# ou seja P(x6 = 3 | x3 = 2) e P(x2 = 4 | x4 = 4 )

p1 = 100 * utils.conditional_probability(x, 5, np.equal, 3,
                                         2, np.equal, 2)

p2 = 100 * utils.conditional_probability(x, 1, np.equal, 4,
                                         3, np.equal, 4)

print 'P (x6=high|x3=2) = {:3.2f}% e P (x2 = 4 | x4 = 4 ) = {:3.2f}%\n'.format(
    p1, p2)
# c) P(x1=low|x2=low,X5=small) e P(x4=4|x1=med,X3=2)
# ou seja P(x1 = 4 | x2 = 4, x5=1) e P(x4 = 4 | x1 = 3, x3=2)

xl = x[np.where(np.logical_and(x[:, 1] == 4, x[:, 4] == 1))[0], :]
p1 = 100.0 * utils.simple_probability(xl, 0, np.equal, 4)

xl = x[np.where(np.logical_and(x[:, 0] == 3, x[:, 2] == 2))[0], :]
p2 = 100.0 * utils.simple_probability(xl, 3, np.equal, 4)

print 'P (x1=low|x2=low,X5=small) = {:3.2f}% e P (x4=4|x1=med,X3=2) = {:3.2f}%\n'.format(
    p1, p2)

# d) P(x2= vhigh,X3=2|X4=2) e P(x3=4,x5=med|x1=med)
# ou seja P(x2 = 1, x3=2 | x4 = 2) e P(x3 = 4, x5=2 | x1 = 3)

xl = x[np.where(x[:, 3] == 2)[0], :]
p1 = 100.0 * utils.joint_probability(xl, 1, np.equal, 1, 2, np.equal, 2)

xl = x[np.where(x[:, 0] == 3)[0], :]
p2 = 100.0 * utils.joint_probability(xl, 2, np.equal, 4, 4, np.equal, 3)

print 'P (x2= vhigh,X3=2|X4=2) = {:3.2f}% e P (x3=4,x5=med|x1=med) = {:3.2f}%\n'.format(
    p1, p2)
