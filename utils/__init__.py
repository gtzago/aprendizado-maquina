import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
import networkx as nx


def draw_graph(g, pos):

    nx.draw_networkx(
        g, pos, node_size=1200, font_size=14, node_shape='s', node_color='b', with_labels=False)

    node_labels = nx.get_node_attributes(g, 'name')
    nx.draw_networkx_labels(g, pos, labels=node_labels)

    edge_labels = nx.get_edge_attributes(g, 'value')
    nx.draw_networkx_edge_labels(g, pos, labels=edge_labels)

    #nx.draw(g, pos)
    # show graph
    plt.savefig('./bases/results/CarDecisionTree.png')
    plt.show()


def cartesian(arrays, out=None):
    """
    Generate a cartesian product of input arrays.

    Parameters
    ----------
    arrays : list of array-like
        1-D arrays to form the cartesian product of.
    out : ndarray
        Array to place the cartesian product in.

    Returns
    -------
    out : ndarray
        2-D array of shape (M, len(arrays)) containing cartesian products
        formed of input arrays.

    Examples
    --------
    >>> cartesian(([1, 2, 3], [4, 5], [6, 7]))
    array([[1, 4, 6],
           [1, 4, 7],
           [1, 5, 6],
           [1, 5, 7],
           [2, 4, 6],
           [2, 4, 7],
           [2, 5, 6],
           [2, 5, 7],
           [3, 4, 6],
           [3, 4, 7],
           [3, 5, 6],
           [3, 5, 7]])

    """
    arrays = [np.asarray(x) for x in arrays]
    dtype = arrays[0].dtype

    n = np.prod([x.size for x in arrays])
    if out is None:
        out = np.zeros([n, len(arrays)], dtype=dtype)

    m = n / arrays[0].size
    out[:, 0] = np.repeat(arrays[0], m)
    if arrays[1:]:
        cartesian(arrays[1:], out=out[0:m, 1:])
        for j in xrange(1, arrays[0].size):
            out[j * m:(j + 1) * m, 1:] = out[0:m, 1:]
    return out


def joint_probability(x, att1, condition_function1, thr1, att2, condition_function2, thr2):
    '''
        Calculates P(att1 COND1 thr1, att2 COND2 thr2 )
    '''
    n = x.shape[0]
    m = x.shape[1]

    p = 1.0 * np.where(np.logical_and(condition_function1(x[:, att1], thr1),
                                      condition_function1(x[:, att2], thr2))
                       )[0].size / n

    return p


def conditional_probability(x, att1, condition_function1, thr1, att2, condition_function2, thr2):
    '''
        Calculates P(att1 COND1 thr1 | att2 COND2 thr2 )
    '''
    n = x.shape[0]
    m = x.shape[1]

    if isinstance(x[0, att2], basestring):
        ind = np.where(np.core.defchararray.equal(x[:, att2], thr2))[0]
    else:
        ind = np.where(condition_function2(x[:, att2], thr2))[0]

    xl = x[ind, :]

    p = simple_probability(xl, att1, condition_function1, thr1)

    return p


def simple_probability(x, att, condition_function=np.equal, thr=None, laplace_smooth=False, probmod='freq'):
    '''
        Calculate the probability that atribute ATT at matrix X
        attend the condition given by condition_function and 
        threshold THR.
    '''
    n = x.shape[0]
    m = x.shape[1]

    if not laplace_smooth:
        if probmod == 'freq':
            if n != 0:
                if isinstance(x[0, att], basestring):
                    p = 1.0 * \
                        np.where(np.core.defchararray.equal(x[:, att], thr))[
                            0].size / n
                else:
                    p = 1.0 * \
                        np.where(condition_function(x[:, att], thr))[
                            0].size / n
            else:
                p = -1
        else:
            mu = np.mean(x[:, att])
            var = np.var(x[:, att])
            p = np.exp(-(thr - mu)**2 / (2 * var)) / np.sqrt(2 * np.pi * var)
    else:
        p = 1.0 * \
            (np.where(condition_function(x[:, att], thr))[
             0].size + 1) / (n + m)

    return p


def voronoi(x, y):
    '''
        x is the input.
        y is the class.
        Feito no codigo.
    '''
    import matplotlib.pyplot as plt

    x = np.array(x)
    y = np.array(y)

    pass


def kendall_coef(x, y, alfa=0.05):
    '''
        Calcula o coeficiente de Kendall entre os dados em x e y.
        Entradas:
            x e y: entradas.
            alfa (opcional): 1 - probabilidade de certeza.
        Retorna:
            tal: coeficiente de Kendall.
            hip: resultado do teste de hipotese.
    '''

    x = np.array(x)
    y = np.array(y)
    n = x.size

    tal = 0
    nxy = 0
    for i in range(1, n):
        for j in range(0, i - 1):
            aux = np.sign(x[j] - x[i]) * np.sign(y[j] - y[i])
            if aux != 0:
                tal = tal + np.sign(x[j] - x[i]) * np.sign(y[j] - y[i])
                nxy = nxy + 1
    tal = 1.0 * tal / nxy

    # hipotesis test
    z = norm.ppf(1 - 1.0 * alfa / 2)

    hip = abs(tal) > z * np.sqrt(2.0 * (2 * n + 5) / (9 * n * (n - 1)))

    return tal, hip


def t_student(n, alfa):
    '''
        Calcula o t_{alfa/2,n} da distribuicao t-student.
    '''
    from scipy.stats import t

    return t.ppf(1 - 1.0 * alfa / 2, n)


def covariancia(x, y):
    '''
        Calcula a covariancia entre os dois vetores de entrada.
    '''
    x = np.array(x)
    y = np.array(y)
    n = x.size

    mux, varx = calcula_media_variancia(x)
    muy, vary = calcula_media_variancia(y)

    return np.sum(np.multiply(x - mux, y - muy)) / (n - 1)


def pearson_coef(x, y, alfa=0.05):
    '''
        Calcula o coeficiente de Pearson entre os dados em x e y.
        Entradas:
            x e y: entradas.
            alfa (opcional): 1 - probabilidade de certeza.
        Retorna:
            p: coeficiente de Pearson.
            hip: resultado do teste de hipotese.
    '''
    x = np.array(x)
    y = np.array(y)
    n = x.size

    mux, varx = calcula_media_variancia(x)
    muy, vary = calcula_media_variancia(y)

    p = covariancia(x, y) / np.sqrt(varx * vary)

    t0 = p * np.sqrt((n - 2) / (1 - np.power(p, 2)))

    hip = np.abs(t0) > t_student(n - 2, alfa)

    return p, hip


def coef_determinacao_ajustado(y, yhat, degree=1):
    '''
        Calcula o coeficiente de determinacao ajustado, que leva em
        consideracao a ordem do modelo utilizado ao calcular o
        coeficiente de ajuste dos dados.
        Entradas:
            y:    vetor de saidas reais.
            yhat: vetor de saidas estimadas.
        Saida:
            r2: coeficiente de determinacao ajustado.
    '''

    y = np.array(y)
    yhat = np.array(yhat)
    n = y.size

    r2 = 1 - ((n - 1) / (n - degree)) * \
        np.sum(np.power(y - yhat, 2)) / np.sum(np.power(y - np.mean(y), 2))

    return r2


def mape(y, yhat):
    '''
        The mean absolute percentage error (MAPE), also known as mean absolute
        percentage deviation (MAPD).
        Entradas:
            y:    vetor de saidas reais.
            yhat: vetor de saidas estimadas.
        Saida:
            MAPE.
    '''
    y = np.array(y)
    yhat = np.array(yhat)
    n = y.size

    return np.sum(np.abs((yhat - y) / y)) / n


def rmse(x, y):
    '''
        Calcula o erro medio quadratico entre os dois vetores de entrada.
        Entradas:
            x:    vetor de saidas reais.
            y:    vetor de saidas estimadas.
        Saida:
            RMSE.
    '''
    x = np.array(x)
    y = np.array(y)
    n = x.size

    return np.sum(np.power(x - y, 2)) / n


def tabela_contingencia(y, yhat, classes=0):
    '''
        Calcula a tabela de contingencia (matriz de confusao).
        Entradas:
            y:    vetor de classes reais.
            yhat: vetor de classes estimadas.
        Retorna:
            c: matriz de confusao.
            onde
                ci,j: associado pelo algoritmo como i, e pelo especialista
                como j.
    '''
    n = len(yhat)
    y = np.reshape(np.array(y), (-1, 1))
    yhat = np.reshape(np.array(yhat), (-1, 1))
    if classes == 0:
        classes = np.union1d(y, yhat)

    c = np.zeros((classes.size, classes.size))
    for i in range(0, classes.size):
        a = np.where(yhat == classes[i])[0]
        for j in range(0, classes.size):
            b = np.where(y == classes[j])[0]
            c[i, j] = np.intersect1d(a, b).size

    return c


def precisao_revocacao(y, yhat, classes=0):
    '''
        Calcula a precisao e a revocacao entre os dados passados.
        Entradas:
            y:    vetor de classes reais.
            yhat: vetor de classes estimadas.
        Retorna:
            pr  = precisao na coluna 0, revocacao na coluna 1.
            Cada linha representa uma classe.
        Forma de calculo: Macro averaged.
    '''
    if classes == 0:
        classes = np.union1d(y, yhat)
    c = tabela_contingencia(y, yhat, classes)

    pr = np.zeros((c.shape[0], 2))

    for cls in range(0, c.shape[0]):
        tp = c[cls, cls]
        fp = np.sum(c[cls, :]) - tp
        fn = np.sum(c[:, cls]) - tp
        if (tp + fp) != 0:
            pr[cls, 0] = tp / (tp + fp)
        else:
            pr[cls, 0] = 0
        if (tp + fn) != 0:
            pr[cls, 1] = tp / (tp + fn)
        else:
            pr[cls, 1] = 0

    return pr


def accuracy(y, yhat):
    '''
    Calcula a acuracia entre os vetores.
    Entradas:
        y:    vetor de classes reais.
        yhat: vetor de classes estimadas.
    Saida:
        acc: acuracia.
    '''
    n = len(yhat)
    y = np.reshape(np.array(y), (-1, 1))
    yhat = np.reshape(np.array(yhat), (-1, 1))

    hit = np.where(y == yhat)

    acc = 1.0 * hit[0].size / n

    return acc


def mode(x):
    '''
    Calcula a moda do vetor x.
    Entrada:
        x: vetor cuja moda sera calculada.
    Saida:
        moda: valor que mais se repete em x.
    '''
    x = np.reshape(np.array(x), (-1, 1))

    conjunto = np.union1d(x, x)

    n = 0

    for value in conjunto:
        num = np.where(x == value)[0].size
        if num > n:
            n = num
            mod = value

    return mod


def produto_interno(u, v):
    '''
        Calcula o produto interno entre dois vetores 1D.
        Entrada:
            u e v: entradas.
        Saida:
            produto interno entre u e v.
    '''
    u = np.reshape(u, (1, -1))
    v = np.reshape(v, (-1, 1))

    return np.dot(u, v)


def cosseno(u, v):
    '''
        Calcula o cosseno do angulo entre dois vetores de 1D.
    '''
    return produto_interno(u, v) / (np.sqrt(produto_interno(u, u)) * np.sqrt(
        produto_interno(v, v)))


def abrir_dados_spiral(csv_file):
    '''
    Acessa a base de dados Spiral localizada em CSV_FILE.
    '''
    import csv
    x = np.array([])
    y = np.array([])
    with open(csv_file, 'rb') as csvfile:
        csvreader = csv.reader(csvfile, delimiter=',', quotechar='|')
        for row in csvreader:
            aux = [float(i) for i in row[0:2]]
            x = np.append(x, aux)
            y = np.append(y, int(row[2]))
    return np.reshape(x,(-1,2)), np.reshape(y,(-1,1))

def abrir_dados_jain(csv_file):
    '''
    Acessa a base de dados Jain localizada em CSV_FILE.
    '''
    import csv
    x = np.array([])
    y = np.array([])
    with open(csv_file, 'rb') as csvfile:
        csvreader = csv.reader(csvfile, delimiter=',', quotechar='|')
        for row in csvreader:
            aux = [float(i) for i in row[0:2]]
            x = np.append(x, aux)
            y = np.append(y, int(row[2]))
    return np.reshape(x,(-1,2)), np.reshape(y,(-1,1))

def abrir_dados_nebulosa(path):
    '''
    Acessa a base de dados nebulosa reduzido localizada em CSV_FILE.
    '''
    import csv

    csv_file = path + '/nebulosa_train.txt'
    x_train = np.array([])
    target_train = []
    with open(csv_file, 'rb') as csvfile:
        csvreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
        for row in csvreader:
            aux = []
            for i in row[0:-1]:
                if i != '?':
                    aux.append(float(i))
                else:
                    aux.append(-1)
            x_train = np.append(x_train, np.array(aux))
            target_train.append(row[-1])

    csv_file = path + '/nebulosa_test.txt'
    x_test = np.array([])
    target_test = []
    with open(csv_file, 'rb') as csvfile:
        csvreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
        for row in csvreader:
            aux = []
            for i in row[0:-1]:
                if i != '?':
                    aux.append(float(i))
                else:
                    aux.append(-1)
            x_test = np.append(x_test, np.array(aux))
            target_test.append(row[-1])

    return np.reshape(x_train, (-1, 7)), target_train, np.reshape(x_test, (-1, 7)), target_test


def abrir_dados_pista(csv_file):
    '''
    Acessa a base de dados Polinomio localizada em CSV_FILE.
    '''
    import csv
    x = np.array([])
    with open(csv_file, 'rb') as csvfile:
        csvreader = csv.reader(csvfile, delimiter=',', quotechar='|')
        flag = False
        for row in csvreader:
            if flag:
                aux = [int(row[i]) for i in range(0, 5)]
                x = np.append(x, aux)
            else:
                flag = True
    x = np.reshape(x, (-1, 5))
    return x.astype(int)

def abrir_dados_servo(csv_file):
    '''
    Acessa a base de dados Servo localizada em CSV_FILE.
    '''
    import csv
    x = np.array([])
    y = np.array([])
    with open(csv_file, 'rb') as csvfile:
        csvreader = csv.reader(csvfile, delimiter=',', quotechar='|')
        for row in csvreader:
            aux = [str(i) for i in row[0:4]]
            x = np.append(x, aux)
            y = np.append(y, float(row[4]))
    x = np.reshape(x, (-1, 4))
    y = np.reshape(y, (y.size, -1))
    return x, y

def abrir_dados_polinomio(csv_file):
    '''
    Acessa a base de dados Polinomio localizada em CSV_FILE.
    '''
    import csv
    x = np.array([])
    y = np.array([])
    with open(csv_file, 'rb') as csvfile:
        csvreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
        for row in csvreader:
            x = np.append(x, float(row[0]))
            y = np.append(y, float(row[2]))
    x = np.reshape(x, (x.size, -1))
    y = np.reshape(y, (y.size, -1))
    return x, y


def abrir_dados_runner(csv_file):
    '''
    Acessa a base de dados Runner localizada em CSV_FILE.
    '''
    import csv
    target = np.array([])
    x = []
    with open(csv_file, 'rb') as csvfile:
        csvreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
        for row in csvreader:
            target = np.append(target, float(row[1]))
            x.append(int(row[0]))
    return np.array(x), target


def abrir_dados_cnae9_reduzido(csv_file):
    '''
    Acessa a base de dados CNAE-9 reduzido localizada em CSV_FILE.
    '''
    import csv
    x = np.array([])
    target = []
    with open(csv_file, 'rb') as csvfile:
        csvreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
        for row in csvreader:
            aux = [float(i) for i in row[1:]]
            x = np.append(x, aux)
            target.append(row[0])
    return np.reshape(x, (-1, 502)), np.array(target)


def abrir_dados_iris(csv_file):
    '''
    Acessa a base de dados iris localizada em CSV_FILE.
    '''
    import csv
    x = np.array([])
    target = []
    with open(csv_file, 'rb') as csvfile:
        csvreader = csv.reader(csvfile, delimiter=',', quotechar='|')
        for row in csvreader:
            aux = [float(i) for i in row[:-1]]
            x = np.append(x, aux)
            target.append(row[-1])
    return np.reshape(x, (-1, 4)), np.reshape(np.array(target), (-1, 1))


def abrir_dados_balance_scale(csv_file):
    '''
    Acessa a base de dados balance-scale localizada em CSV_FILE.
    '''
    import csv
    x = np.array([])
    target = np.array([])
    with open(csv_file, 'rb') as csvfile:
        csvreader = csv.reader(csvfile, delimiter=',', quotechar='|')
        for row in csvreader:
            aux = [int(i) for i in row[1:]]
            x = np.append(x, aux)
            target = np.append(target, row[0])
    return np.reshape(x, (-1, 4)), target


def abrir_dados_wine(csv_file):
    '''
    Acessa a base de dados wine localizada em CSV_FILE.
    '''
    import csv
    x = np.array([])
    target = []
    with open(csv_file, 'rb') as csvfile:
        csvreader = csv.reader(csvfile, delimiter=',', quotechar='|')
        for row in csvreader:
            aux = [float(i) for i in row[1:]]
            x = np.append(x, aux)
            target.append(row[0])
    return np.reshape(x, (-1, 13)), np.array(target)


def abrir_dados_car(csv_file):
    '''
    Acessa a base de dados car evaluation localizada em CSV_FILE.
    '''
    import csv
    x = np.array([])
    target = []

    conversion = [{'vhigh': 1, 'high': 2, 'med': 3, 'low': 4}, {'vhigh': 1, 'high': 2, 'med': 3, 'low': 4}, {
        '2': 2, '3': 3, '4': 4, '5more': 5}, {'2': 2, '4': 4, 'more': 5}, {'small': 1, 'med': 2, 'big': 3}, {'low': 1, 'med': 2, 'high': 3}]

    with open(csv_file, 'rb') as csvfile:
        csvreader = csv.reader(csvfile, delimiter=',', quotechar='|')
        for row in csvreader:
            for att in range(0, 6):
                # converte de literal em numerico.
                x = np.append(x, conversion[att][row[att]])
            target.append(row[-1])
    return np.reshape(x, (-1, 6)), np.array(target)


def abrir_dados_car_str(csv_file):
    '''
    Acessa a base de dados car evaluation localizada em CSV_FILE.
    '''
    import csv
    x = np.array([])
    target = []

    with open(csv_file, 'rb') as csvfile:
        csvreader = csv.reader(csvfile, delimiter=',', quotechar='|')
        for row in csvreader:
            x = np.append(x, row[0:6])
            target.append(row[-1])
    return np.reshape(x, (-1, 6)), np.array(target)

def abrir_dados_dados(csv_file):
    '''
    Acessa a base de dados car evaluation localizada em CSV_FILE.
    '''
    import csv
    x = np.array([])

    with open(csv_file, 'rb') as csvfile:
        csvreader = csv.reader(csvfile, delimiter=',', quotechar='|')
        for row in csvreader:
            x = np.append(x, float(row[0]))
    return x


def calcula_media_variancia(x):
    '''
    Calcula a media e a variancia dos dados em x.
    Entrada:
        x: matriz N x M em que cada linha representa
    uma amostra com M atributos.
    Saida:
        mu: media de x.
        var: variancia de x.
    '''
    mu = np.sum(x, axis=0) / x.shape[0]

    var = np.sum(
        np.power(x - mu * np.ones_like(x), 2), axis=0) / x.shape[0]

    return mu, var


def autocovariancia(x):
    '''
    Calcula a matriz de autocovariancia dos dados em x.
    Entrada:
        x:    matriz N x M em que cada linha representa
    uma amostra com M atributos.
    '''
    n = x.shape[0]  # numero de amostras

    mu, var = calcula_media_variancia(x)
    # Calcular a matriz C de autocovariancia/correlacao dos dados;

    # Normalizacao
    # subtrai a media
    x_hat = (x - mu)

    # calcula a matriz de correlacao
    c = np.dot(np.transpose(x_hat), x_hat) / (n - 1)

    return c


def correlacao(x):
    '''
    Calcula a matriz de correlacao dos dados em x.
    Entrada:
        x:    matriz N x M em que cada linha representa
    uma amostra com M atributos.
    Saida:
        matriz de autocorrelacao de x.
    '''
    n = x.shape[0]  # numero de amostras

    mu, var = calcula_media_variancia(x)
    # Calcular a matriz C de autocovariancia/correlacao dos dados;

    # Normalizacao
    # subtrai a media e divide pelo desvio padrao
    x_hat = (x - mu) / np.sqrt(var)

    # calcula a matriz de correlacao
    c = np.dot(np.transpose(x_hat), x_hat) / (n - 1)

    return c


def eig(x):
    '''
    Calcula os autovalores e autovetores dos dados em x.
    Tanto os autovalores W quanto os autovetores V[:,]
    estao ordenados de forma decrescente.
    Entrada:
        x:    matriz N x M em que cada linha representa
    uma amostra com M atributos.
    Retorna:
        w:    autovalores em ordem decrescente.
        v:    autovetores em ordem decrescente de autovalores.
    '''

    # Obter os autovalores e autovetores da matriz C;
    w, v = np.linalg.eig(x)

    index = np.argsort(w)  # coloca os autovalores de forma crescente
    aux = [i for i in index[-1::-1]]
    index = aux  # agora de forma descrescente
    w = w[index]
    v = v[:, index]

    return w, v
