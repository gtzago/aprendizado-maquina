def naive_bayes(x_train, target_train, x_test, laplace_smooth=False, probmod = 'freq'):
    import numpy as np
    import utils

    x_train = np.array(x_train)
    target_train = np.array(target_train)
    x_test = np.array(x_test)

    cls = np.array([])

    n = x_test.shape[0]
    m = x_test.shape[1]

    classes = np.union1d(target_train, target_train)

    xi = {}  # exemplos de cada classe.
    pc_log = np.array([])  # P(c=ci).
    for c in range(0, classes.size):
        ind = np.where(target_train == classes[c])[0]
        pc_log = np.append(
            pc_log, np.log10(1.0 * ind.size / x_train.shape[0]))
        xi[c] = x_train[ind, :]

    p = np.zeros(classes.size)
    for i in range(0, n):
        for c in range(0, classes.size):
            p[c] = pc_log[c]
            for d in range(0, m):
                p[c] = p[c] + \
                    np.log10(
                        utils.simple_probability(xi[c], d, np.equal,
                                                 x_test[i, d], laplace_smooth, probmod))
        cls = np.append(cls, classes[np.argmax(p)])

    return np.reshape(cls, (-1, 1))


def similaridade_cosseno(x_train, target_train, x_test):
    '''
    Classificador por similaridade cosseno.
    Entradas:
        x_train (n x m):    Conjunto de n amostras com m atributos
    cada que serao utilizadas para treinar o modelo.
        target_train (n x 1): classes de x_train.
        x_test (n2 x m):      Conjunto de n2 amostras com m atributos
    cada que serao classificados utilizando o modelo treinado.
    Retorna:
        yhat (n2x1):    classificacao de x_test.
    '''
    import numpy as np
    import utils

    x_train = np.array(x_train)
    target_train = np.array(target_train)
    x_test = np.array(x_test)

    cls = []

    n = x_test.shape[0]
    m = x_test.shape[1]

    # encontrando os centroides
    centroides = []
    classes = list(set(target_train))
    for i in classes:
        indexes = np.where(target_train == i)
        examples = x_train[indexes, :]
        centroides.append(np.mean(examples, 1))

    centroides = np.reshape(centroides, (-1, m))

    for i in range(0, n):
        distance = np.zeros((centroides.shape[0], 1))
        for j in range(0, centroides.shape[0]):
            distance[j] = utils.cosseno(x_test[i, :], centroides[j, :])
        ind = np.argmax(distance)
        cls.append(classes[ind])

    return np.array(cls)


def rocchio(x_train, target_train, x_test):
    '''
    Classificador Rocchio.
    Entradas:
        x_train (n x m):    Conjunto de n amostras com m atributos
    cada que serao utilizadas para treinar o modelo.
        target_train (n x 1): classes de x_train.
        x_test (n2 x m):      Conjunto de n2 amostras com m atributos
    cada que serao classificados utilizando o modelo treinado.
    Retorna:
        yhat (n2x1):    classificacao de x_test.
    '''
    import numpy as np

    cls = []

    n = x_test.shape[0]
    m = x_test.shape[1]

    # encontrando os centroides
    centroides = []
    classes = np.union1d(target_train, target_train)
    for i in classes:
        indexes = np.where(target_train == i)[0]
        examples = x_train[indexes, :]
        centroides.append(np.mean(examples, 0))

    centroides = np.reshape(centroides, (-1, m))

    for i in range(0, n):
        distance = np.sqrt(np.sum(np.power(centroides - x_test[i, :], 2), 1))
        ind = np.argmin(distance)
        cls.append(classes[ind])

    return np.array(cls)


def knn(x_train, target_train, x_test, k):
    '''
    Classificador Knn.
    Entradas:
        x_train (n x m):    Conjunto de n amostras com m atributos
    cada que serao utilizadas para treinar o modelo.
        target_train (n x 1): classes de x_train.
        x_test (n2 x m):      Conjunto de n2 amostras com m atributos
    cada que serao classificados utilizando o modelo treinado.
        k:    numero de vizinhos mais proximos que serao utilizados.
    Retorna:
        yhat (n2x1):    classificacao de x_test.
    '''

    import numpy as np
    import utils

    cls = []

    n = x_test.shape[0]
    m = x_test.shape[1]

    for i in range(0, n):
        distance = np.sqrt(np.sum(np.power(x_train - x_test[i, :], 2), 1))
        ind = np.argsort(distance)
        cls.append(utils.mode(target_train[ind[0:k]]))

    return np.reshape(np.array(cls), (-1, 1))


def ransac(x, y, degree=1, s=-1, TAL=-1):
    '''
    Realiza a regressao linear da funcao que leva as entradas x
    aa saida y utilizando o algoritmo de minimos quadrados e
    o metodo de RANSAC para eliminar outliers.
    Entradas:
        x:    matriz de entrada: n x m.
        y:    matriz de saidas: n x 1.
        degree: grau do polinomio que sera estimado.
        s:    numnero de amostras utilizadas para estimar o
        modelo.
        TAL:    distancia da funcao estimada, a partir desta
        distancia, os dados sao considerados outliers.
    Retorna:
        what = matriz dos parametros estimados: 1 x m+1.
    '''

    import numpy as np
    from scipy.stats import norm
    x = np.array(x)
    y = np.array(y)
    n = x.shape[0]
    m = x.shape[1]

    # determinacao de s caso o mesmo nao seja passado
    if s == -1:
        s = np.max([2, m + 1])

    # determinacao de L
    # probabilidade que ao menos um dos conjuntos de s amostras nao inclua um
    # outlier
    p = 0.99
    e = 0.2  # probabilidade de selecionar um outlier
    L = np.log10(1 - p) / np.log10(1 - np.power(1 - e, s))

    # determinacao de T
    T = n * (1 - e)

    # determinacao de TAL
    if TAL == -1:
        TAL = 2

    count = 0
    biggerk = 0

    while count < L:
        # Selecione aleatoriamente um conjunto de s amostras;
        random_ind = np.random.randint(0, n, size=s)
        # Estime os parametros do modelo com as s amostras;
        what = linear_regression(x[random_ind, :], y[random_ind], degree)

        # Encontre todas as k amostras da matriz X que estao dentro de uma distancia
        # limite TAL do modelo;
        yhat = linear_regression_estimate_output(x, what, degree)

        X = np.append(y, x, 1)
        Xhat = np.append(yhat, x, 1)

        dist = np.array([])
        for i in range(0, n):
            dist = np.append(
                dist, np.min(np.sum(np.power(Xhat - X[i, :], 2), 1)))
        xk = x[np.where(dist < TAL)[0], :]
        yk = y[np.where(dist < TAL)[0], :]
        k = np.where(dist < TAL)[0].size

        # Se k for maior ou igual a um limiar T, os parametros do modelo sao estimados
        # usando as k amostras de X e o codigo termina;
        if k > T:
            what = linear_regression(xk, yk, degree)
            return what

        # Se k for menor do que um limiar T, volta ao passo 1;

        # Apos L tentativas o maior conjunto de k amostras e selecionado e o modelo e
        # reestimado com eles.

        if k > biggerk:
            biggerk = k
            x_biggerk = xk
            y_biggerk = yk

        count = count + 1

    what = linear_regression(x_biggerk, y_biggerk, degree)
    return what


def linear_regression(x, y, degree=1):
    '''Realiza a regressao linar da funcao que leva as entradas x
    aa saida y utilizando o algoritmo de minimos quadrados.
    Entradas:
        x:    matriz de entrada: n x m.
        y:    matriz de saidas: n x 1.
        degree: grau do polinomio que sera estimado.
    Retorna:
        what = matriz dos parametros estimados: 1 x m+1.
    '''
    import numpy as np
    x = np.array(x)
    y = np.array(y)
    n = x.shape[0]
    if x.size == x.shape[0]:
        m = 1
        x = np.reshape(x, (n, 1))
    else:
        m = x.shape[1]

    # coloca as potencias das entradas como se fossem outros
    # atributos:
    # X = [
    #        x1, x2, x1^2, x2^2
    #]
    aux = x
    if degree > 1:
        for d in range(2, degree + 1):
            aux = np.append(aux, np.power(x, d), 1)
    x = np.copy(aux)

    x = np.append(np.ones((n, 1)), x, 1)  # adiciona os 1s para w0.

    xt = np.transpose(x)
    yt = np.transpose(y)
    what = np.transpose(np.dot(np.dot(yt, x), np.linalg.inv(np.dot(xt, x))))

    return what


def linear_regression_estimate_output(x, w, degree=1):
    '''Realiza regressao linar a partir da matriz de parametros
    utilizando o algoritmo de minimos quadrados.
    Entradas:
        x = matriz de entrada: n x m
        w = matriz dos parametros estimados: 1 x m+1
    Retorna:
        y = matriz de saidas: n x 1
    '''
    import numpy as np
    x = np.array(x)
    w = np.array(w)
    n = x.shape[0]
    if x.size == x.shape[0]:
        m = 1
        x = np.reshape(x, (n, 1))
    else:
        m = x.shape[1]

    aux = x
    if degree > 1:
        for d in range(2, degree + 1):
            aux = np.append(aux, np.power(x, d), 1)
    x = np.copy(aux)

    x = np.append(np.ones((n, 1)), x, 1)  # adiciona os 1s para w0.

    return np.transpose(np.dot(np.transpose(w), np.transpose(x)))


def kfold_cross_validation(x, y, k, metrica, classificador, param=-1):
    '''
        Realiza o k-fold crossvalidation utilizando o classificador
        fornecido (e seu parametro) e testando sobre a metrica fornecida.
        Entradas:
            x = (n x m):    n amostras com m atributos cada.
            y = (n x 1):    classes de cada amostra.
            k:              numero de folds que serao utilizados.
            metrica:        funcao que calculara a metrica para selecionar o
        melhor conjunto. A funcao deve ser do tipo metr = metrica(y, yhat)
            classificador:  eh a funcao do classificador que sera utilizado para separar as
        classes a partir dos atributos selecionados.
            param:           eh o parametro do classificador, se existir.
        Retorna:
            metr:            metrica media.
            contingencia:    matriz de contingencia media.
            pr:              matriz de precisao e revocacao media.
    '''
    import numpy as np
    import utils

    x = np.array(x)
    y = np.array(y)
    n = x.shape[0]
    m = x.shape[1]

    classes = np.array(list(set(y)))

#     folds contendo os dados. fold[fold][n,m]
    fold = {}
    for f in range(0, k):
        fold[f] = np.array([])

    f = 0  # fold em que a amostra sera guardada
    for cls in classes:
        i = 0  # amostra
        ind = np.where(y == cls)[0]

        while i < len(ind):
            if fold[f].size == 0:
                fold[f] = ind[i]
            else:
                fold[f] = np.vstack((fold[f], ind[i]))
            i = i + 1
            f = f + 1
            if f == k:
                f = 0

    # treina mudando os conjuntos de treino/teste.
    contingencia = np.zeros((classes.size, classes.size))  # matriz de confusao
    pr = np.zeros((classes.size, 2))  # precisao e revocacao

    metr = []
    for f in range(0, k):
        # usa o fold f para teste.
        # separa os conjuntos de teste e treino.
        x_test = np.reshape(x[fold[f], :], (-1, m))
        y_test = y[fold[f]]
#         x_train = np.ones((1, m))
#         y_train = np.ones((1, 1))
#         for f1 in range(0, k):
#             if f1 != f:
#                 x_train = np.vstack(
#                     (x_train, np.reshape(x[fold[f1]], (-1, m))))
#                 y_train = np.vstack((y_train, y[fold[f1]]))
#         x_train = np.delete(x_train, 0, 0)
#         y_train = np.delete(y_train, 0, 0)

        # separa o conjunto de treino entre treino e validacao.
        metrf = 0
        for f1 in range(0, k):
            if f1 != f:
                # usa o fold f1 como validacao.
                x_val = np.reshape(x[fold[f1], :], (-1, m))
                y_val = y[fold[f1]]
                x_train = np.ones((1, m))
                y_train = np.ones((1, 1))
                for f2 in range(0, k):
                    if f2 != f and f2 != f1:
                        x_train = np.vstack(
                            (x_train, np.reshape(x[fold[f2]], (-1, m))))
                        y_train = np.vstack((y_train, y[fold[f2]]))
                x_train = np.delete(x_train, 0, 0)
                y_train = np.delete(y_train, 0, 0)

                if param == -1:
                    yhat = classificador(x_train, y_train, x_val)
                else:
                    yhat = classificador(x_train, y_train, x_val, param)

                aux = metrica(y_val, yhat)
                if aux > metrf:
                    metrf = aux
                    x_train_chosen = np.copy(x_train)
                    y_train_chosen = np.copy(y_train)

        # testa o melhor conjunto de treino no conjunto de teste f.
        print 'Testando no fold ' + str(f)
        if param == -1:
            yhat = classificador(x_train_chosen, y_train_chosen, x_test)
        else:
            yhat = classificador(x_train_chosen, y_train_chosen, x_test, param)

        # guarda o resultado da metrica para o fold f.
        metr.append(metrica(y_test, yhat))
        print 'Acuracia = ' + str(metr[-1]) + '\n'

        contingencia = contingencia + \
            utils.tabela_contingencia(y_test, yhat, classes)
        pr = pr + utils.precisao_revocacao(y_test, yhat, classes)

    # return acuracia_media, macroprecision_medio, macrorecall_medio,
    # tabela_contingencia_medio
    contingencia = contingencia / k
    pr = pr / k
    return np.mean(metr), contingencia, pr
