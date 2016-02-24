

def sequential_forward_selection(x_train, y_train, x_test, y_test, metrica,
                                 parada, classificador, param=-1):
    '''
        Seleciona atributos utilizando SFS utilizando os parametros fornecidos.
        Entradas:
            x = (n x m):    n amostras com m atributos cada.
            y = (n x 1):    classes de cada amostra.
            metrica:        funcao que calculara a metrica para selecionar o
        melhor atributo. A funcao deve ser do tipo metr = metrica(y, yhat)
            parada:         sera utilizada para selecionar o criterio de
        parada. se for maior que 1, fica subentendido que PARADA e o numero de
        atributos desejado.
        TODO: se for entre 0 e 1, fica subentendido que PARADA e o
        incremento de metrica abaixo do qual deve-se para de adicionar atributos.
            classificador:  eh a funcao do classificador que sera utilizado para separar as
        classes a partir dos atributos selecionados.
            param:           eh o parametro do classificador, se existir.
        Retorna:
            u:                vetor com os indices dos atributos escolhidos.
    '''
    import numpy as np
    from scipy.stats import norm
    x_train = np.array(x_train)
    y_train = np.reshape(np.array(y_train), (-1, 1))

    x_test = np.array(x_test)
    y_test = np.reshape(np.array(y_test), (-1, 1))

    n = x_train.shape[0]
    m = x_train.shape[1]

    # transform string attributes into numbers in order to classifier to work.
    for att in range(0, m):
        if isinstance(x_train[0, att], basestring):
            cl = np.union1d(x_train[:, att], x_test[:, att])
            for c in range(0, cl.size):
                ind = x_train[:, att] == cl[c]
                x_train[ind, att] = c + 1
                ind = x_test[:, att] == cl[c]
                x_test[ind, att] = c + 1

    x_train = x_train.astype(float)
    x_test = x_test.astype(float)
    u = np.array([])

    if parada >= 1:
        for i in range(0, parada):
            ul = np.array([])
            metr = 0
            for k in range(0, m):
                if k not in u:
                    ul = np.append(u, k).astype(int)
                    xl_train = x_train[:, ul]
                    xl_test = x_test[:, ul]
                    if param == -1:
                        yhat = classificador(xl_train, y_train, xl_test)
                    else:
                        yhat = classificador(xl_train, y_train, xl_test, param)
                    aux = metrica(y_test, yhat)
                    if aux > metr:
                        metr = aux
                        add = k

            u = np.append(u, add).astype(int)

    return u


def sequential_backward_elimination(x_train, y_train, x_test, y_test, metrica, parada, classificador, param=-1):
    '''
        Seleciona atributos utilizando SBE utilizando os parametros fornecidos.
            x = (n x m):    n amostras com m atributos cada.
            y = (n x 1):    classes de cada amostra.
            metrica:        funcao que calculara a metrica para selecionar o
        melhor atributo. A funcao deve ser do tipo metr = metrica(y, yhat)
            parada:         sera utilizada para selecionar o criterio de
        parada. se for maior que 1, fica subentendido que PARADA e o numero de
        atributos desejado.
        TODO: se for entre 0 e 1, fica subentendido que PARADA e o
        incremento de metrica abaixo do qual deve-se para de adicionar atributos.
            classificador:  eh a funcao do classificador que sera utilizado para separar as
        classes a partir dos atributos selecionados.
            param:           eh o parametro do classificador, se existir.
        Retorna:
            u:                vetor com os indices dos atributos escolhidos.
    '''
    import numpy as np
    from scipy.stats import norm
    x_train = np.array(x_train)
    y_train = np.reshape(np.array(y_train), (-1, 1))

    x_test = np.array(x_test)
    y_test = np.reshape(np.array(y_test), (-1, 1))

    n = x_train.shape[0]
    m = x_train.shape[1]

    u = np.arange(0, m)

    if parada >= 1:
        for i in range(0, m - parada):
            ul = np.copy(u)
            metr = 0
            for k in range(0, m):
                if k in u:
                    index = np.where(u == k)[0]
                    ul = np.delete(u, index).astype(int)
                    xl_train = x_train[:, ul]
                    xl_test = x_test[:, ul]
                    if param == -1:
                        yhat = classificador(xl_train, y_train, xl_test)
                    else:
                        yhat = classificador(xl_train, y_train, xl_test, param)
                    aux = metrica(y_test, yhat)
                    if aux > metr:
                        metr = aux
                        ret = k

            index = np.where(u == ret)[0]
            u = np.delete(u, index).astype(int)

    return u


def pca(x, k=0):
    '''
    Analise de componentes principais da matriz de dados X.
    Entrada:
        x:    matriz N x M em que cada linha representa
    uma amostra com M atributos.
        k:    numero de componentes principais a serem utilizadas.
    Retorna:
        xhat: retorna os dados na nova base utilizando K componentes principais.
        wl:   k autovalores.
        vl:   k autovetores que transformam os dados em x para xhat.
    '''
    import numpy as np
    import utils
    n = x.shape[0]  # numero de amostras
    if k == 0:
        k = n
    elif k > n:
        k = n

    # calcula a matriz de correlacao
    c = utils.correlacao(x)

    # calcula os autovalores / autovetores da matriz acima
    w, v = utils.eig(c)

    index = np.argsort(w)  # coloca os autovalores de forma crescente
    aux = [i for i in index[-1::-1]]
    index = aux  # agora de forma descrescente
    w = w[index]
    v = v[:, index]

    wl = w[:k]
    vl = v[:, :k]

    mu, var = utils.calcula_media_variancia(x)

    xhat = np.real(np.dot((x - mu), vl))  # dados na nova base

    return xhat, wl, vl


def pca_whitening(x, k=0):
    '''
    Analise de componentes principais da matriz de dados X com branqueamento 
    dos dados.

    Entrada:
        x:    matriz N x M em que cada linha representa
    uma amostra com M atributos.
        k:    numero de componentes principais a serem utilizadas.
    Retorna:
        xhat: retorna os dados na nova base utilizando K componentes principais.
        wl:   k autovalores.
        vl:   k autovetores que transformam os dados em x para xhat.
    '''
    import numpy as np
    import utils
    n = x.shape[0]  # numero de amostras
    if k == 0:
        k = n
    elif k > n:
        k = n

    # calcula a matriz de correlacao
    c = utils.correlacao(x)

    # calcula os autovalores / autovetores da matriz acima
    w, v = utils.eig(c)

    index = np.argsort(w)  # coloca os autovalores de forma crescente
    aux = [i for i in index[-1::-1]]
    index = aux  # agora de forma descrescente
    w = w[index]
    v = v[:, index]

    wl = w[:k]
    m = v[:, :k]

    mu, var = utils.calcula_media_variancia(x)

    xhat = (x - mu) / np.sqrt(var)

    v = np.zeros((k, k))

    for i in range(0, k):
        v[i, i] = np.power(wl[i], -0.5)

    vl = np.transpose(v)

    omega = np.dot(xhat, m)
    omega = np.real(np.dot(omega, vl))

    return omega, vl, m
