import numpy as np
import utils
import reduc_dim
import classif_regres
import networkx as nx


class RegressionTree(object):

    def __init__(self, nclasses):
        self.root = None
        self.children = None
        self.att = None
        self.n = 0
        self.m = 0
        self.nclasses = nclasses

    def train(self, x, y, SDRMIN=0.05, NMIN=0, xvalues={}):
        self.x = np.array(x)
        self.y = np.reshape(y, (-1, 1))
        self.n = self.y.size
        self.m = self.x.shape[1]

        if len(xvalues.keys()) == 0:
            for att in range(0, self.m):
                xvalues[att] = np.union1d(self.x[:, att], self.x[:, att])

        if self.n < NMIN or self.n <= 1 or np.std(y) == 0.0:
            self.root = np.mean(self.y)
        else:
            sdr = np.zeros(self.m)
            for att in range(0, self.m):
                # APENAS PARA ATRIBUTOS NOMINAIS
                sdr[att] = self.sdr(self.x, self.y, att)
            att = np.argmax(sdr)
            values = xvalues[att]

            sdr = sdr[att]
            if sdr < SDRMIN:
                self.root = np.mean(self.y)  # valor mais comum
            else:
                self.root = None
                self.children = {}
                for v in values:
                    self.children[v] = RegressionTree(self.nclasses)

                # true if attribute is discrete (string)
                if isinstance(self.x[0, att], basestring):
                    self.att = att
                    x = self.x
                    y = self.y
                    del self.x
                    del self.y
                    for v in values:
                        ind = np.where(x[:, att] == v)
                        if len(ind[0]) != 0:
                            ind = ind[0]
                            self.children[v].train(
                                x[ind, :], y[ind, 0], SDRMIN, NMIN, xvalues)
                        else:
                            # valor mais comum
                            self.children[v].root = np.mean(y)
                else:
                    # TODO: variaveis continuas
                    pass

    def gerar_grafo(self, g=nx.Graph(), pos={}, idnode=0, x=0, y=0):

        filhos = self.children.keys()
        deltax = np.linspace(-np.power(4.0, 10 + y),
                             np.power(4.0, 10 + y), len(filhos))
        j = 1

        pos[idnode] = [x, y]
        for thr in filhos:
            if self.children[thr].root is None:
                # liga um no a outro
                g.add_edge(idnode, 10 * idnode + j)
                # da nome a ligacao
                g.edge[idnode][10 * idnode + j]['value'] = thr
                g, pos = self.children[thr].gerar_grafo(
                    g, pos, 10 * idnode + j, x + deltax[j - 1], y - 1)
                j = j + 1
            else:
                # liga um no a outro
                g.add_edge(idnode, 10 * idnode + j)
                # da nome a ligacao
                g.edge[idnode][10 * idnode + j]['value'] = thr
                # da nome ao no folha
                g.node[10 * idnode + j]['name'] = str(self.children[thr].root)

                pos[10 * idnode + j] = [x + deltax[j - 1], y - 1]
                j = j + 1
        # da nome ao no
        g.node[idnode]['name'] = str(self.att)
        return g, pos

    def estimate(self, x):

        self.x = np.array(x)
        n = self.x.shape[0]
        self.yhat = np.array([])
        for i in range(0, n):
            self.yhat = np.append(self.yhat, self.estimate_one(self.x[i, :]))
        return self.yhat

    def estimate_one(self, x):

        if self.root is None:
            for v in self.children.keys():
                if x[self.att] == v:
                    return self.children[v].estimate_one(x)
        else:
            if self.root is None:
                print 'aaaaaaaahhhhhhhh'
            return self.root

    def sdr(self, x, y, att):
        '''
        Calcula a reducao do desvio padrao.
        Entradas:
            x: matriz de exemplos.
            y: vetor de classes/saidas.
            att: atributo a ser testado.
            thr: se atributo for continuo, limiar de separacao.
        '''
        values = np.union1d(x[:, att], x[:, att])
        # true if attribute is discrete (string)
        if isinstance(x[0, att], basestring):
            sdr = np.std(y)
            for v in values:
                sdr = sdr - \
                    utils.simple_probability(
                        x, att, np.equal, v) * np.std(y[x[:, att] == v])
        else:
            sdr = 0
        return sdr
