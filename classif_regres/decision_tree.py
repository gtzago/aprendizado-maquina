import numpy as np
import utils
import reduc_dim
import classif_regres
import networkx as nx


class DecisionTree(object):

    def __init__(self, nclasses):
        self.root = None
        self.children = None
        self.att = None
        self.n = 0
        self.m = 0
        self.nclasses = nclasses

    def train(self, x, y, IGMIN=0.05, NMIN=0):
        self.x = np.array(x)
        self.y = np.reshape(y, (-1, 1))
        self.n = self.y.size
        self.m = self.x.shape[1]

        # MAXIMIZAR ATRAVES DO IG!!!!!!
        # att = reduc_dim.sequential_forward_selection(
        # self.x, self.y, self.x, self.y, utils.accuracy, 1,
        # classif_regres.knn, 3)[0]

        if np.union1d(y, y).size == 1 or self.n < NMIN:
            self.root = utils.mode(self.y)
        else:
            ig = np.zeros(self.m)
            for att in range(0, self.m):
                # APENAS PARA ATRIBUTOS NOMINAIS
                ig[att] = self.ig(self.x, self.y, att, 0)
            att = np.argmax(ig)
            values = np.union1d(self.x[:, att], self.x[:, att])

    #         if isinstance(x[0, att], basestring):
    #             thr = utils.mode(self.x[:, att])
    #         else:
    #             # TODO: implementar funcao que encontre o limiar otimo
    #             # de separacao e coloca-lo dentro da funcao.
    #             '''
    #             Os valores dos atributos sao primeiro ordenados;
    #
    #             O ponto medio entre dois valores consecutivos eh
    #             um possivel ponto de corte e eh avaliado pela
    #             funcao merito;
    #
    #             O possivel ponto de corte que maximiza a funcao
    #             merito eh escolhido.
    #             '''
    #             thr = np.mean(self.x[:, att])
    #         ig = self.ig(self.x, self.y, att, thr)

            ig = ig[att]
            if ig < IGMIN:
                self.root = utils.mode(y)  # valor mais comum
            else:
                self.root = None
                self.children = {}
                for v in values:
                    self.children[v] = DecisionTree(self.nclasses)

                # true if attribute is discrete (string)
                if isinstance(self.x[0, att], basestring):
                    self.att = att
                    x = self.x
                    y = self.y
                    del self.x
                    del self.y
                    for v in values:
                        ind = np.core.defchararray.equal(x[:, att], v)
                        self.children[v].train(
                            x[ind, :], y[ind, 0], IGMIN, NMIN)
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

#     def gerar_grafo(self, graph=set(), pos={}, idnode=0, x=0, y=0):
#         '''
#             Gera o grafo que sera utilizado para desenhar a arvore.
#         '''
#
#         filhos = self.children.keys()
#         deltax = np.linspace(-np.power(4.0, 10 + y),
#                              np.power(4.0, 10 + y), len(filhos))
#         j = 1
#         pos[str(idnode) + '_' + str(self.att)] = [x, y]
#         for thr in filhos:
#             if self.children[thr].root is None:
#                 graph.add(
#                     (str(idnode) + '_' + str(self.att), (str(10 * idnode + j) + '_' + str(self.children[thr].att))))
#
#                 graph, pos = self.children[thr].gerar_grafo(
#                     graph, pos, 10 * idnode + j, x + deltax[j - 1], y - 1)
#                 j = j + 1
#             else:
#                 graph.add(
#                     (str(idnode) + '_' + str(self.att), (str(10 * idnode + j) + '_' + str(self.children[thr].root))))
#                 pos[(str(10 * idnode + j) + '_' + str(self.children[thr].root))
#                     ] = [x + deltax[j - 1], y - 1]
#                 j = j + 1
#         return graph, pos

    def classify(self, x):

        self.x = np.array(x)
        n = self.x.shape[0]
        self.yhat = np.array([])
        for i in range(0, n):
            self.yhat = np.append(self.yhat, self.classify_one(self.x[i, :]))
        return self.yhat

    def classify_one(self, x):

        if self.root is None:
            for v in self.children.keys():
                if x[self.att] == v:
                    return self.children[v].classify_one(x)
        else:
            return self.root

    def entropia(self, y):
        '''
            Calcula a entropia.
        '''
        classes = np.union1d(y, y)
        h0 = 0
        for c in classes:
            p = utils.simple_probability(y, 0, np.equal, c)
            if p != 0:
                h0 = h0 - p * np.log(p) / np.log(self.nclasses)
        return h0

    def informacao(self, x, y, att):
        '''
            Calcula a informacao do atributo ATT
        '''
        values = np.union1d(x[:, att], x[:, att])
        e = 0  # entropy
        for v in values:
            e = e + \
                utils.simple_probability(
                    x, att, np.equal, v) * self.entropia(y[x[:, att] == v])
        return e

    def ig(self, x, y, att, thr):
        '''
        Calcula o ganho de informacao.
        Entradas:
            x: matriz de exemplos.
            y: vetor de classes/saidas.
            att: atributo a ser testado.
            thr: se atributo for continuo, limiar de separacao.
        '''
        classes = np.union1d(y, y)
        # true if attribute is discrete (string)
        if isinstance(x[0, att], basestring):
            h = self.entropia(y)
            e = self.informacao(x, y, att)
            ig = h - e
        else:
            h0 = 0
            p = utils.simple_probability(x, att, np.less, thr)
            h0 = h0 - p * np.log2(p) - (1 - p) * np.log2((1 - p))
            h = 0
            for c in classes:
                p = utils.conditional_probability(
                    np.hstack((x, y)), self.m, np.equal, c, att, np.less, thr)
                if p != 0.0 and p != 1.0:
                    h = h - p * np.log2(p) - (1 - p) * np.log2((1 - p))
            ig = h0 - h
        return ig
