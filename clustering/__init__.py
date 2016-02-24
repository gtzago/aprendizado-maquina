import matplotlib.pyplot as plt
import numpy as np
import utils
import numpy.matlib as mat


class Hierarchical(object):

    def __init__(self, x):
        self.x = x
        self.m = x.shape[1]
        self.n = x.shape[0]

    def diana(self, NC):

        self.__calculate_distances()

        self.z = []
        self.z.append(range(0, self.n))

        self.nc = 1

        while self.nc < NC:
            self.__split_clusters()

    def __split_clusters(self):
        self.__create_new_clusters()

    def __create_new_clusters(self):

        diameter_clusters = []

        for i in range(0, self.nc):
            aux = self.d[:, self.z[i]]
            diameter = np.sum(aux, 1) / (aux.shape[1] - 1)
            diameter_clusters.append(diameter[self.z[i]])

        flag = True
        for i in range(0, self.nc):
            if flag:
                max = np.max(diameter_clusters[i])
                chosen_cluster = i
                flag = False
            elif np.max(diameter_clusters[i]) > max:
                max = np.max(diameter_clusters[i])
                chosen_cluster = i

        chosen_ex = self.z[chosen_cluster][
            np.argmax(diameter_clusters[chosen_cluster])]
        self.z.append([chosen_ex])
        del self.z[chosen_cluster][
            np.argmax(diameter_clusters[chosen_cluster])]
        self.nc = self.nc + 1

        # AS DISTANCIAS DEVEM SER APENAS ATE O CLUSTER ANTIGO

        # distancias ate os elementos fora do novo cluster (mas dentro do
        # cluster do qual sairam.

        #aux1 = np.delete(self.d, self.z[self.nc - 1], 1)
        aux1 = self.d[:, self.z[chosen_cluster]]

        # distancias ate os elementos do novo cluster.
        aux2 = self.d[:, self.z[self.nc - 1]]
        d = np.sum(aux1, 1) / \
            (aux1.shape[1] - 1) - np.sum(aux2, 1) / (aux2.shape[1])
        ind = list(np.where(d > 0)[0])

        # uso apenas os elementos do cluster original.
        ind = list(set(ind).intersection(set(self.z[chosen_cluster])))

        self.z[self.nc - 1] = self.z[self.nc - 1] + ind
        # elimina elementos repetidos
        self.z[self.nc - 1] = list(set(self.z[self.nc - 1]))

        for i in range(0, self.nc - 1):
            # retiro os elementos que mudaram de cluster
            self.z[i] = list(set(self.z[i]) - set(self.z[self.nc - 1]))

    def agnes(self, NCMAX):

        self.__calculate_distances()

        self.z = []
        for i in range(0, self.n):
            self.z.append([i])
        self.nc = self.n

        while self.nc > NCMAX:
            self.__distance_cluster_all()
            self.__group_clusters()

    def __group_clusters(self):
        d = np.copy(self.d_cluster)

        # aumenta a diagonal prinicpal.
        # d = d - np.max(self.d) * np.eye(self.nc, self.nc)

        (inda, indb) = np.unravel_index([np.argmin(d)], (self.nc, self.nc))

        # nao eh soma, pois estou trabalhando com lista.
        self.z[inda] = self.z[inda] + self.z[indb]
        del self.z[indb]
        self.nc = self.nc - 1

    def __calculate_distances(self):
        self.d = np.zeros((self.n, self.n))

        for i in range(0, self.n):
            self.d[i, :] = self.__euclidian_distance_several_to_one(
                self.x, self.x[i, :])

    def __distance_cluster_all(self):

        self.d_cluster = 2 * np.max(self.d) * np.ones((self.nc, self.nc))
        for i in range(0, self.nc):
            for j in range(i + 1, self.nc):
                self.d_cluster[i, j] = self.__distance_cluster_one_to_one(
                    self.z[i], self.z[j])

    def __distance_cluster_one_to_one(self, cluster1, cluster2):

        #         # utilizando a distancia media de exemplo a exemplo.
        #         n1 = len(cluster1)
        #         n2 = len(cluster2)
        #         d = np.array([])
        #         for i in range(0, n1):
        #             for j in range(0, n2):
        #                 d = np.append(d, self.d[cluster1[i], cluster2[j]])
        #         return np.mean(d)

        # utilizando a distancia minima entre dois clusters.
        n1 = len(cluster1)
        n2 = len(cluster2)
        d = np.array([])
        for i in range(0, n1):
            for j in range(0, n2):
                d = np.append(d, self.d[cluster1[i], cluster2[j]])
        return np.min(d)

    def __euclidian_distance_several_to_one(self, x, y):
        x = np.reshape(x, (-1, self.m))
        y = np.reshape(y, (-1, self.m))
        d = np.sum(np.power(x - y, 2), 1)
        return d

    def scatter_plot(self):

        c = np.zeros(self.n)
        for i in range(0, self.n):
            for j in range(0, self.nc):
                if i in self.z[j]:
                    c[i] = j

        area = np.pi * (10.0)**2  # 0 to 15 point radiuses
        f = plt.scatter(self.x[:, 0], self.x[:, 1], s=area, c=c, alpha=0.5)
        plt.title('Scatter plot - Hierarchical - k = {:d}'.format(self.nc))

        return f

    def calculate_accuracy(self, y):

        acc = 0
        for i in range(0, self.nc):
            c = utils.mode(y[self.z[i]])
            yhat = mat.repmat(c, len(self.z[i]), 1)
            acc = acc + len(self.z[i]) * utils.accuracy(y[self.z[i]], yhat)
        acc = acc / self.n

        return acc


class KMeans(object):

    def __init__(self, x):
        self.x = x
        self.m = x.shape[1]
        self.n = x.shape[0]

    def scatter_plot(self):

        colors = np.random.rand(self.k)

        area = np.pi * (10.0)**2  # 0 to 15 point radiuses
        f = plt.Figure
        plt.scatter(self.x[:, 0], self.x[:, 1], s=area, c=self.c, alpha=0.5)
        plt.title('Scatter plot - Kmeans - k = {:d}'.format(self.k))
        return f

    def train(self, k, MAXITER=1000, MINDIST=0.001, centroid=None):
        self.k = k
        self.c = np.zeros(self.n)
        if centroid is None:
            self.__initialize_centroids()

        self.__associate_all()
        for i in range(0, MAXITER):
            #             print 'iteracao {:d}'.format(i)
            c_ant = np.copy(self.c)
            self.__calculate_centroid()
            self.__associate_all()

            if np.equal(c_ant, self.c).all():
                break

    def __calculate_centroid(self):

        for i in range(0, self.k):
            self.centroid[i, :] = np.mean(self.x[self.c == i, :], 0)

    def __associate_all(self):

        for i in range(0, self.n):
            self.c[i] = self.__associate_one(self.x[i, :], self.centroid)

    def __associate_one(self, x, centroid):

        c = np.argmin(self.__euclidian_distance_several_to_one(x, centroid))

        return c

    def __euclidian_distance_one_to_one(self, x, y):

        d = np.sqrt(np.sum(np.power(x - y, 2)))

        return d

    def __euclidian_distance_several_to_one(self, x, y):
        n = np.min(x.shape)

        x = np.reshape(x, (-1, n))
        y = np.reshape(y, (-1, n))

        d = np.sum(np.power(x - y, 2), 1)
        return d

    def __initialize_centroids(self):

        self.centroid = np.zeros((self.k, self.m))  # matrix with means

        self.centroid[0, :] = self.x[np.random.randint(0, self.n), :]

        for i in range(1, self.k):
            d = np.zeros(self.n)
            for j in range(0, i):
                d = d + self.__euclidian_distance_several_to_one(
                    self.centroid[j, :], self.x)
            self.centroid[i, :] = self.x[np.argmax(d), :]

    def calculate_accuracy(self, y):

        acc = 0
        for i in range(0, self.k):
            ind = np.where(self.c == i)[0]
            c = utils.mode(y[ind])
            yhat = mat.repmat(c, ind.size, 1)
            acc = acc + ind.size * utils.accuracy(y[ind], yhat)
        acc = acc / self.n

        return acc
