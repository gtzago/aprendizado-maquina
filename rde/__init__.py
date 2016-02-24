import numpy as np


class RecursiveDensityEstimation(object):

    def __init__(self, thr=None):
        self.k = 0
        self.mean = 0
        self.density = 0
        self.var = 0
        self.mean_density = 0

        self.caps_x = 0

        if thr is None:
            self.thr = 2

    def insert(self, x):
        self.k = self.k + 1
        self.mean = self.mean * (self.k - 1) / self.k + x / self.k
        self.caps_x = self.caps_x * \
            (self.k - 1) / self.k + self.norm2squared(x) / self.k

        self.density = 1.0 / \
            (1 + self.norm2squared(x - self.mean) +
             self.caps_x - self.norm2squared(self.mean))

        self.mean_density = self.mean_density * \
            (self.k - 1) / self.k + self.density / self.k

        self.var = self.var * \
            (self.k - 1) / self.k + \
            np.power(self.density - self.mean_density, 2) / self.k

        if np.abs(self.density - self.mean_density) > self.thr * np.sqrt(self.var):
            return True
        else:
            return False

    def norm2(self, x):
        return np.sqrt(np.sum(np.power(x, 2)))

    def norm2squared(self, x):
        return np.sum(np.power(x, 2))
