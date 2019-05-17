import numpy as np

from interpretability import WebotsRunnablePatch
from opensimplex import OpenSimplex

class RandomTerrain(WebotsRunnablePatch):
    def __init__(self, patch_size, feature_size=24, scale=10):
        super().__init__(patch_size)
        self.feature_size = feature_size
        self.scale = scale
        self.simplex = OpenSimplex()

    def __call__(self):
        h, w = self.hm.shape[0], self.hm.shape[1]
        for y in range(0, h):
            for x in range(0, w):
                value = self.simplex.noise2d(x / self.feature_size, y / self.feature_size)
                self.hm[x, y] += value / self.scale
        return self

    def __iadd__(self, p):
        self.hm += p.hm
        return self

def elevate(patch, strenght=3, mode='linear'):
    factors = np.linspace(22, strenght, patch.hm.shape[0])
    if mode == 'pow': factors = factors ** 2
    patch.hm = patch.hm + factors


def reduce(patch, factor=3, shuffle=False):
    factors = np.linspace(1, factor, patch.hm.shape[0])
    if shuffle:  np.random.shuffle(factors)
    patch.hm = patch.hm / factors
    return patch


def get_rocks(shape):
    rocky1 = RandomTerrain(shape, feature_size=5, scale=15)()
    rocky2 = RandomTerrain(shape, feature_size=10, scale=15)()
    rocky3 = RandomTerrain(shape, feature_size=25, scale=15)()

    return [reduce(rocky1, 5).hm.T, reduce(rocky2, 3).hm.T, reduce(rocky3, 3).hm.T]