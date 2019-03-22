import cv2

import numpy as np
from mpl_toolkits.mplot3d import Axes3D # This import has side effects required for the kwarg projection='3d' in the call to fig.add_subplot
import matplotlib.pyplot as plt
import seaborn as sns

class Patch():
    def __init__(self, size):
        self.hm = np.zeros(size)
        self.size = size

    def __call__(self, *args, **kwargs):
        self.hm = self.make(*args, **kwargs)
        return self.hm

    def make(self, *args, **kwargs):
        raise NotImplementedError

    def plot2d(self):
        fig = plt.figure()
        sns.heatmap(self.hm)
        plt.show()

    def plot3d(self):
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        X,Y = np.meshgrid(range(self.hm.shape[0]), range(self.hm.shape[1]))

        surf = ax.plot_surface(X, Y, self.hm,
                        cmap=plt.cm.viridis,
                        vmax=1.0,
                        linewidth=0.2)
        fig.colorbar(surf, shrink=0.5, aspect=5)

        plt.show()

    @classmethod
    def from_hm(cls, hm):
        p = cls(hm.shape)
        p.hm =hm
        return p
    
    @classmethod
    def from_tensor(cls, tensor):
        hm = tensor.squeeze().cpu().numpy()
        return cls.from_hm(hm)


class BarPatch(Patch):
    def make(self, offset=16, size=4, strength=1):
        self.hm[offset: offset + size, :] = strength
        print(-offset - size, -offset)
        self.hm[-offset - size: -offset:, :] = strength

        return self.hm

class BumpsPatch(Patch):
    def make(self, strength=1.0, resolution=(4,4), size=(1,1)):
        self.hm = cv2.resize(self.hm, resolution, interpolation = cv2.INTER_LANCZOS4)

        self.hm[self.hm.shape[0] //2 - size[0] :
                self.hm.shape[0] //2 + size[0], - size[1] : ] = strength

        self.plot2d()
        self.hm = cv2.resize(self.hm, self.size)

        return self.hm

class HolesPatch(BumpsPatch):
    def make(self, *args, **kwargs):
        self.hm = -1 * super().make(*args, **kwargs)

        return self.hm


if __name__ == '__main__':
#     little test
    p = BarPatch((92,92))
    p(offset=2)
    p.plot2d()
    p.plot3d()
    #
    # p = HolesPatch((92,92))
    # p(strength=0.5)
    # p.plot2d()
    # p.plot3d()

