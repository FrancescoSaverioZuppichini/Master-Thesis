import cv2

import numpy as np
from mpl_toolkits.mplot3d import Axes3D # This import has side effects required for the kwarg projection='3d' in the call to fig.add_subplot
import matplotlib.pyplot as plt
import seaborn as sns
from collections.abc import Iterable

class Patch():
    def __init__(self, size):
        self.hm = np.zeros(size)
        self.size = size
        self.texture = None

    def __call__(self, *args, **kwargs):
        self.hm = self.make(*args, **kwargs)
        return self

    def make(self, *args, **kwargs):
        raise NotImplementedError

    def plot2d(self, title=''):
        fig = plt.figure()
        sns.heatmap(self.hm)
        plt.title(title)
        plt.show()

    @property
    def norm(self):
        min, max = self.hm.min(), self.hm.max()
        return (self.hm - min) / ( max - min )

    def plot3d(self, title='', texture=None):
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        X,Y = np.meshgrid(range(self.hm.shape[0]), range(self.hm.shape[1]))

        ax.set_zlim3d(-1, 1)
        # I have to transpose the heightmap to correctly show it -> I am not sure why
        surf = ax.plot_surface(X, Y, self.hm.T,
                        cmap=plt.cm.viridis,
                        linewidth=0.2)

        fig.colorbar(surf, shrink=0.5, aspect=5)
        plt.title(title)

        plt.show()

    @classmethod
    def from_hm(cls, hm):
        p = cls(hm.shape)
        p.hm =hm
        return p

    @classmethod
    def from_range(cls, size,  **kwargs):
        patches = []

        static_fields = {k:v for k,v in kwargs.items() if not isinstance(v, Iterable)}

        for key, range in kwargs.items():
            if isinstance(range, Iterable):
                for value in range:
                    p  = cls(size)
                    p(**static_fields, **{key: value})
                    patches.append(p)
        return patches

    @classmethod
    def from_tensor(cls, tensor):
        hm = tensor.squeeze().cpu().numpy()
        return cls.from_hm(hm)

    @classmethod
    def from_tensors(cls, tensors):
        return [Patch.from_tensor(tensor) for tensor in tensors]

    @classmethod
    def from_hms(cls, hms):
        return [Patch.from_hm(hm) for hm in hms]

    def store(self, out_path):
        cv2.imwrite(self.hm, out_path)

    def add_texture(self, tex):
        self.texture = Patch.from_hm(tex)

class BarPatch(Patch):
    def make(self, offset=16, size=4, strength=1, up=True, down=True):
        if up: self.hm[offset: offset + size, :] = strength
        if down: self.hm[-offset - size: -offset:, :] = strength

        return self.hm

class WallPatch(BarPatch):
    def make(self, front=True, back=True, *args, **kwargs):
        super().make(up=back, down=front, *args, **kwargs)
        self.hm = self.hm.T

        return self.hm

class BumpsPatch(Patch):
    def make(self, strength=1.0, resolution=(4,4), size=(1,1)):
        self.hm = cv2.resize(self.hm, resolution, interpolation = cv2.INTER_LINEAR)

        self.hm[self.hm.shape[0] //2 - size[0] :
                self.hm.shape[0] //2 + size[0], - size[1] : ] = strength

        self.hm = cv2.resize(self.hm, self.size)

        return self.hm

class HolesPatch(BumpsPatch):
    def make(self, *args, **kwargs):
        self.hm = -1 * super().make(*args, **kwargs)

        return self.hm

class RampPatch(BumpsPatch):
    def make(self, strength=1, orientation=- 1):
        self.hm = orientation * super().make(strength=strength,  size=(2,1), resolution=(2,1))
        return self.hm


if __name__ == '__main__':
    patches = WallPatch.from_range(size=(88,88), offset=list(range(2)))
    print(patches)
    for p in patches:
        p.plot2d()
    # p(back=False, offset=18, size=20)
    # p.plot2d()
    # p.plot3d()

