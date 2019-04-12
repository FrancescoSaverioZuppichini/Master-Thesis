import cv2

import numpy as np
from mpl_toolkits.mplot3d import \
    Axes3D  # This import has side effects required for the kwarg projection='3d' in the call to fig.add_subplot
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from collections.abc import Iterable


class Patch():
    def __init__(self, shape, *args, **kwargs):
        self.hm = np.zeros(shape)
        self.shape = shape
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
        return (self.hm - min) / (max - min)

    def plot3d(self, title=None, texture=None):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        X, Y = np.meshgrid(range(self.hm.shape[0]), range(self.hm.shape[1]))

        ax.set_zlim3d(-1, 1)
        # TODO we should show the texture
        # colours = plt.cm.viridis(self.texture)

        # I have to transpose the heightmap to correctly show it -> I am not sure why
        surf = ax.plot_surface(X, Y, self.hm.T,
                               # facecolors=colours,
                               cmap=plt.cm.viridis,
                               linewidth=0.2)

        # import matplotlib.cm as cm
        # m = cm.ScalarMappable(cmap=plt.cm.viridis)
        # m.set_array(self.hm.T)
        # fig.colorbar(m)
        fig.colorbar(surf, shrink=0.5, aspect=5)

        title = title if title is not None else self.__repr__()
        plt.title(title)

        plt.show()

    @classmethod
    def from_hm(cls, hm):
        p = cls(hm.shape)
        p.hm = hm
        return p

    @classmethod
    def from_range(cls, shape, **kwargs):
        patches = []

        static_fields = {k: v for k, v in kwargs.items() if not isinstance(v, Iterable)}

        for key, range in kwargs.items():
            if isinstance(range, Iterable):
                for value in range:
                    p = cls(shape, **static_fields, **{key: value})
                    p()
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
        cv2.imwrite(out_path, self.to_gray())

    def add_texture(self, tex):
        self.texture = Patch.from_hm(tex)

    def shift(self, px):
        self.hm = np.roll(self.hm, px)

    def to_gray(self):
        return (self.hm * 255).astype(np.uint8)

    def __repr__(self):
        return "Shape = {}".format(self.shape)


class BarPatch(Patch):
    def __init__(self, shape, offset=16, size=4, strength=1, up=True, down=True):
        super().__init__(shape)
        self.offset, self.size, self.strength = offset, size, strength
        self.up, self.down = up, down

    def make(self):
        if self.up: self.hm[self.offset: self.offset + self.size, :] = self.strength
        if self.down: self.hm[-self.offset - self.size: -self.offset:, :] = self.strength

        return self.hm

    @property
    def between(self):
        from_border = self.offset + self.size
        between = self.shape[0] - (from_border) * 2
        return between

    def __repr__(self):

        return "{} Offset = {} Size = {} Strength = {} Between = {}".format(super().__repr__(),
                                                                            self.offset,
                                                                            self.size,
                                                                            self.strength,
                                                                            self.between)


class WallPatch(BarPatch):
    def __init__(self, shape, front=True, back=True, *args, **kwargs):
        super().__init__(shape, up=back, down=front, *args, **kwargs)

    def make(self, *args, **kwargs):
        super().make(*args, **kwargs)
        self.hm = self.hm.T
        return self.hm


class BumpsPatch(Patch):
    def __init__(self, shape, resolution=(4, 4), size=(1, 1), strength=1):
        super().__init__(shape)
        self.resolution, self.size, self.strength = resolution, size, strength

    def make(self):
        self.hm = cv2.resize(self.hm, self.resolution, interpolation=cv2.INTER_LINEAR)

        self.hm[self.hm.shape[0] // 2 - self.size[0]:
                self.hm.shape[0] // 2 + self.size[0], - self.size[1]:] = self.strength

        self.hm = cv2.resize(self.hm, self.shape)

        return self.hm


class HolesPatch(BumpsPatch):
    def make(self):
        self.hm = -1 * super().make()

        return self.hm


class RampPatch(BumpsPatch):
    def __init__(self, shape, strength=1, orientation=-1):
        super().__init__(shape, strength=strength, size=(2, 1), resolution=(2, 1))
        self.orientation = orientation

    def make(self):
        self.hm = self.orientation * super().make()
        return self.hm
