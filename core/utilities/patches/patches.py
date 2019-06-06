import cv2

import numpy as np
from mpl_toolkits.mplot3d import \
    Axes3D  # This import has side effects required for the kwarg projection='3d' in the call to fig.add_subplot
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from functools import partialmethod, partial

from collections.abc import Iterable
import mayavi.mlab as mlab
class Mayavi3dPlottable():

    def setup_scene(self, s):
        s.actor.property.interpolation = 'phong'
        s.actor.property.specular = 0.0
        s.actor.property.specular_power = 10
        s.actor.property.ambient_color = (1, 1, 1)
        s.actor.property.diffuse_color = (0.7, 0.7, 0.7)
        s.actor.property.color = (0.7, 0.7, 0.7)
        s.actor.property.ambient = 0.02

    def plot3d_mayavi(self, pixelsize, save_path=None, size=(1000, 1000), azimuth=45, elevation=45, distance=25,
                      mesh=False, colormap=None, color=(1.0, 1.0, 1.0), *args, **kwargs):
        fig = mlab.figure(size=size)
        fig.scene.background = (1, 1, 1)

        y, x = np.meshgrid(np.arange(self.hm.shape[0]) * pixelsize, np.arange(self.hm.shape[1]) * pixelsize)

        if mesh:
            if colormap is None:
                s = mlab.mesh(x, y, self.hm, color=color, *args, **kwargs)
            else:
                s = mlab.mesh(x, y, self.hm, colormap=colormap, *args, **kwargs)

        else:
            s = mlab.surf(x, y, self.hm, color=(1.0, 1.0, 1.0))

        self.setup_scene(s)

        mlab.view(azimuth=azimuth, elevation=elevation, distance=distance)

        if save_path:
            mlab.savefig(save_path)

        mlab.close(fig)
        return fig

class Patch(Mayavi3dPlottable):
    """
    Based class for defining a custom patch.
    When the class is instantiated, it creates a 'tabula rasa'
    flat patch with size = shape.
    The patch is built when the Patch.__call__() method is called,
    internally it calls the .make method that MUST be implemented.
    """

    def __init__(self, shape, *args, **kwargs):
        self.hm = np.zeros(shape)
        self.shape = shape
        self.texture = None

    def __call__(self, *args, **kwargs):
        self.hm = self.make(*args, **kwargs)
        return self

    def make(self, *args, **kwargs):
        raise NotImplementedError

    @property
    def title(self):
        return ''

    def _plot2d_ax(self, *args, **kwargs):
        return sns.heatmap(self.hm, cmap=plt.cm.viridis, *args, **kwargs)

    def plot2d(self, title='', vmax=1, *args, **kwargs):
        fig = plt.figure()
        self._plot2d_ax(vmax=vmax, *args, **kwargs)
        plt.title(self.title)

        return fig

    @property
    def norm(self):
        min, max = self.hm.min(), self.hm.max()
        return (self.hm - min) / (max - min)

    def plot3d(self, title=None, colorbar=True, rstride=2, cstride=2, *args, **kwargs):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        X, Y = np.meshgrid(range(self.hm.shape[0]), range(self.hm.shape[1]))

        ax.set_zlim3d(min(np.min(self.hm), -1), max(np.max(self.hm), 1))
        # ax.set_zlim3d = (-1, 2)

        surf = ax.plot_surface(X, Y, self.hm.T,
                               rstride=rstride,
                               cstride=cstride,
                               vmax=1,
                               cmap=plt.cm.viridis,
                               *args, **kwargs,
                               linewidth=0.1)
        ax.view_init(azim=-20)

        if colorbar: fig.colorbar(surf)

        title = title if title is not None else self.title
        plt.title(title)

        return fig, ax

    @classmethod
    def from_hm(cls, hm, *args, **kwargs):
        p = cls(hm.shape, *args, **kwargs)
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
    def from_tensor(cls, tensor, *args, **kwargs):
        hm = tensor.squeeze().cpu().numpy()
        return cls.from_hm(hm, *args, **kwargs)

    @classmethod
    def from_tensors(cls, tensors):
        return [Patch.from_tensor(tensor) for tensor in tensors]

    @classmethod
    def from_hms(cls, hms):
        return [Patch.from_hm(hm) for hm in hms]

    def save(self, out_path):
        cv2.imwrite(out_path, self.to_gray())

    def add_texture(self, tex):
        self.texture = Patch.from_hm(tex)

    def shift(self, px):
        self.hm = np.roll(self.hm, px)

    def to_gray(self):
        return (self.hm * 255).astype(np.uint8)

    @classmethod
    def from_path(cls, path):
        hm = cv2.imread(path)
        hm = cv2.cvtColor(hm, cv2.COLOR_BGR2GRAY)
        p = cls(hm.shape)
        p.hm = hm / 255
        return p

    @staticmethod
    def plot_all_2d(patches):
        return [p.plot2d() for p in patches]

    @staticmethod
    def plot_all_3d(patches, *args, **kwargs):
        return [p.plot3d(*args, **kwargs) for p in patches]

    def __repr__(self):
        return "Shape = {}".format(self.shape)


class BarPatch(Patch):
    def __init__(self, shape, offset=16, size=4, strength=1, up=True, down=True):
        super().__init__(shape)
        self.offset, self.size, self.strength = offset, size, strength
        self.up, self.down = up, down

    def make(self):
        if self.up:
            self.hm[self.offset: self.offset + self.size, :] = self.strength
        if self.down:
            self.hm[-self.offset - self.size: -self.offset:, :] = self.strength

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


class WallPatch(Patch):
    def __init__(self, shape, front=True, back=True, strength=1, offset=8, size=4):
        super().__init__(shape)
        self.back = back
        self.front = front
        self.strength = strength
        self.offset = offset
        self.size = size

    def make(self):
        if self.back:
            self.hm[:, self.offset: self.offset + self.size] = self.strength
        if self.front:
            self.hm[:, -self.offset - self.size: -self.offset] = self.strength

        self.hm = self.hm
        return self.hm

    @property
    def title(self):
        return 'height = {}, offset = {}'.format(self.strength, (self.offset * 2) / 100)


class BumpsPatch(Patch):
    def __init__(self, shape, resolution=(4, 4), size=(1, 1), strength=1):
        super().__init__(shape)
        self.resolution, self.size, self.strength = resolution, size, strength

    def make(self):
        self.hm = cv2.resize(self.hm, self.resolution,
                             interpolation=cv2.INTER_LINEAR)

        self.hm[self.hm.shape[0] // 2 - self.size[0]:
                self.hm.shape[0] // 2 + self.size[0], - self.size[1]:] = self.strength

        self.hm = cv2.resize(self.hm, self.shape)

        return self.hm


class HolesPatch(BumpsPatch):
    def make(self):
        self.hm = -1 * super().make()

        return self.hm


class RampPatch(Patch):
    def __init__(self, shape, strenght=1, direction='up', factors_func = lambda x: x):
        super().__init__(shape)
        self.strenght = strenght
        self.direction = direction
        self.factors = factors_func(np.linspace(0, self.strenght, self.hm.shape[0]))


    def make(self, factors_func = lambda x: x):
        if self.direction == 'down': self.factors *= -1

        self.hm = self.hm + self.factors
        return self.hm


class HeatMapShowable():
    def heatmap(self, size=(16, 16)):
        hm = cv2.resize(self.hm, size)
        p = Patch.from_hm(hm)
        p._plot2d_ax = partial(p._plot2d_ax, annot=True, fmt=".2f")
        return p


