import numpy as np
import mayavi.mlab as mlab
from utilities.patches import Patch

class Mayavi3dPlottable():

    def plot3d_mayavi(self, pixelsize, save_path=None, size=(1541, 1281)):
        fig = mlab.figure(size=size)

        fig.scene.background = (1, 1, 1)

        y, x = np.meshgrid(np.arange(self.hm.shape[0]) * pixelsize, np.arange(self.hm.shape[1]) * pixelsize)
        s = mlab.mesh(x, y, self.hm, colormap='viridis')

        s.actor.property.interpolation = 'phong'
        s.actor.property.specular = 0.0
        s.actor.property.specular_power = 10
        s.actor.property.ambient_color = (1, 1, 1)
        s.actor.property.diffuse_color = (0.7, 0.7, 0.7)
        s.actor.property.color = (0.7, 0.7, 0.7)
        s.actor.property.ambient = 0.02
        mlab.view(azimuth=-90, elevation=45)

        if save_path:
            mlab.savefig(save_path)
            mlab.close(fig)

        return fig

class MyPatch(Patch, Mayavi3dPlottable):
    pass

from utilities.postprocessing.handlers.functional import read_image
from skimage.io import imread

hm = read_image('/home/francesco/Documents/Master-Thesis/core/maps/new-train/bars1.png').astype(np.float32) / 255



p = MyPatch.from_hm(hm)
fig = p.plot3d_mayavi(0.02, '/home/francesco/Desktop/wee.png')
