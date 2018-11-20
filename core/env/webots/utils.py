import numpy as np
import time
import matplotlib.pyplot as plt

from matplotlib.pyplot import imshow
from tempfile import NamedTemporaryFile


def plot_terrain(terrain):
    imgplot = imshow(terrain)
    plt.colorbar()
    plt.show()

def image2webots_terrain(image, src_world, config, output_path=None, verbose=False):
    if image.dtype == 'uint8':
        image = image / 256.
    if image.dtype == 'uint16':
        image = image / 65536.
    if image.dtype == 'uint32':
        image = image / 4294967296.

    height, resolution = config['height'], config['resolution']

    terrain = image * height

    if verbose: print('mod image type: ', terrain.dtype, ' height factor: ', height, ' max val (m): ',
                           np.amax(terrain), ' shape', terrain.shape)

    if verbose: plot_terrain(terrain)

    data = []
    l1 = l2 = -1
    with open(src_world) as f:
        for i, line in enumerate(f):
            data.append(line)
            if line.find('      geometry DEF EL_GRID ElevationGrid {') != -1:
                l1 = i + 1
            if line.find('name "terrain"'):
                l2 = i

    terrain_flatted = terrain.reshape((-1))
    np.set_printoptions(threshold=terrain.shape[0] * terrain.shape[1])  # default

    data[l2 - 9] = '        ]\n'
    data[l2 - 8] = '        xDimension ' + str(terrain.shape[0]) + '\n'
    data[l2 - 7] = '        xSpacing ' + str(resolution) + '\n'
    data[l2 - 6] = '        zDimension ' + str(terrain.shape[1]) + '\n'
    data[l2 - 5] = '        zSpacing ' + str(resolution) + '\n } \n'

    start = time.time()

    # take it down to zero
    terrain_flatted -= np.max(terrain_flatted)

    if not output_path:
        f = NamedTemporaryFile(mode='w')
        output_path = f.name

    with open(output_path, 'w') as f:
        for line in data[0:l1]:
            f.write(line)
        f.write('height [')
        #       use .tofile since it is super fast!
        terrain_flatted.tofile(f, sep=' ')

        for line in data[l2 - 9:]:
            f.write(line)

    end = time.time()

    if verbose: print('Wrote new world in {:.2f}s'.format((end - start)))

    return output_path


