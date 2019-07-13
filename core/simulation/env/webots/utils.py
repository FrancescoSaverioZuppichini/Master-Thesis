import numpy as np
import time
import matplotlib.pyplot as plt
import seaborn as sns
from tempfile import NamedTemporaryFile


def image2webots_terrain(image, src_world, config, output_path=None):

    height, resolution = config['height'], config['resolution']
    image = image.astype(np.float)
    image = image / 255
    terrain = image * height

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


    return output_path


