import numpy as np
import cv2
import seaborn as sns
import matplotlib.pyplot as plt

from skimage.util.shape import view_as_windows
from sklearn.cluster import KMeans

class FlatGroundSpawnStrategy():
    """
    This class finds the position in an height map in which there are no obstacles.
    It extract patches using a rolling window in the image and for each patch
    - it centers the patch value at zero in the center to remove the height factor
    - it checks if the absolute  value of the mean value of the pixel in that patch is less than a set threshold,
      if  so the patch position is added to a buffer.
    Later on, the buffer is processed by a k-means in order to extract k spawn points.
    """
    def __init__(self, hm_path, debug=False,  scale=1):
        self.hm = cv2.imread(hm_path)
        self.hm = cv2.cvtColor(self.hm, cv2.COLOR_BGR2GRAY)
        self.hm  = self.hm .astype(np.float32)
        self.debug = debug
        self.hm *= scale

    def center_patch(self, patch):
        w, h = patch.shape
        patch = patch.astype(np.float32)
        center = patch[w // 2, h // 2]
        patch -= center

        return patch

    def show_spawn_pos(self, positions, size):
        fig = plt.figure()
        ax = plt.subplot(1, 1, 1)
        sns.heatmap(self.hm, ax=ax)

        for pos in positions:
            x, y = pos
            ax.plot(x, y, marker='o', color='r', ls='', label='finish')

    def find_spawn_points(self, size=40, step=2, tol=1e-3):
        positions = []
        patches = view_as_windows(self.hm, (size, size), step)
        y = 0
        for row in patches:
            x = 0
            for patch in row:
                patch = self.center_patch(patch)
                if np.abs(patch.mean()) < tol:  positions.append((x + size // 2, y  + size // 2))

                x += step
            y += step

        return positions

    def reduce_positions_by_clustering(self, positions, k=100):
        k = len(positions) if len(positions) < k else k
        X = np.array(positions)
        self.estimator = KMeans(n_clusters=k)
        self.estimator.fit(X)

        clusters2points = {i: X[np.where(self.estimator.labels_ == i)] for i in range(self.estimator.n_clusters)}
        new_positions = [clusters2points[key][len(item) // 2] for key, item in clusters2points.items()]

        return new_positions

    def __call__(self, k=100, size=40, *args, **kwargs):
        positions = self.find_spawn_points(size=size, *args, **kwargs)
        print(len(positions))
        if self.debug: self.show_spawn_pos(positions, size)
        new_positions = self.reduce_positions_by_clustering(positions, k=k)
        if self.debug: self.show_spawn_pos(new_positions, size)

        return new_positions
