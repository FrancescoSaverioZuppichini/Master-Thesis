import cv2

import numpy as np
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

class BarPatch(Patch):
    def make(self, offset=16, size=4):
        self.hm[offset: offset + size] = 1
        self.hm[-offset: -offset + size] = 1

        return self.hm

class BumpsPatch(Patch):
    def make(self):
        self.hm = cv2.resize(self.hm, (3,3))

        self.hm[self.hm.shape[0] //2,-1] = 1

        self.hm = cv2.resize(self.hm, self.size)

        return self.hm


class HolesPatch(BumpsPatch):
    def make(self, *args, **kwargs):
        self.hm = -1 * super().make(*args, **kwargs)

        return self.hm


# p = BarPatch((92,92))
# p()
# p.plot2d()

p = BumpsPatch((92,92))
p()
p.plot2d()
