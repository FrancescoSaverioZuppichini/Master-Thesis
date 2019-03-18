import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

class Patch():
    def __init__(self, size):
        self.hm = np.zeros(size)

    def __call__(self, *args, **kwargs):
        self.hm = self.make(*args, **kwargs)
        return self.hm

    def make(self, *args, **kwargs):
        raise NotImplementedError

    def plot2d(self):
        fig = plt.figure()
        sns.heatmap(self.hm)
        plt.show()

class BarKrockPatch(Patch):
    def make(self, *args, **kwargs):
        pass


p = Patch((92,92))
p.plot2d()