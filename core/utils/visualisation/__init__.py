import matplotlib.lines as mlines

import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as mpatches

from utils.visualisation import *
from utils.visualisation import *
from utils.postprocessing.utils import *
from matplotlib import gridspec

class VisualiseSimulation():
    """
    This class shows different features of one or more dataframes.
    """
    def __init__(self, hm, patch_size=100):
        self.hm = hm
        self.patch_size = patch_size

    @property
    def hm_ax(self):
        fig = plt.figure()
        ax = sns.heatmap(self.hm)
        return ax

    def show_map(self, hm):
        self.hm_ax
        plt.show()

    def show_traces(self, dfs, ax=None):
        if ax is None: ax = self.hm_ax

        start_marker = mlines.Line2D([], [], marker='o', color='g', ls='', label='start')
        finish_marker = mlines.Line2D([], [], marker='o', color='r', ls='', label='finish')

        for df in dfs:
            initial_pos = df.hm_x.iloc[0], df.hm_y.iloc[0]
            last_position = df.hm_x.iloc[-1], df.hm_y.iloc[-1]
            ax.plot(*initial_pos, marker='o', color='g', ls='', label='start')
            ax.plot(*last_position, marker='o', color='r', ls='', label='finish')
            ax.plot(df.hm_x, df.hm_y, '--', linewidth=2, color='white', label='path')

        ax.legend(handles=[start_marker, finish_marker])

        return ax

    def show_patches(self, df, center=False, n_samples=4, scale=1, random_state=0):
        sample = df.sample(n_samples, random_state=random_state)
        fig, ax = plt.subplots(nrows=n_samples // 2, ncols=n_samples // 2)
        fig.suptitle('patches center={}'.format(center))
        for row in ax:
            for idx, (col, (i, row)) in enumerate(zip(row, sample.iterrows())):
                x, y = row["hm_x"], row["hm_y"]
                patch, _ = hmpatch(self.hm, x, y, np.rad2deg(row['pose__pose_e_orientation_z']), self.patch_size, scale=1)
                patch = patch.astype(np.float32)
                if center: patch = patch - patch[patch.shape[0] // 2, patch.shape[1] // 2]
                col.plot(self.patch_size // 2, self.patch_size // 2, marker='o', color='r', ls='', linewidth=10,
                         label='finish')
                sns.heatmap(patch, ax=col, vmin=0, vmax=1)

        plt.show()

    def show_patch_on_the_map(self, sample):
        fig = plt.figure(figsize=(6, 12))

        x, y, ang, ad = sample["hm_x"], \
                        sample["hm_y"], \
                        sample['pose__pose_e_orientation_z'], \
                        sample["advancement"]

        ax1 = plt.subplot(2, 1, 1)
        ax2 = plt.subplot(2, 1, 2)

        ax1.set_title("advancement: {:.4f}, x={:.0f}, y={:.0f}".format(ad,
                                                                       x,
                                                                       y))
        sns.heatmap(self.hm, ax=ax1, vmin=0, vmax=1)
        rect = mpatches.Rectangle((x - self.patch_size // 2, y - self.patch_sizplot_patch_map_advancement_in_timee // 2), self.patch_size,
                                 self.patch_size, linewidth=1, edgecolor='r', facecolor='none', angle=np.rad2deg(ang))
        ax1.add_patch(rect)

        patch, _ = hmpatch(self.hm, x, y, np.rad2deg(ang), self.patch_size, scale=1)
        sns.heatmap(patch, ax=ax2)

    def show_patches_on_the_map(self, df, n_show=4, title='patches', compress=True):

        fig = plt.figure(figsize=(4 * 2,  4 * 3))
        size = (2 + n_show // 2, n_show // 2)
        gridspec.GridSpec(*size)

        ax_hm = plt.subplot2grid((size), (0, 0), colspan=2, rowspan=2)
        ax_hm.set_title(title)
        sns.heatmap(self.hm / 255, ax=ax_hm, vmin=0, vmax=1)

        self.show_traces([df], ax=ax_hm)

        if compress:
            df = df.loc[list(range(0, len(df), len(df) // n_show)), :]


        hm_patches = []
        for i, row in df.iterrows():
            x, y, ang, ad = row["hm_x"], \
                            row["hm_y"], \
                            row["pose__pose_e_orientation_z"], \
                            row["advancement"]
            rect = mpatches.Rectangle((x - self.patch_size // 2, y - self.patch_size // 2), self.patch_size,
                                     self.patch_size, linewidth=1, edgecolor='r', facecolor='none')
            ax_hm.add_patch(rect)

            patch, _ = hmpatch(self.hm, x, y, np.rad2deg(ang), self.patch_size, scale=1)
            hm_patches.append((patch, ad))

        for i in range(n_show // 2):
            for j in range(n_show // 2):
                ax_patch = plt.subplot2grid((size), (2 + i, j), colspan=1, rowspan=1)
                patch, ad = hm_patches[i + j]
                sns.heatmap(patch, ax=ax_patch, vmin=0, vmax=1)
                ax_patch.set_title(ad)


    def show_labeled_patches(self, df, tr=None, label='True'):
        # useful if we want to test an other tr on the fly
        if tr is not None: df['label'] = df[df['advancement'] > tr]


        tmp = df.loc[df['label'] == True, ]

        sample = tmp.sort_values("advancement").head(4)

        self.show_patches_on_the_map(sample, title='true', compress=False)

        plt.show()

        tmp = df[df['label'] == False]

        sample = tmp.sort_values("advancement").tail(4)

        self.show_patches_on_the_map(sample, title='false', compress=False)

        plt.show()

    def show_traversability_in_time(self, df, dt=100):
        fig = plt.figure()
        plt.title('advancement')
        plt.plot(df.index, df['advancement'])
        plt.show()

        fig = plt.figure()
        plt.title('advancement box')
        df['advancement'].plot.box()
        plt.show()

    def plot_rotation(self, df):
        fig = plt.figure()
        plt.title('rotation')
        ax1 = plt.subplot(3, 1, 1)
        ax2 = plt.subplot(3, 1, 2)
        ax3 = plt.subplot(3, 1, 3)
        ax1.set_title('yaw')
        df['pose__pose_e_orientation_x'].plot(ax=ax1)
        ax3.set_title('pitch')
        df['pose__pose_e_orientation_y'].plot(ax=ax2)
        ax3.set_title('roll')
        df['pose__pose_e_orientation_z'].plot(ax=ax3)
        plt.legend()

    def plot_position(self, df):
        fig = plt.figure()
        plt.title('position')
        df['pose__pose_position_x'].plot(label='x')
        df['pose__pose_position_y'].plot(label='y')
        plt.legend()

    def plot_patch_map_advancement_in_time(self, df):
        fig = plt.figure(figsize=(8, 8))

        plt.ion()

        patch_size = 92

        gridspec.GridSpec(2, 2)

        fig.show()
        fig.canvas.draw()

        for i, row in df.iterrows():
            ax_ad = plt.subplot2grid((2, 2), (0, 0), colspan=2, rowspan=1)

            ax_ad.plot(df['advancement'][:i])

            #         ax = plt.subplot2grid((2,2), (1, 0), colspan=1, rowspan=1)
            patch = cv2.imread(row.loc['image_path'])
            patch = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)

            ax_patch = plt.subplot2grid((2, 2), (1, 0), colspan=1, rowspan=1)
            sns.heatmap(patch / 255, vmin=0, vmax=1, ax=ax_patch)

            x, y, ang, ad = row["hm_x"], \
                            row["hm_y"], \
                            row["pose__pose_e_orientation_z"], \
                            row["advancement"]


            ax_hm = plt.subplot2grid((2, 2), (1, 1), colspan=1, rowspan=1)
            sns.heatmap(self.hm / 255, vmin=0, vmax=1, ax=ax_hm)

            rect = mpatches.Rectangle((x - self.patch_size // 2, y - self.patch_size  // 2), self.patch_size ,
                                      self.patch_size , linewidth=1, edgecolor='r', facecolor='none')
            ax_hm.add_patch(rect)

            fig.canvas.draw()
            plt.pause(0.025)

    def show_classes(self, df, tr):
        fig = plt.figure()

        if tr is not None: temp = df['advancement'] > tr
        else: temp = df['label']

        temp.value_counts().plot.bar()
        plt.show()

    def __call__(self, df, tr=None):
        self.show_patches_on_the_map(df)
        self.plot_rotation(df)
        self.plot_position(df)
        self.show_traversability_in_time(df)
        # self.show_labeled_patches(df)
        self.show_classes(df, tr)


if __name__ == '__main__':

    hm = cv2.imread('/home/francesco/Documents/Master-Thesis/core/maps/train/bars1.png')
    hm = cv2.cvtColor(hm, cv2.COLOR_BGR2GRAY)

    deb_pip = VisualiseSimulation(hm, patch_size=92)
    df = pd.read_csv('/home/francesco/Desktop/bars1-run-recorded/csvs-light/bars1/1551992796.2643805-patch.csv')
    deb_pip.show_labeled_patches(df.dropna())
