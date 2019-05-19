import glob
import cv2

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import seaborn as sns

from utilities.postprocessing.utils import *
from matplotlib import gridspec
from utilities.postprocessing.handlers import add_advancement
from utilities.postprocessing.utils import PatchExtractStrategy
from utilities.visualisation.utils import *

class DataFrameVisualization():
    """
    Visualize a visualization dataframe
    """
    def __init__(self, df, time_window=None,*args, **kwargs):

        self.df = df

    def __call__(self, tr, time_window=None):
        self.plot_advancement().show()
        self.plot_advancement_box().show()
        self.plot_rotation().show()
        self.show_classes(tr).show()

    def plot_rotation(self):
        df = self.df
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
        return fig

    def plot_advancement(self):
        df = self.df
        fig = plt.figure()
        plt.title('advancement')
        df['advancement'].plot.line()
        return fig

    def plot_advancement_box(self):
        fig = plt.figure()
        plt.title('advancement box')
        self.df['advancement'].plot.box()
        return fig

    def show_classes(self, tr):
        df = self.df
        fig = plt.figure()
        if tr is not None: temp = df['advancement'] > tr
        else: temp = df['label']

        plt.title("tr={:3f}".format(tr))
        temp.value_counts().plot.bar()

        return fig

    @classmethod
    def from_root(cls, root, *args, **kwargs):
        dfs_paths = glob.glob(root + '/*.csv')
        if len(dfs_paths) <= 0: dfs_paths = glob.glob(root + '/**/*.csv')
        dfs = [pd.read_csv(df_paths) for df_paths in dfs_paths]
        return cls.from_dfs(dfs, *args, **kwargs)

    @classmethod
    def from_dfs(cls, dfs, time_window=None, *args, **kwargs):
        if time_window is not None: dfs = [add_advancement(df, time_window) for df in dfs]
        df_total = pd.concat(dfs, sort=False)
        df_total = df_total.dropna()
        df_total = df_total.reset_index(drop=True)

        return cls(df_total, *args, **kwargs)

    @classmethod
    def from_path(cls, path, *args, **kwargs):
        return cls(pd.read_csv(path), *args, **kwargs)

class PatchesAndDataframeVisualization(DataFrameVisualization):
    def __init__(self, df, hm, max_advancement=1, image_dir=None, patch_extractor=KrockPatchExtractStrategy, *args, **kwargs):
        super().__init__(df, *args, **kwargs)
        self.hm = hm
        self.max_advancement = max_advancement
        self.image_dir = image_dir
        self.patch_extractor = patch_extractor(max_advancement)
        # run path extractor one time to get the path_size -> maybe we need later
        dummy_path, _ = self.patch_extractor(hm, hm.shape[0]//2, hm.shape[1]//2, 0)
        self.patch_size = dummy_path.shape

    @property
    def hm_ax(self):
        fig = plt.figure()
        ax = sns.heatmap(self.hm)
        return ax


    def fill_patch_on_ax(self, ax, corners):
        ax.fill(corners[[0, 1, 2, 3, 0], 0], corners[[0, 1, 2, 3, 0], 1], alpha=0.4,
                   facecolor='g')
        return ax

    def show_patches_on_the_map(self, n_show=4, title='patches', compress=True, df=None, res=0.02, disable_patch_axis=True):
        if df is None : df = self.df
        fig = plt.figure(figsize=(4 * 2, 4 * 3))
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
            patch, corners = self.patch_extractor(self.hm, x, y, np.rad2deg(ang))

            self.fill_patch_on_ax(ax_hm, corners)

            patch = patch.astype(np.float32)
            patch = patch - patch[patch.shape[0] // 2, patch.shape[1] // 2]
            hm_patches.append((patch, ad))

        for i in range(n_show // 2):
            for j in range(n_show // 2):
                ax_patch = plt.subplot2grid((size), (2 + i, j), colspan=1, rowspan=1)
                patch, ad = hm_patches[i + j]
                sns.heatmap(patch, ax=ax_patch)
                ax_patch.set_title('{:.3f}'.format(ad))

                if disable_patch_axis:
                    ax_patch.get_yaxis().set_visible(False)
                    ax_patch.get_xaxis().set_visible(False)

        return fig

    def show_patches(self, center=False, n_samples=4, scale=1, random_state=0, sample=None, disable_patch_axis=False):
        df = self.df
        # sample = df.sample(n_samples, random_state=random_state)
        if sample is None: sample = df[:n_samples]

        fig, ax = plt.subplots(nrows=n_samples // 2, ncols=n_samples // 2)
        fig.suptitle('patches center={}'.format(center))
        for row in ax:
            for idx, (col, (i, row)) in enumerate(zip(row, sample.iterrows())):
                x, y = row["hm_x"], row["hm_y"]
                patch, corners = self.patch_extractor(self.hm, x, y, np.rad2deg(row['pose__pose_e_orientation_z']))
                patch = patch.astype(np.float32)
                # if center: patch = patch - patch[patch.shape[0] // 2, patch.shape[1] // 2]
                # col.plot(self.patch_size // 2, self.patch_size // 2, marker='o', color='r', ls='', linewidth=10,
                #          label='finish')
                if disable_patch_axis:
                    col.get_yaxis().set_visible(False)
                    col.get_xaxis().set_visible(False)
                col.set_title('{:.2f}m'.format(row['advancement']))
                sns.heatmap(patch, ax=col)

        return fig


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



    def show_labeled_patches(self, tr=None, n=4):
        df = self.df
        # useful if we want to test an other tr on the fly
        df['label'] = df['advancement'] > tr
        tmp = df.loc[df['label'] == True]

        sample = tmp.sort_values("advancement", ascending=False).head(n)

        self.show_patches(sample=sample)

        plt.show()

        tmp = df[df['label'] == False]

        sample = tmp.sort_values("advancement", ascending=False).tail(n)

        self.show_patches(sample=sample)

        plt.show()

    def plot_box_on_hm(self, row):
        fig = plt.figure()
        ax = plt.gca()
        x, y, ang, ad = row["hm_x"], \
                        row["hm_y"], \
                        row["pose__pose_e_orientation_z"], \
                        row["advancement"]

        sns.heatmap(self.hm / 255, vmin=0, vmax=1, ax=ax)

        rect = mpatches.Rectangle((x - self.patch_size // 2, y - self.patch_size // 2), self.patch_size,
                                  self.patch_size, linewidth=1, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
        plt.show()


    def plot_patch_map_advancement_in_time(self):
        df = self.df
        fig = plt.figure(figsize=(8, 8))

        plt.ion()

        patch_size = 92

        gridspec.GridSpec(2, 2)

        fig.show()
        fig.canvas.draw()

        for i, row in df.iterrows():
            ax_ad = plt.subplot2grid((2, 2), (0, 0), colspan=2, rowspan=1)

            ax_ad.plot(df['advancement'][:i])

            x, y, ang, ad = row["hm_x"], \
                            row["hm_y"], \
                            row["pose__pose_e_orientation_z"], \
                            row["advancement"]
            patch, corners = self.patch_extractor(self.hm, x, y, np.rad2deg(row['pose__pose_e_orientation_z']))
            patch = patch.astype(np.float32)

            ax_patch = plt.subplot2grid((2, 2), (1, 0), colspan=1, rowspan=1)
            sns.heatmap(patch / 255, vmin=0, vmax=1, ax=ax_patch)



            ax_hm = plt.subplot2grid((2, 2), (1, 1), colspan=1, rowspan=1)
            sns.heatmap(self.hm / 255, vmin=0, vmax=1, ax=ax_hm)

            self.fill_patch_on_ax(ax_hm, corners)

            fig.canvas.draw()
            plt.pause(0.025)


    @classmethod
    def from_df_path_and_hm_path(cls, df_path, hm_path, *args, **kwargs):
        df = pd.read_csv(df_path)

        hm = cv2.imread(hm_path)
        hm = cv2.cvtColor(hm, cv2.COLOR_BGR2GRAY)

        return cls(df, hm, *args, **kwargs)

    @classmethod
    def from_df_path_and_hm_dir(cls, df_path, hm_dir, *args, **kwargs):
        df = pd.read_csv(df_path)
        map_name = df['map_name'][0]

        hm = cv2.imread(glob.glob('{}/**/{}.png'.format(hm_dir, map_name))[0])
        hm = cv2.cvtColor(hm, cv2.COLOR_BGR2GRAY)

        return cls(df, hm, *args, **kwargs)

def find_tr(df):
    return df['advancement'].mean() / 2