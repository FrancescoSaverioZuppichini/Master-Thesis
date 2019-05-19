import numpy as np
import pandas as pd
import cv2
import dateutil

from tf.transformations import euler_from_quaternion


def add_advancement(df, dt):
    """
    Project the distance x and y computed using a rolling window
    into the current line to compute the advancement
    :param df:
    :param dt:
    :return:
    """

    if len(df) > 0:
        # look dt in the future and compute the distance for booth axis
        df["S_dX"] = df.rolling(window=(dt + 1))['pose__pose_position_x'].apply(lambda x: x[-1] - x[0],
                                                                                raw=True).shift(-dt)
        df["S_dY"] = df.rolling(window=(dt + 1))['pose__pose_position_y'].apply(lambda x: x[-1] - x[0],
                                                                                raw=True).shift(-dt)
        # project x and y in the current line and compute the advancement
        df["advancement"] = np.einsum('ij,ij->i', df[["S_dX", "S_dY"]], df[["S_oX", "S_oY"]])  # row-wise dot product

        df = df.dropna()

    return df


def convert_date2timestamp(df):
    """
    Convert the index column of the given dataframe which contains the converted bag time
    to a time expressed in second starting from the beginning. For example,
    1549572176.951121,
    1549572176.8528721

    becomes

    0.0,
    0.1df['timestamp']

    :param df:
    :return:
    """
    df.index = df[df.columns[0]]
    df['ros_time'] = df.index
    try:
        df['timestamp'] = df['ros_time'].apply(lambda x: dateutil.parser.parse(str(x)).timestamp())
        df['timestamp'] -= df['timestamp'].iloc[0]
        df = df.set_index(df['timestamp'])
    except Exception:
        print('[INFO] something exploded while converting the ros time.')
    return df


def convert_quaterion2euler(df):
    """
    Decorate the given dataframe with the euler orientation computed from the existing
    quaternion.
    :param df:
    :return:
    """

    def convert(row):
        quaternion = [row['pose__pose_orientation_x'],
                      row['pose__pose_orientation_y'],
                      row['pose__pose_orientation_z'],
                      row['pose__pose_orientation_w']]

        euler = euler_from_quaternion(quaternion)

        return pd.Series(euler)

    df[['pose__pose_e_orientation_x', 'pose__pose_e_orientation_y', 'pose__pose_e_orientation_z']] = df.apply(
        convert,
        axis=1)

    return df


def extract_cos_sin(df):
    df["S_oX"] = np.cos(df['pose__pose_e_orientation_z'].values)
    df["S_oY"] = np.sin(df['pose__pose_e_orientation_z'].values)

    assert (np.allclose(1, np.linalg.norm(df[["S_oX", "S_oY"]], axis=1)))
    return df


def parse_dataframe(df):
    df = convert_date2timestamp(df)
    df = convert_quaterion2euler(df)
    df = extract_cos_sin(df)

    return df


def clean_dataframe(df, hm, lower_bound, offset=0):
    #       drop first second (spawn time)
    df = df.loc[df.index >= lower_bound]
    # robot upside down
    df = df.loc[df['pose__pose_e_orientation_x'] >= -2.0].dropna()
    df = df.loc[df['pose__pose_e_orientation_x'] <= 2.0].dropna()
    df = df.loc[df['pose__pose_e_orientation_y'] >= -2.0].dropna()
    df = df.loc[df['pose__pose_e_orientation_y'] <= 2.0].dropna()
    df = df.dropna()

    index = df[(df['hm_y'] > (hm.shape[0] - offset)) | (df['hm_y'] < offset)
               | (df['hm_x'] > (hm.shape[1] - offset)) | (df['hm_x'] < offset)
               ].index

    # print('removing {} outliers'.format(len(index)))
    # if there are some outliers, we remove all the rows after the first one
    if len(index) > 0:
        idx = index[0]
        df = df.loc[0:idx]

    df = drop_uselesss_columns(df)
    return df


def add_hm_coordinates2row(row, hm, resolution=0.02, translation=[5, 5]):
    x, y = (row['pose__pose_position_x'], row['pose__pose_position_y'])
    x_max, y_max = hm.shape[0] * resolution, hm.shape[1] * resolution
    x_min, y_min = translation

    xs = x + x_min
    ys = -y + y_min

    return pd.Series([xs / resolution, ys / resolution])


def add_hm_coordinates2df(df, hm, resolution=0.02, translation=[5, 5]):
    df[['hm_x', 'hm_y']] = df.apply(
        lambda x: add_hm_coordinates2row(x, hm, resolution, translation),
        axis=1)

    return df


def drop_uselesss_columns(df):
    df = df.drop(['pose__pose_orientation_y',
                  'pose__pose_orientation_x',
                  'pose__pose_orientation_z',
                  'pose__pose_orientation_w',
                  'timestamp',
                  'ros_time'], axis=1)

    return df


def read_image(heightmap_png):
    """
    Read a given image and convert it to gray scale, then scale to [0,1]
    :param heightmap_png:
    :return:
    """
    hm = cv2.imread(heightmap_png)
    hm = cv2.cvtColor(hm, cv2.COLOR_BGR2GRAY)
    return hm


def store_data_keeping_same_name(df, name, out_dir):
    df.to_csv(out_dir + '/' + name + '.csv')
    return df


def open_df_and_hm_from_meta_row(row, base_dir, hm_dir):
    filename, map = row['filename'], row['map']
    df = pd.read_csv(base_dir + '/' + filename + '.csv')

    hm = cv2.imread(hm_dir + '/' + map + '.png')
    hm = cv2.cvtColor(hm, cv2.COLOR_BGR2GRAY)

    return df, hm


def extract_patches(df, hm, patch_extract_stategy):
    patches = []

    for (idx, row) in df.iterrows():
        patch = \
            patch_extract_stategy(hm, row["hm_x"], row["hm_y"], np.rad2deg(row['pose__pose_e_orientation_z']))[0]
        patches.append(patch)

    return patches


def store_patches(df, filename, patches, meta_out_dir, patches_out_dir):
    meta_dir = meta_out_dir
    images_dir = patches_out_dir

    paths = []
    for idx, patch in enumerate(patches):
        patch_file_name = '{}-{}.png'.format(filename, idx)
        path = '{}/{}'.format(images_dir, patch_file_name)
        patch = patch * 255
        patch = patch.astype(np.uint8)
        cv2.imwrite(path, patch)
        paths.append(patch_file_name)

    df['images'] = paths

    df.to_csv('{}/{}.csv'.format(meta_dir, filename))
    del patches  # free up memory

    return df, filename
