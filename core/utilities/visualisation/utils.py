import pandas as pd

def meta2clusters(meta_path, base_dir):

    meta = pd.read_csv(meta_path)
    clusters = {}
    for idx, row in meta.iterrows():
        map_name = row['map']
        file_name = row['filename']
        if map_name not in clusters: clusters[map_name] = []
        try:
            clusters[map_name].append(pd.read_csv( base_dir + file_name + '.csv'))
        except FileNotFoundError:
            pass

    for key, dfs in clusters.items():
        df = pd.concat(dfs)
        df = df.reset_index(drop=True)
        clusters[key] = df
        clusters[key] = df

    return clusters