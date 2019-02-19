import argparse

parser = argparse.ArgumentParser(description='Traversability Simulation')


parser.add_argument('-i',
                    '--base_dir',
                    type=str,
                    help='Directory in which there is a /bag subdirectory that contains the .bag files. This directory is used to store the dataframes',
                    required=True)


parser.add_argument('-m',
                    '--maps_folder',
                    type=str,
                    help='Folder that contains the maps',
                    required=True)

parser.add_argument('-c',
                    '--csv_dir',
                    type=str,
                    help='Output folders for the csvs.',
                    required=False)

parser.add_argument('-o',
                    '--out_dir',
                    type=str,
                    help='Output folders for the dataset.',
                    required=False)

parser.add_argument('-t',
                    '--time_window',
                    type=str,
                    help='Time window',
                    default=125,
                    required=False)

parser.add_argument('-a',
                    '--advancement_th',
                    type=str,
                    help='Advancement threshold used for labeling.',
                    default=0.12,
                    required=False)

parser.add_argument('-s',
                    '--skip_every',
                    type=str,
                    help='How many rows to skip while post processing the bags file/',
                    default=25,
                    required=False)

parser.add_argument('-p',
                    '--patch_size',
                    type=int,
                    help='Size of the patch in pixel',
                    default=100,
                    required=False)

parser.add_argument(
                    '--translation',
                    type=list,
                    help='Translation to be applied to each position in the bag',
                    default=[5,5],
                    required=False)

args = parser.parse_args()
