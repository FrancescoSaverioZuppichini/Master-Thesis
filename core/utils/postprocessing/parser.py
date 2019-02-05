import argparse
# TODO add it
parser = argparse.ArgumentParser(description='Traversability Simulation')
parser.add_argument('-i',
                    '--bags',
                    type=str,
                    help='The folder that contains the bags files.',
                    required=True)

parser.add_argument('-c',
                    '--csvs',
                    type=str,
                    help='Output folders for the csvs.',
                    required=False)

parser.add_argument('-o',
                    '--dataset',
                    type=str,
                    help='Output folders for the dataset.',
                    required=True)

parser.add_argument('-t',
                    '--time_window',
                    type=str,
                    help='Time window',
                    default=100,
                    required=True)

parser.add_argument('-a',
                    '--advancement_th',
                    type=str,
                    help='Advancement threshold used for labeling.',
                    default=0.95,
                    required=True)

parser.add_argument('-s',
                    '--skip_every',
                    type=str,
                    help='How many rows to skip while post processing the bags file/',
                    default=25,
                    required=True)

parser.add_argument('-p',
                    '--path_size',
                    type=str,
                    help='Size of the path in cm',
                    required=True)

args = parser.parse_args()
