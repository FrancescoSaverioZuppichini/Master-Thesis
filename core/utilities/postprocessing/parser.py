import argparse
from art import *

parser = argparse.ArgumentParser(description='Traversability Simulation')

parser.add_argument('-r',
                    '--root',
                    type=str,
                    help='Directory in which there is a /bags subdirectory that contains the .bag files. This directory is used to store the dataframes',
                    required=True)

parser.add_argument('-m',
                    '--maps_dir',
                    type=str,
                    help='Folder that contains the maps',
                    required=True)

parser.add_argument('-t',
                    '--time_window',
                    type=str,
                    help='Time window',
                    default=100,
                    required=False)

parser.add_argument('-a',
                    '--advancement',
                    type=str,
                    help='Maximum advancement',
                    default=0.66,
                    required=False)

utility_args = parser.parse_args()

art = text2art('Traversability Postprocessing')
print(art)
