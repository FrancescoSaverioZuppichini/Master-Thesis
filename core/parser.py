import argparse
import tabulate

from art import *

class CheckEngine(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        if values.lower() not in ['webots, gazebo']:
            raise ValueError("Engine should be one of: webouts, gazebo")
        setattr(namespace, self.dest, values)

parser = argparse.ArgumentParser(description='Traversability Simulation')
parser.add_argument('-w',
                    '--world',
                    type=str,
                    help='The path to the world file',
                    default='../../resources/worlds/webots/krock2.wbt',
                    required=False)
parser.add_argument('-e',
                    '--engine',
                    type=str,
                    help='The Engine used to run the simulation. '
                         'Available: webots, gazebo. Default webots',
                    default='webots',
                    action=CheckEngine)
parser.add_argument('-n',
                    '--n-sim',
                    type=int,
                    help="Number of simulation used.",
                    default=100)
parser.add_argument('-t',
                    '--time',
                    type=float,
                    help='Maximum time per simulation',
                    default=5)
parser.add_argument('-r',
                    '--robot',
                    type=str,
                    help='Name of the robot. Available: krock',
                    default='krock')

args = parser.parse_args()

art = text2art('Traversability Simulator')
print(art)

header = list(args.__dict__.keys())
rows = [list(args.__dict__.values())]
print(tabulate.tabulate(rows, header))

# with open('./krock.txt', 'r') as f:
#     print(f.read())