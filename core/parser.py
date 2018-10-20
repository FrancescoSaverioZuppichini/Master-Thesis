import argparse
import tabulate

from art import *

QUARRY = '/home/francesco/Documents/krock-sim/krock/krock2_ros/map_generation/quarry.wbt'
BASIC = '/home/francesco/Documents/krock-sim/krock/krock2_ros/worlds/krock2_camera.wbt'

class CheckEngine(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        if values.lower() not in ['webots2ros, gazebo']:
            raise ValueError("Engine should be one of: webouts, gazebo")
        setattr(namespace, self.dest, values)

parser = argparse.ArgumentParser(description='Traversability Simulation')
parser.add_argument('-w',
                    '--world',
                    type=str,
                    help='The path to the world file',
                    default=QUARRY,
                    required=False)
parser.add_argument('-e',
                    '--engine',
                    type=str,
                    help='The Engine used to run the simulation. '
                         'Available: webots2ros, gazebo. Default webots2ros',
                    default='webots2ros',
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
                    default=1)
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