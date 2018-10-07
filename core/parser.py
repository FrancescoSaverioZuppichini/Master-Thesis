import argparse

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
                    required=True
                    )
parser.add_argument('-e',
                    '--engine',
                    type=str,
                    help='The Engine used to run the simulation. '
                         'Available: webots, gazebo. Default webots',
                    default='webots',
                    action=CheckEngine
                    )
args = parser.parse_args()

print(args)