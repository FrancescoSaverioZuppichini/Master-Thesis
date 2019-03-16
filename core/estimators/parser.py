import argparse


parser = argparse.ArgumentParser(description='Estimatorss')
parser.add_argument('-i',
                    '--dataset_dir',
                    type=str,
                    help='Path the a directory that contains train, val and test dataset',
                    required=True)

parser.add_argument('--train_dir',
                    type=str,
                    help='Path to train directory',
                    default=None,
                    required=False)

parser.add_argument('--val_dir',
                    type=str,
                    help='Path to val directory',
                    default=None,
                    required=False)

parser.add_argument('--tests_dir',
                    type=str,
                    help='Path to test directory',
                    default=None,
                    required=False)

args = parser.parse_args()
# TODO import an use it
print(args.maps)
