import glob
import random

from os import makedirs
import sys
sys.path.append("../")

from simulation.agent.callbacks import *
from simulation.env.webots.krock import KrockWebotsEnv
from simulation.env.spawn import FlatGroundSpawnStrategy, spawn_points2webots_pose
from utilities import run_for
import tqdm
import random

def make_env(map, args):
    agent = None
    env = None

    map_name, _ = path.splitext(path.basename(map))
    bags_map_dir = path.normpath(args.save_dir + '/' + map_name)

    if args.engine == 'webots':
        if args.robot == 'krock':
            height = args.height
            # height = random.randint(2,5)
            # makedirs(bags_map_dir, exist_ok=True)
            env = KrockWebotsEnv.from_image(
                map,
                path.abspath('./env/webots/krock/krock_no_tail.wbt'),
                {'height': height,
                 'resolution': 0.02},
                output_dir=path.abspath('./env/webots/krock/krock2_ros/worlds/'),
                agent_callbacks=[RosBagSaver(args.save_dir, topics=['pose'])]
            )

    return env, map_name, bags_map_dir, height

class Simulation():
    def __init__(self):
        self.meta = None
    #TODO # better add a constructor .from_args
    def __call__(self, args, **kwargs):
        rospy.init_node("traversability_simulation")

        if args.maps is None:  args.maps = [args.world]

        start = time.time()

        maps_bar = tqdm.tqdm(args.maps)
        for map in maps_bar:

            maps_bar.set_description('Running {}'.format(map))
            spawn_strategy = FlatGroundSpawnStrategy(map, scale=args.height)
            random_spawn = True
            # todo add a flag for the random spawn
            try:
                spawn_points = spawn_strategy(k=args.n_sim, tol=1e-2, size=45)
            except ValueError:
                print('No flat points!.')
            #     there are no flat spawn points, fall back to random spawn
                random_spawn = True
            # random_spawn = True
            print('[INFO] random_spawn = {}'.format(random_spawn))
            env, _, bags_map_dir, height = make_env(map, args)

            n_sim_bar = tqdm.tqdm(range(args.n_sim), leave=False)
            for i in n_sim_bar:
                if not random_spawn:
                    spawn_point = random.choice(spawn_points)
                    if i < len(spawn_points): spawn_point = spawn_points[i]
                else:
                    spawn_point = [random.randint(22, 513 - 22), random.randint(22, 513 - 22)]
                env.reset(pose=spawn_points2webots_pose(spawn_point, env))


                if i % 5 == 0:
                    n_sim_bar.set_description('Reanimate robot')
                    env.reanimate()


                elapsed = 0
                start = time.time()

                while elapsed <= (int(args.time)):
                    obs, r, done, _ = env.step(env.GO_FORWARD)
                    elapsed = time.time() - start

                    if done: break

                # we want to store after each each spawn
                map_name = path.splitext(path.basename(map))[0]
                file_name = '{}-{}-{}'.format(map_name, height, i)
                env.agent.die(env, file_name)

                temp = pd.DataFrame(data={'filename': [file_name], 'map': [map_name], 'height': [height]})

                if self.meta is None:
                    try:
                        self.meta = pd.read_csv('{}/meta.csv'.format(args.save_dir), index_col=[0])
                        self.meta = pd.concat([self.meta, temp])
                        self.meta = self.meta.reset_index(drop=True)
                    except FileNotFoundError:
                        self.meta = temp
                else:
                    self.meta = pd.concat([self.meta, temp])
                    self.meta = self.meta.reset_index(drop=True)
                self.meta.to_csv('{}/meta.csv'.format(args.save_dir))
                # rospy.loginfo('Done after {:.2f} seconds, callback was called {} times.'.format(elapsed, env.agent.called))


        # rospy.loginfo('Iter={}'.format(str(i)))
        # return all the bags stored
        return glob.glob('{}/**/*.bags'.format(args.save_dir))