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

def make_env(map, args):
    agent = None
    env = None

    map_name, _ = path.splitext(path.basename(map))
    bags_map_dir = path.normpath(args.save_dir + '/' + map_name)

    if args.engine == 'webots':
        if args.robot == 'krock':
            makedirs(bags_map_dir, exist_ok=True)
            env = KrockWebotsEnv.from_image(
                map,
                path.abspath('./env/webots/krock/krock_no_tail.wbt'),
                {'height': args.height,
                 'resolution': 0.02},
                output_dir=path.abspath('./env/webots/krock/krock2_ros/worlds/'),
                agent_callbacks=[RosBagSaver(bags_map_dir, topics=['pose'])]
            )

    return env, map_name, bags_map_dir


class Simulation():

    #TODO # better add a constructor .from_args
    def __call__(self, args, **kwargs):
        rospy.init_node("traversability_simulation")

        if args.maps is None:  args.maps = [args.world]

        start = time.time()

        print('')
        rospy.loginfo('Simulation starting with {} maps'.format(len(args.maps)))
        maps_bar = tqdm.tqdm(args.maps)
        for map in maps_bar:
            env, _, bags_map_dir = make_env(map, args)
            spawn_strategy = FlatGroundSpawnStrategy(map, scale=args.height)
            spawn_points = spawn_strategy(k=args.n_sim, tol=1e-2, size=45)

            n_sim_bar = tqdm.tqdm(range(args.n_sim))
            for i in n_sim_bar:
                spawn_point = random.choice(spawn_points)
                if i < len(spawn_points): spawn_point = spawn_points[i]

                env.reset(pose=spawn_points2webots_pose(spawn_point, env))
                if i % 5 == 0:
                    rospy.loginfo('Reanimate robot')
                    env.reanimate()


                elapsed = 0
                start = time.time()

                while elapsed <= (int(args.time)):
                    obs, r, done, _ = env.step(env.GO_FORWARD)
                    elapsed = time.time() - start

                    if done: break

                # we want to store after each each spawn
                env.agent.die(env)
                rospy.loginfo('Done after {:.2f} seconds, callback was called {} times.'.format(elapsed, env.agent.called))


        rospy.loginfo('Iter={}'.format(str(i)))
        # return all the bags stored
        return glob.glob('{}/**/*.bags'.format(args.save_dir))