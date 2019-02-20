import glob
import random

from os import makedirs
from agent.callbacks import *
from env.webots.krock.KrockWebotsEnv import KrockWebotsEnv
from env.spawn.SpawnStragety import FlatGroundSpawnStrategy
from env.spawn import spawn_points2webots_pose

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


class SimulationPipeline():
    def __call__(self, args, **kwargs):
        rospy.init_node("traversability_simulation")

        if args.maps is None:  args.maps = [args.world]

        start = time.time()

        print('')
        rospy.loginfo('Simulation starting with {} maps'.format(len(args.maps)))

        for map in args.maps:
            env, _, bags_map_dir = make_env(map, args)
            spawn_strategy = FlatGroundSpawnStrategy(map, scale=args.height)
            spawn_points = spawn_strategy(k=args.n_sim, tol=1e-2, size=45)

            # TODO we should store the state in order to be faulty tolerant
            for i in range(args.n_sim):
                if i % 5 == 0:
                    rospy.loginfo('Reanimate robot')
                    env.reanimate()

                spawn_point = random.choice(spawn_points)
                if i < len(spawn_points): spawn_point = spawn_points[i]

                env.reset(pose=spawn_points2webots_pose(spawn_point, env))
                for i in range(int(args.time)):
                    env.render()
                    obs, r, done, _ = env.step(env.GO_FORWARD)
                    if done: break
                print('Done after {}'.format(i))
                # we want to store after each each spawn
                env.agent.die(env)

            end = time.time() - start

        rospy.loginfo('Iter={:} Elapsed={:.2f}'.format(str(i), end))
        # return all the bags stored
        return glob.glob('{}/**/*.bags'.format(args.save_dir))