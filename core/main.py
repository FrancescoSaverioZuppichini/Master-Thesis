from os import makedirs

from agent.callbacks import *
from env.webots.krock.KrockWebotsEnv import KrockWebotsEnv

from geometry_msgs.msg import Pose

from parser import args
from tf import transformations

N_SIM = args.n_sim
SIM_TIME = args.time
WORLD = args.world

rospy.init_node("traversability_simulation")

if args.maps == None:  args.maps = [WORLD]

agent = None

# TODO main file is polluted. Create something like Topology .from_args that returns the correct env
def make_env(map):
    env = None

    map_name, _ = path.splitext(path.basename(map))
    bags_map_dir = path.normpath(args.save_dir + '/' + map_name)

    if args.engine == 'webots':
        if args.robot == 'krock':
            makedirs(bags_dir, exist_ok=True)
            env = KrockWebotsEnv.from_image(
                map,
                path.abspath('./env/webots/krock/krock.wbt'),
                {'height': args.height,
                 'resolution': 0.02},
                output_dir=path.abspath('./env/webots/krock/krock2_ros/worlds/'),
                agent_callbacks=[RosBagSaver(bags_map_dir, topics=['pose'])]
            )

    return env, map_name, bags_map_dir

b = range(N_SIM)

start = time.time()


print('')
rospy.loginfo('Simulation starting with {} maps'.format(len(args.maps)))

class SimulationPipeline():
    def __call__(self, args, **kwargs):
        for map in args.maps:
            env, _, bags_map_dir  = make_env(map)
            # TODO we should store the state in order to be faulty tolerant

            for i in range(N_SIM):
                if i % 5 == 0:
                    rospy.loginfo('Reanimate robot')
                    env.reanimate()

                env.reset()
                for i in range(int(args.time)):
                    env.render()
                    obs, r, done, _ = env.step(env.GO_FORWARD)
                    if done: break
                print('Done after {}'.format(i))
                # we want to store after each each spawn
                env.agent.die(env)

            end = time.time() - start

        rospy.loginfo('Iter={:} Elapsed={:.2f}'.format(str(i), end))

        return args.save_dir

if __name__ == '__main__':
    sim_pip = SimulationPipeline()
    sim_pip(args)
