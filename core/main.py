
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
    if args.engine == 'webots':
        if args.robot == 'krock':
            map_name, _ = path.splitext(path.basename(map))
            bag_save_path = path.normpath(args.save_dir + '/' + map_name)
            makedirs(bag_save_path, exist_ok=True)
            # TODO decouple WebotsEnv and Agent
            env = KrockWebotsEnv.from_image(
                map,
                path.abspath('./env/webots/krock/krock.wbt'),
                {'height': 1,
                 'resolution': 0.02},
                output_dir=path.abspath('./env/webots/krock/krock2_ros/worlds/'),
                agent_callbacks=[RosBagSaver(bag_save_path, topics=['pose'])]
            )

    return env

b = range(N_SIM)

start = time.time()


print('')
rospy.loginfo('Simulation starting with {} maps'.format(len(args.maps)))

for map in args.maps:
    env = make_env(map)

    for i in range(N_SIM):
        # TODO as always the reanimation breaks something
        # if i % 20 == 0:
        #     rospy.loginfo('Reanimate robot')
        #     w.reanimate()
        env.reset()

        for i in range(200):
            env.render()
            obs, r, done, _ = env.step(env.GO_FORWARD)
            if done: break
        print('Done after {}'.format(i))
        # we want to store at each spawn
        env.agent.die(env)


    end = time.time() - start

rospy.loginfo('Iter={:} Elapsed={:.2f}'.format(str(i), end))
