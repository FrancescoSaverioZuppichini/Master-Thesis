import rospy
import time

from os import path

from agent.callbacks import *
from webots.krock import Krock

from simulation.conditions import *

from KrockWebotsEnv import KrockWebotsEnv
from webots import *

from parser import args

import pprint

N_SIM = args.n_sim
SIM_TIME = args.time
WORLD = args.world

rospy.init_node("traversability_simulation")

if args.maps == None:  args.maps = [WORLD]

agent = None
worlds = []

# TODO main file is polluted. Create something like Topology .from_args that returns agent, worlds and sim
if args.engine == 'webots':
    if args.robot == 'krock':
        src_world = path.abspath('./webots/krock/krock.wbt')
        agent = Krock

        for map in args.maps:
            w = WebotsWorld.from_image(
                map,
                path.abspath('./webots/krock/krock.wbt'),
                {'height': 1,
                 'resolution': 0.02},
                # output_path='/krock/krock2_ros/worlds/temp.wbt')
                output_dir=path.abspath('./webots/krock/krock2_ros/worlds/'))

            worlds.append(w)

if w == None:
    raise ValueError('No world created. Probably you selected a no supported engine. Run main.py --help')

if agent == None:
    raise ValueError('No agent created. Probably you selected a no supported agent. Run main.py --help')


def create_agent(w):
    krock = agent()
    krock.add_callback(RosBagSaver(args.save_dir,
                                   topics=['pose',
                                           'frontal_camera']))

    krock(w)

    return krock

N_SIM = 10

b = range(N_SIM)

start = time.time()
print('')

# try:
for w in worlds:
    w()

    for i in range(N_SIM):
        a = create_agent(w)

        env = KrockWebotsEnv(agent=a, world=w, stop=IfOneFalseOf([IsNotStuck(n_last=50), IsInside()]))
        # TODO as always the reanimation breaks something
        # if i % 20 == 0:
        #     rospy.loginfo('Reanimate robot')
        #     w.reanimate()

        obs = env.reset()

        for _ in range(500):
            env.render()
            obs, r, done, _ = env.step(env.GO_FORWARD)
            if done: break
        a.die(env, w)



    end = time.time() - start

    rospy.loginfo('Iter={:} Elapsed={:.2f}'.format(str(i), end))
# except Exception as e:
#     print(e)
#     # rospy.signal_shutdown('KeyboardInterrupt')
