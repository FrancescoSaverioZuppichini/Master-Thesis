import rospy
import time

from os import path

from agent.callbacks import *
from webots.krock import Krock

from simulation import BasicSimulation, Simulation
from simulation.callbacks import *
from simulation.conditions import *

from world import World
from webots import *

from parser import args

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
                                           'frontal_camera'
                                           ]))

    krock(w)

    return krock


# TODO check if robot fall upside down
sim = BasicSimulation(name=args.robot)
sim.add_callbacks([Alarm(stop_after_s=SIM_TIME)])

b = range(N_SIM)

start = time.time()
print('')

try:
    for w in worlds:
        w()
        # TODO these info should be taken directly from the current world
        cond = IfOneFalseOf([IsNotStuck(), IsInside()])
        for i, _ in enumerate(b):
            if i % 10 == 0: w.reanimate()
            a = create_agent(w)

            sim(world=w,
                agent=a,
                until=cond)
            end = time.time() - start

            rospy.loginfo('Iter={:} Error={:} Elapsed={:.2f}'.format(str(i), sim.history['error', -1], end))
except Exception as e:
    sim.should_stop = True
    print(e)
    # rospy.signal_shutdown('KeyboardInterrupt')
