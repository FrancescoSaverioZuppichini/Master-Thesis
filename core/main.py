import rospy
import time

from os import path

from agent.callbacks import *
from webots.krock import Krock

from simulation import BasicSimulation
from simulation.callbacks import *

from world import World
from webots import *

from parser import args

N_SIM = args.n_sim
SIM_TIME = args.time
WORLD = args.world

rospy.init_node("traversability_simulation")

w = None
agent = None

# TODO main file is polluted. Create something like Topology .from_args that returns agent, world and sim
if args.engine == 'webots':
    if args.robot == 'krock':
        src_world = path.abspath('./webots/krock/krock.wbt')
        agent = Krock
        w = WebotsWorld.from_image(
            WORLD,
            path.abspath('./webots/krock/krock.wbt'),
            {'height': 1,
             'resolution': 0.02},
            output_path='/krock/krock2_ros/worlds/temp.wbt')

if w == None:
    raise ValueError('No world created. Probably you selected a no supported engine. Run main.py --help')

if agent == None:
    raise ValueError('No agent created. Probably you selected a no supported agent. Run main.py --help')
w()


def create_agent():
    krock = agent()
    krock.add_callback(RosBagSaver(args.save_dir,
                                   topics=['pose']))

    # krock.add_callback(RosBagSaver('./data/{}.bag'.format(time.time()),
    #                                topics=['pose']))
    krock()

    return krock


# TODO check if robot fall upside down
sim = BasicSimulation(name=args.robot)
sim.add_callbacks([Alarm(stop_after_s=SIM_TIME),
                   OutOfMap(x=w.x, y=w.y)
                   ])

b = range(N_SIM)

start = time.time()
print('')

for iter, _ in enumerate(b):
    if (iter + 1) % 10 == 0: w.reanimate()
    a = create_agent()

    sim(world=w,
        agent=a)
    end = time.time() - start

    rospy.loginfo('Iter={:} Error={:} Elapsed={:.2f}'.format(str(iter), sim.history['error', -1], end))
