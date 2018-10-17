import rospy
import time
import tqdm

from agent.krock import Krock
from agent.callbacks import RosBagSaver

from simulation import BasicSimulation
from simulation.callbacks import *

from webots import *

from parser import args

import pprint

N_SIM = args.n_sim
SIM_TIME = args.time
WORLD = args.world

rospy.init_node("traversability_simulation")

w = WebotsWorld(file_path=WORLD)
w()

def create_agent():
    krock = Krock()
    # krock.add_callback(RosBagSaver('./data/{}.bag'.format(time.time()),
    #                                topics=['pose']))
    krock()
    return krock


sim = BasicSimulation(name=args.robot)
sim.add_callbacks([Alarm(stop_after_s=1000),
                   OutOfMap(x=(-5, 5), y=(-5, 5))])

# bar = tqdm.tqdm(range(N_SIM), leave=False)
# bar.set_description('Running simulations')

b = range(N_SIM)
# with bar as b:

start = time.time()
print('')
print('{:20s} {:^40s} {:12s}'.format('iteration', 'error', 'dur'))
for iter, _ in enumerate(b):
    a = create_agent()
    sim(world=w,
        agent=a)
    end = time.time() - start

    print('{:20s} {:40s} {:20.4f}'.format(str(iter), sim.history['error', -1], end))
