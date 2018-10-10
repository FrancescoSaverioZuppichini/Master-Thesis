import rospy
import time
import tqdm

from agent.krock import Krock
from agent.callbacks import RosBagSaver

from simulation.callbacks import *

from world import *

from webots import *

from parser import args

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

sim = WebotsSimulation(name=args.robot)
sim.add_callbacks([Alarm(stop_after_s=1000),
                   OutOfMap(x=(-5, 5), y=(-5, 5))])

bar = tqdm.tqdm(range(N_SIM))
bar.set_description('Running simulations')

with bar as b:
    for _ in b:
        a = create_agent()
        sim(world=w,
            agent=a)
