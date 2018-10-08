import rospy
import time
import tqdm
import subprocess

from agent.krock import Krock
from agent.callbacks import RosBagSaver

from simulation import Simulation
from simulation.callbacks import *

from world import *
from utils import Supervisor

from parser import args
from pprint import pprint

# pprint(args.__dict__)

N_SIM = args.n_sim
SIM_TIME = args.time
WORLD = args.world

rospy.init_node("record_single_trajectory")

nap = rospy.Rate(hz=10)

w = World(file_path=WORLD)

# subprocess.call('webots --stdout --minimize --batch', shell=True)

# time.sleep(10)


# TODO move this class away and create a WebotsSimulation class
class MySimulation(Simulation, Supervisor):
    name = '/krock'

    def on_start(self, sim, world, agent, *args, **kwargs):
        self.load_world(str(world.path))
        krock.spawn(pos=None)

    def loop(self, world, agent, *args, **kwargs):
        nap.sleep()
        agent.move(gait=1,
                   frontal_freq=1.0,
                   lateral_freq=0,
                   manual_mode=True)
        nap.sleep()

    def on_finish(self, sim, *args, **kwargs):
        krock.move(gait=1,
                   frontal_freq=0,
                   lateral_freq=0,
                   manual_mode=True)
        krock.die()


sim = MySimulation()
sim.add_callbacks([Alarm(stop_after_s=SIM_TIME),
                   OutOfMap(x=(-5, 5), y=(-5, 5))])

bar = tqdm.tqdm(range(N_SIM))
bar.set_description('Running simulations')

for _ in bar:
    krock = Krock()
    krock.add_callback(RosBagSaver('./data/{}.bag'.format(time.time()),
                                   topics=['pose']))
    krock()
    sim(world=w, agent=krock)
