import rospy
import numpy as np
import time
from geometry_msgs.msg import Pose, PoseStamped
import tqdm

from agent.krock import Krock
from agent.callbacks import RosBagSaver

from simulation import Simulation
from simulation.callbacks import *

from world import *

N_SIM = 100
SIM_TIME = 5
WORLD = 'krock2'

rospy.init_node("record_single_trajectory")

nap = rospy.Rate(hz=10)

krock = Krock()
krock.add_callback(RosBagSaver('./data/{}.bag'.format(time.time()), topics=['pose']))
krock()

w = World('krock2',
          format='wbt',
          base_dir='../../resources/worlds/webots')

class MySimulation(Simulation):
    def on_start(self, *args, **kwargs):
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
    sim(world=None, agent=krock)
