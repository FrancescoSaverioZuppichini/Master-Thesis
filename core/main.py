import rospy
import numpy as np
import time
from geometry_msgs.msg import Pose, PoseStamped
import tqdm

from agent.krock import Krock
from agent.callbacks import RosBagSaver
from simulation import Simulation
from simulation.callbacks import Alarm

N_SIM = 20
SIM_TIME = 4

rospy.init_node("record_single_trajectory")

nap = rospy.Rate(hz=10)

krock = Krock()
krock.add_callback(RosBagSaver('./data.bag', topics=['pose']))
krock()
rospy.on_shutdown(krock.on_shut_down)

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

sim = MySimulation()
sim.add_callback(Alarm(stop_after_s=SIM_TIME))

bar = tqdm.tqdm(range(N_SIM))
bar.set_description('Running simulations')

for _ in bar:
    sim(world=None, agent=krock)
