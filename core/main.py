import rospy
import numpy as np

from geometry_msgs.msg import Pose, PoseStamped

from agent.krock import Krock
from agent.callbacks import RosBagSaver
from simulation import Simulation


rospy.init_node("record_single_trajectory")

move_r = rospy.Rate(hz=10)
spawn_r = rospy.Rate(hz=0.1)

k = Krock()
k.add_callback(RosBagSaver('./data.bag', topics=['pose']))
k()
rospy.on_shutdown(k.on_shut_down)

class RosSimulation(Simulation):
    def run(self, world, agent, *args, **kwargs):
        while not self.should_stop or not rospy.is_shutdown():
            agent.move(gait=1,
                       frontal_freq=1.0,
                       lateral_freq=0,
                       manual_mode=True)
            move_r.sleep()

sim = RosSimulation()

sim(world=None, agent=k)
