import rospy

from .Simulation import Simulation

class TraversabilitySimulation(Simulation):

    def run(self, world, agent, schedule):
        while not self.should_stop or not rospy.is_shutdown():
            schedule(world, agent)