from protocols import Callbackable
import rospy

class Simulation(Callbackable):
    def __init__(self):
        self.should_stop = False
        self.set_callbacks([])

    def __call__(self, *args, **kwargs):
        pass

class RosSimulation(Simulation):
    def __call__(self, world, agent, schedule, *args, **kwargs):
        self.notify('on_start')
        while not self.should_stop or not rospy.is_shutdown():
            schedule(world, agent, *args, **kwargs)
        self.notify('on_finished')

