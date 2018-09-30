from protocols import Callbackable
import rospy

class Simulation(Callbackable):
    def __init__(self):
        self.should_stop = False
        self.set_callbacks([])

    def __call__(self, world, agent, *args, **kwargs):
        self.notify('on_start')
        self.run(world, agent, *args, **kwargs)
        self.notify('on_finished')

    def run(self, world, agent, *args, **kwargs):
        pass


