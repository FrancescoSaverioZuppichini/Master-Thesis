from protocols import Callbackable
from .callbacks.SimulationCallback import SimulationCallback
import rospy

class Simulation(Callbackable, SimulationCallback):
    def __init__(self):
        self.should_stop = False
        self.set_callbacks([self])

    def __call__(self, world, agent, *args, **kwargs):
        self.should_stop = False
        self.notify('on_start', self, world, agent)
        self.run(world, agent, *args, **kwargs)
        self.notify('on_finished', self, world, agent)

    def run(self, world, agent, *args, **kwargs):
        while not self.should_stop:
            self.loop(world, agent, *args, **kwargs)
            self.notify('tick', self, world, agent)

    def loop(self, world, agent, *args, **kwargs):
        pass