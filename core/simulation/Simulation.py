from protocols import Callbackable
from .callbacks.SimulationCallback import SimulationCallback
from .errors import SimulationException
import rospy


class Simulation(Callbackable, SimulationCallback):
    def __init__(self, name='simulation'):
        self.name = name
        self.should_stop = False
        self.set_callbacks([self])

    def __call__(self, world, agent, *args, **kwargs):
        self.should_stop = False
        self.notify('on_start', self, world, agent)
        self.run(world, agent, *args, **kwargs)
        self.notify('on_finish', self, world, agent)

    def run(self, world, agent, *args, **kwargs):
        while not self.should_stop:
            try:
                self.loop(world, agent, *args, **kwargs)
                self.notify('tick', self, world, agent)
            except SimulationException  as e:
                self.should_stop = True

    def loop(self, world, agent, *args, **kwargs):
        pass

    def stop(self):
        self.should_stop = True
