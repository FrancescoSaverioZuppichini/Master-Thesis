from protocols import Callbackable
from .callbacks.SimulationCallback import SimulationCallback
from .errors import SimulationException

from world import World
from agent import Agent
import rospy


class Simulation(Callbackable, SimulationCallback):
    """
    Basic Simulation interface. A simulation holds the main logic to run
    an agent, or multiple ones, into a world. It maintains a infinite loop
    until something happens. It can be stop by raising a SimulationException
    or to manually set the should_stop field to False.
    Before, during and after the main loop it fires event that can be subscribed
    using the SimulationCallback class.
    """
    def __init__(self, name='simulation'):
        self.name = name
        self.should_stop = False
        self.set_callbacks([self])

    def __call__(self, world, agent, *args, **kwargs):
        """
        When called the simulation start and continue to loop until it is stopped
        :param world:
        :param agent:
        :param args:
        :param kwargs:
        :return:
        """
        self.should_stop = False
        self.notify('on_start', self, world, agent)
        self.run(world, agent, *args, **kwargs)
        self.notify('on_finish', self, world, agent)

    def run(self, world: World, agent: Agent, *args, **kwargs):
        """
        Main routine. Inside this function the simulation should move the agent
        around the world according to the agent internal implementation.
        :param world:
        :param agent:
        :param args:
        :param kwargs:
        :return:
        """
        while not self.should_stop:
            try:
                self.loop(world, agent, *args, **kwargs)
                self.notify('tick', self, world, agent)
            except SimulationException  as e:
                self.should_stop = True

    def loop(self, world: World, agent: Agent, *args, **kwargs):
        """
        This function is called every iteration and must be overwrite.
        :param world:
        :param agent:
        :param args:
        :param kwargs:
        :return:
        """
        pass

    def stop(self):
        self.should_stop = True
