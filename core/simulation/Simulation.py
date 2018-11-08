from protocols import Callbackable
from .callbacks.SimulationCallback import SimulationCallback
from .errors import SimulationException
from utils.History import History

from world import World
from agent import Agent

class Simulation(Callbackable, SimulationCallback):
    """
     Basic Simulation interface. A simulation holds the main logic to run
    an agent, or multiple ones, into a world. It maintains an infinite loop until something happens. It can be stopped by raising a SimulationException
    or to manually set the should_stop field to False.
    Before, during and after the main loop it fires event that can be subscribed
    using the SimulationCallback class.
    """
    # TODO add a n parameter to decide the max number of iteration?
    def __init__(self, name='simulation'):
        self.name = name
        self.should_stop = False
        self.history = History()
        self.set_callbacks([self])

    def __call__(self, world: World, agent: Agent, *args, **kwargs):
        """
        When called the simulation starts and continue to loop until it is stopped
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

    def run(self, world: World, agent: Agent, n=None, *args, **kwargs):
        """
        Main routine. Inside this function the simulation should move the agent
        around the world according to the agent's internal implementation.
        :param world:
        :param agent:
        :param args:
        :param kwargs:
        :return:
        """
        self.history.new_epoch()
        i = 0
        while not self.should_stop:
            try:
                self.loop(world, agent, i, *args, **kwargs)
                self.notify('tick', self, world, agent)
                i += 1
                if n != None:
                    self.should_stop = i == n
            except SimulationException  as e:
                self.history.record('error', str(e))
                self.should_stop = True

    def loop(self, world: World, agent: Agent, n, *args, **kwargs):
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
