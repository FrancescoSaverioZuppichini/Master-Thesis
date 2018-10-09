from protocols import Callbackable
from .callbacks import AgentCallback

from world import World

class Agent(Callbackable, AgentCallback):
    """
    Basic Agent interface. An agent is an entity that interacts with a world,
    and environment.
    It is defined by the action it can do. This interfaces exposes some basic
    interaction such as 'move'.
    # TODO I should add a Brain that decide how to act based on the enviroment
    """
    def __init__(self):
        self.state = {}
        self.set_callbacks([self])

    # TODO what about pass the world param?
    def spawn(self, pos, world: World, *args, **kwargs):
        """
        Spawn the robot in the world.
        :param pos:
        :return:
        """
        pass

    # TODO what about pass the world param?
    def move(self, *args, **kwargs):
        """
        This function moves the agent in the world. It depends on the
        internet implementation and the simulator used. However, it should be
        as genere as possible.
        :param args:
        :param kwargs:
        :return:
        """
        pass

    def act(self, world: World, *args, **kwargs):
        """
        This function should do something based on the world the agent
        is in. Ideally it should move it based on some input.
        :param world:
        :param args:
        :param kwargs:
        :return:
        """
        pass

    def stop(self):
        """
        Stop the agent.
        :return:
        """
        pass

    def sleep(self):
        """
        This function can be useful to allow the robot to idle.
        :return:
        """
        pass

    def die(self):
        """
        This function kill the agent.
        :return:
        """
        self.notify('on_shut_down')
