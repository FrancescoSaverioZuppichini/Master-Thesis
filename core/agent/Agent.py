from protocols import Callbackable
from .callbacks import AgentCallback

from world import World

class Agent(Callbackable, AgentCallback):
    """
    Basic Agent interface. An agent is an entity that interacts with a world.
    It is defined by the action it can do. This interfaces exposes some basic
    interaction such as 'move'.
    # TODO I should add a Brain that decide how to act based on the enviroment
    """
    def __init__(self):
        self.state = AgentState(self)
        self.set_callbacks([self])

    def __call__(self,  *args, **kwargs):
        pass

    def move(self, *args, **kwargs):
        """
        A generic function to move the agent in the world. Use it to implement your own logic. E.g

        ```
        class MyAgent(Agent):
            def move(self, action, *args, **kwargs):
                if action == 'forward':
                    self.wheels[0].spin
                    self.wheels[1].spin
                elif action == 'left':
                    self.wheels[0].stop
                    self.wheels[1].spin
                ...
        ```
        :param args:
        :param kwargs:
        :return:
        """
        pass

    def act(self, env, *args, **kwargs):
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

    def die(self, env, *args, **kwargs):
        """
         This function kill the agent. This should be implemented,
         hook to the simulation instead

        :param sim:
        :param world:
        :param args:
        :param kwargs:
        :return:
        """
        self.notify('on_shut_down')
        # self.state = AgentState(self)


class AgentState(dict):
    """
    The knapsack of an agent. It notify the Agent when something
    it is updated using the 'on_state_change' event. Use it wisely.
    """
    def __init__(self, agent: Agent):
        super().__init__()
        self.agent = agent

    def __setitem__(self, key, value):
        super().__setitem__(key, value)
        self.agent.notify('on_state_change', self.agent, key, value)
