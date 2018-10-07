import rospy

from .Agent import Agent


class RospyAgent(Agent):
    def __init__(self, rate=None):
        super().__init__()
        self.rate = rospy.Rate(hz=10) if rate == None else rate
        self.state = AgentState(self)
        # rospy.on_shutdown(self.die)

    def __call__(self, *args, **kwargs):
        self.subscribers = self.init_subscribers()
        self.publishers = self.init_publishers()

    def init_publishers(self):
        return {}

    def init_subscribers(self):
        return {}


class AgentState(dict):
    def __init__(self, agent: RospyAgent):
        super().__init__()
        self.agent = agent

    def __setitem__(self, key, value):
        super().__setitem__(key, value)
        self.agent.notify('on_state_change', key, value)
