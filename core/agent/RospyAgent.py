import rospy

from .Agent import Agent


class RospyAgent(Agent):
    """
    ROS agent. This class adds some features to the basic Agent interface.
    I allows to initialize subscribers and publisher and keep a organize
    reference to them into a dictionary.
    """
    def __init__(self, rate=None):
        super().__init__()
        self.rate = rospy.Rate(hz=10) if rate == None else rate

    def __call__(self, *args, **kwargs):
        super().__call__(*args, **kwargs)
        self.subscribers = self.init_subscribers()
        self.publishers = self.init_publishers()

    def init_publishers(self):
        return {}

    def init_subscribers(self):
        return {}

    def sleep(self):
        self.rate.sleep()

    def die(self, sim, world, *args, **kwargs):
        [sub.unregister() for sub in self.subscribers.values()]
        super().die(sim, world)

