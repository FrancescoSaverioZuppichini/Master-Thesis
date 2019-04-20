import rospy

from .Agent import Agent


class RospyAgent(Agent):
    """
    ROS agent. This class adds some features to the basic Agent interface.
    It allows to initialize subscribers and publisher and keep a organize
    reference to them into a dictionary.
    """
    def __init__(self, rate=None):
        super().__init__()
        self.rate = rospy.Rate(hz=10) if rate == None else rate
        self.subscribers = None

    def __call__(self, *args, **kwargs):
        super().__call__(*args, **kwargs)
        if self.subscribers is not None: self.unregister()
        self.subscribers = self.init_subscribers()
        self.publishers = self.init_publishers()

    def init_publishers(self):
        return {}

    def init_subscribers(self):
        return {}

    def sleep(self):
        self.rate.sleep()

    def unregister(self):
        [sub.unregister() for sub in self.subscribers.values()]

    def die(self, env, *args, **kwargs):
        super().die(env)

