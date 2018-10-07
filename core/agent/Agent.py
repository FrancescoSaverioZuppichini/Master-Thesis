from protocols import Callbackable
from .callbacks import AgentCallback

class Agent(Callbackable, AgentCallback):
    def __init__(self):
        self.state = {}
        self.set_callbacks([self])

    def spawn(self, pos):
        pass

    def move(self, *args, **kwargs):
        pass

    def stop(self):
        pass

    def die(self):
        self.notify('on_shut_down')
