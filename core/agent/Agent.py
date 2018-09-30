from protocols import Callbackable

class Agent(Callbackable):
    def __init__(self):
        self.state = {}
        self.set_callbacks([self])

    def spawn(self, pos):
        pass

    def move(self, *args, **kwargs):
        pass

    def stop(self):
        pass
