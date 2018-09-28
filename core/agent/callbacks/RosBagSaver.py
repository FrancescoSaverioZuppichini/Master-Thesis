import rosbag

from .AgentCallback import AgentCallback

class RosBagSaver(AgentCallback):
    def __init__(self, save_dir, topics=None):
        self.bag = rosbag.Bag(save_dir, 'w')
        self.topics = topics

    def on_state_change(self, key, value):
        store = True
        if self.topics != None: store = key in self.topics
        if store: self.bag.write(key, value)

    def on_shut_down(self):
        self.bag.close()