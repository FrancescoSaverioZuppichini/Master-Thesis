import rosbag
import threading

from os import path
from .AgentCallback import AgentCallback

class RosBagSaver(AgentCallback):
    """
    This callback stores the topics in which the Agent is subscribed into a bag file.
    """

    def __init__(self, save_dir, topics=None, max_size=1024, workers=1):
        self.save_dir = save_dir
        self.topics = topics
        self.cache = {}
        self.max_size = max_size
        self.size = 0
        self.workers = workers
        self.tr = threading.Thread(target=self.store, args=(self,))

    def on_state_change(self, agent, key, value):
        self.agent = agent
        store = True

        if self.topics != None: store = key in self.topics

        if key not in self.cache: self.cache[key] = []

        if store: self.cache[key].append(value)
        self.size = len(self.cache[key])


    def write(self, data):
        key, values = data
        for value in values:
            self.bag.write(key, value)
        return True

    def store(self, file_name):
        file_name = path.normpath(self.save_dir + '/{}.bag'.format(file_name))

        self.bag = rosbag.Bag(file_name, 'w')

        for key in self.cache.keys():
            for value in self.cache[key]:
                self.bag.write(key, value)

        self.bag.close()

        # clear cache
        self.cache = {}
        self.size = 0


    def on_shut_down(self, env, file_name, *args, **kwargs):
        self.store(file_name)
