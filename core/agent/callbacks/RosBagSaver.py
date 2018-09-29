import rosbag

from .AgentCallback import AgentCallback
import threading

class RosBagSaver(AgentCallback):
    def __init__(self, save_dir, topics=None):
        self.save_dir = save_dir
        self.topics = topics
        self.cache = {}
        self.max_size = 16
        self.iter = 0
        self.tr = threading.Thread(target=self.store)

    def on_state_change(self, key, value):
        store = True
        if self.topics != None: store = key in self.topics
        if key not in self.cache: self.cache[key] = []
        self.cache[key].append(value)

    def store(self):
        print('Writing to disk...')
        self.iter = 0
        bag = rosbag.Bag(self.save_dir, 'w')

        for key, values in self.cache.items():
            for value in values:
                bag.write(key, value)
        print('Done')
        bag.close()


    def on_shut_down(self):
        self.tr.start()
        # self.bag.close()
