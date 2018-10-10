import rosbag
import threading
import tqdm

from .AgentCallback import AgentCallback


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
        self.iter = 0
        bag = rosbag.Bag(self.save_dir, 'w')
        # bar = tqdm.tqdm(self.cache.items())
        # bar.set_description('Writing to disk...')
        # TODO parallize this loop!
        for key, values in self.cache.items():
            for value in values:
                bag.write(key, value)
        bag.close()

    def on_shut_down(self):
        self.tr.start()
        self.tr.join()
        self.tr = threading.Thread(target=self.store)
        # self.bag.close()
