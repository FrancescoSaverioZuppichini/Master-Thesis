import rosbag
import threading
import rospy
import time

from .AgentCallback import AgentCallback
from pypeln import thread as th

class RosBagSaver(AgentCallback):
    def __init__(self, save_dir, topics=None, max_size=256, workers=2):
        self.save_dir = save_dir
        self.topics = topics
        self.cache = {}
        self.max_size = max_size
        self.size = 0
        self.tr = threading.Thread(target=self.store)
        self.workers = workers

    def on_state_change(self, key, value):
        store = True
        if self.topics != None: store = key in self.topics
        if key not in self.cache: self.cache[key] = []
        if store: self.cache[key].append(value)
        self.size += 1

        if self.size == self.max_size: self.store()

    def write(self, data):
        file_name = self.save_dir + '/{}.bag'.format(time.time())
        bag = rosbag.Bag(file_name, 'w')

        key, values = data
        for value in values:
            bag.write(key, value)

        bag.close()

        return file_name

    def store(self):
        data = self.cache.items()

        stage = th.map(self.write, data, workers=self.workers)
        res = list(stage)

        print(res)

        rospy.loginfo('Wrote bag file to disk.')
        # clear cache
        self.cache = {}
        self.size = 0

    def on_shut_down(self):
        self.tr.start()
        self.tr.join()
        self.tr = threading.Thread(target=self.store)
        # self.bag.close()
