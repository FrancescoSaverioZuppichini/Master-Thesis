import rosbag
import threading
import rospy
import time

from .AgentCallback import AgentCallback
from pypeln import thread as th

class RosBagSaver(AgentCallback):
    def __init__(self, save_dir, topics=None, max_size=1024, workers=2):
        self.save_dir = save_dir
        self.topics = topics
        self.cache = {}
        self.max_size = max_size
        self.size = 0
        self.workers = workers

    def on_state_change(self, key, value):
        store = True

        if self.topics != None: store = key in self.topics

        if key not in self.cache: self.cache[key] = []

        if store: self.cache[key].append(value)

        self.size = len(self.cache[key])

        if self.size == self.max_size: self.store()

    def write(self, data):
        # file_name = self.save_dir + '/{}.bag'.format(time.time())

        key, values = data
        for value in values:
            self.bag.write(key, value)

        return True

    def store(self):
        data = self.cache.items()
        file_name = self.save_dir + '/{}.bag'.format(time.time())

        self.bag = rosbag.Bag(file_name, 'w')

        stage = th.map(self.write, data, workers=self.workers)
        files_list = list(stage)

        self.bag.close()

        rospy.loginfo('Wrote bag file to disk.')
        # clear cache
        self.cache = {}
        self.size = 0

    def on_shut_down(self):
        self.store()

# import rosbag
# import threading
# import tqdm
# import time
#
# from .AgentCallback import AgentCallback
#
#
# class RosBagSaver(AgentCallback):
#     def __init__(self, save_dir, topics=None):
#         self.save_dir = save_dir
#         self.topics = topics
#         self.cache = {}
#         self.max_size = 16
#         self.iter = 0
#         self.tr = threading.Thread(target=self.store)
#
#     def on_state_change(self, key, value):
#         store = True
#         if self.topics != None: store = key in self.topics
#         if key not in self.cache: self.cache[key] = []
#         self.cache[key].append(value)
#
#     def store(self):
#         self.iter = 0
#         bag = rosbag.Bag(self.save_dir, 'w')
#         # bar = tqdm.tqdm(self.cache.items())
#         # bar.set_description('Writing to disk...')
#         # TODO parallize this loop!
#         for key, values in self.cache.items():
#             for value in values:
#                 bag.write(key, value)
#         bag.close()
#
#     def on_shut_down(self):
#         self.tr.start()
#         self.tr.join()
#         self.tr = threading.Thread(target=self.store)
#         # self.bag.close()
#
