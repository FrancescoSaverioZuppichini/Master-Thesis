import rosbag
import threading
import rospy
import time

from os import path

import pandas as pd

from .AgentCallback import AgentCallback
from pypeln import thread as th

# REVIEW: probably store all topic in the bag is not the best idea due to the fact that
# if we want to just load one topic we are forced to load them all!
class RosBagSaver(AgentCallback):

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
        # REVIEW: for now we want to store only in the end
        # if self.size == self.max_size: self.store(agent)

    def write(self, data):
        key, values = data
        for value in values:
            self.bag.write(key, value)
        return True

    def store(self, ctx):
        file_name = path.normpath(self.save_dir + '/{}.bag'.format(time.time()))

        self.bag = rosbag.Bag(file_name, 'w')

        for key in self.cache.keys():
            for value in self.cache[key]:
                self.bag.write(key, value)

        # self.write(data)
        # stage = th.map(self.write, data, workers=self.workers)
        # files_list = list(stage)

        self.bag.close()

        rospy.loginfo('Wrote bag file to disk.')
        # clear cache
        self.cache = {}
        self.size = 0


    def on_shut_down(self):
        # print(len(self.cache['pose']))
        self.tr.start()
        self.tr.join()
        self.tr = threading.Thread(target=self.store, args=(self,))

