import rosbag
import threading
import rospy
import time

import pandas as pd

from .AgentCallback import AgentCallback
from pypeln import thread as th

class RosBagSaver(AgentCallback):
    FILENAME2MAP_PATH = './filename2map.csv'

    def __init__(self, save_dir, topics=None, max_size=1024, workers=2):
        self.save_dir = save_dir
        self.topics = topics
        self.cache = {}
        self.max_size = max_size
        self.size = 0
        self.workers = workers

    def create_or_update_filename2map(self, filename, map):
        new = pd.DataFrame({'filename': [filename], 'map' : [map] })

        try:
            old = pd.read_csv(self.FILENAME2MAP_PATH)
            old = pd.concat([old,new], sort=False, join_axes=[new.columns], ignore_index=True)
        except FileNotFoundError:
            old = new
        old.to_csv(self.FILENAME2MAP_PATH)

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

    def store(self, agent):
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

        self.create_or_update_filename2map(file_name, agent.world.world_path)

    def on_shut_down(self):
        self.store(self.agent)
