import sys

sys.path.append("../")
import rospy
import tqdm
import random

import time
import pandas as pd
import os
from simulation.agent.callbacks import *
from simulation.env.webots.krock import KrockWebotsEnv
from simulation.env.spawn import spawn_points2webots_pose, RandomSpawnStrategy
from utilities.utils import hmread

class Simulation():
    """
    This class run an agent on an environment multiple times. It uses a spawn stategy to define candidate
    spawing points and then run the robot for a certain amount of time the terrain.
    TODO This class is not so generic, it would be better to also pass an env to make it as generic as possible or have a constructor like .from_webots_env

    """
    REANIMATE_EVERY = 20

    def __init__(self, map_path, n, height=1,  out_dir='/tmp/',
                 spawn_strategy=RandomSpawnStrategy,
                 max_time=20):

        self.map_path = map_path
        self.map = hmread(map_path)

        self.n, self.height, self.out_dir = n, height, out_dir
        self.max_time = max_time
        self.meta = None
        self.spawn_strategy = spawn_strategy

    def __call__(self):
        rospy.init_node("traversability_simulation")

        self.spanwer = self.spawn_strategy(self.map)
        spawn_points = self.spanwer(self.n)
        self.env = KrockWebotsEnv.from_numpy(
            self.map,
            KrockWebotsEnv.WORLD_PATH,
            {'height': self.height,
             'resolution': 0.02},
            output_path=path.abspath(
                path.join(path.dirname(__file__), './env/webots/krock/krock2_ros/worlds/{}.wbt').format(
                    self.map_name)),
            agent_callbacks=[RosBagSaver(self.agent_out_dir, topics=['pose'])]
        )

        n_sim_bar = tqdm.tqdm(range(self.n), leave=False)

        for i in n_sim_bar:
            spawn_point = spawn_points[i]
            # if we have more simulation than spawn points just select one random
            if i > len(spawn_points): spawn_point = random.choice(spawn_points)
            # we have to convert the spawn points coordinate to webots
            pose = spawn_points2webots_pose(spawn_point, self.env)
            self.env.reset(pose=pose)
            # one in a while we recreate the robot to be sure it is not damaged
            if i % self.REANIMATE_EVERY == 0:
                n_sim_bar.set_description('Reanimate robot')
                self.env.reanimate()
            self.run_env_for(self.max_time)
            # We now need to kill the agent and pass the arguments to the callbacks
            file_name = '{}-{}-{}'.format(self.map_name, self.height, time.time())
            self.env.agent.die(self.env, file_name=file_name)
            # store the meta information to self.out_dir
            self.to_meta(file_name)

    def run_env_for(self, seconds):
        elapsed = 0
        start = time.time()

        while elapsed <= (int(seconds)):
            obs, r, done, _ = self.step()
            elapsed = time.time() - start

            if done: break

    def step(self):
        return self.env.step(self.env.GO_FORWARD)

    def to_meta(self, file_name, append=True):
        df = pd.DataFrame(data={'filename': [file_name],
                                'map': [self.map_name],
                                'height': [self.height]})

        if self.meta is None or append is False:
            try:
                self.meta = pd.read_csv('{}/meta.csv'.format(self.out_dir), index_col=[0])
                self.meta = pd.concat([self.meta, df])
                self.meta = self.meta.reset_index(drop=True)
            except FileNotFoundError:
                self.meta = df
        else:
            self.meta = pd.concat([self.meta, df])
            self.meta = self.meta.reset_index(drop=True)
        self.meta.to_csv('{}/meta.csv'.format(self.out_dir))

    @property
    def map_name(self):
        return path.splitext(path.basename(self.map_path))[0]

    @property
    def agent_out_dir(self):
        agent_out_dir = self.out_dir + '/bags'
        os.makedirs(agent_out_dir, exist_ok=True)

        return agent_out_dir
