import glob
import random

from os import makedirs
import sys
sys.path.append("../")

from simulation.agent.callbacks import *
from simulation.env.webots.krock import KrockWebotsEnv
from simulation.env.spawn import FlatGroundSpawnStrategy, spawn_points2webots_pose, RandomSpawnStrategy
from utilities import run_for
import tqdm
import random
import cv2
import numpy as np
from utilities.patches.texture import get_rocks

def hmread(hm_path):
    hm = cv2.imread(hm_path)
    print(hm_path)
    hm = cv2.cvtColor(hm, cv2.COLOR_BGR2GRAY)
    return hm

def make_env(map, args):
    agent = None
    env = None

    map_name, _ = path.splitext(path.basename(map))
    bags_map_dir = path.normpath(args.save_dir + '/' + map_name)

    if args.engine == 'webots':
        if args.robot == 'krock':
            height = args.height
            # height = random.randint(2,5)
            # makedirs(bags_map_dir, exist_ok=True)
            env = KrockWebotsEnv.from_image(
                map,
                path.abspath('./env/webots/krock/krock_no_tail.wbt'),
                {'height': height,
                 'resolution': 0.02},
                output_dir=path.abspath('./env/webots/krock/krock2_ros/worlds/'),
                agent_callbacks=[RosBagSaver(args.save_dir, topics=['pose'])]
            )

    return env, map_name, bags_map_dir, height

class Simulation():
    def __init__(self, map_path, n, height, bags_dirs,
                 spawn_strategy=RandomSpawnStrategy,
                 max_time=20,
                 texture=None,
                 name='sim'):

        self.map_path = map_path
        self.map = hmread(map_path)
        self.map_name = path.splitext(path.basename(self.map_path))[0]

        self.n, self.height, self.bags_dir, self.name = n, height, bags_dirs, name
        self.texture = texture
        self.max_time = max_time
        self.meta = None
        self.spawn_strategy = spawn_strategy

    def __call__(self):
        rospy.init_node("traversability_simulation")
        self.spawn_strategy = self.spawn_strategy(self.map)

        temp = self.map.copy().astype(np.float32) * self.height
        if self.texture is not None:
            temp += self.texture * 255

        env = KrockWebotsEnv.from_numpy(
            temp,
            path.abspath('/home/francesco/Documents/Master-Thesis/core/simulation/env/webots/krock/krock_no_tail.wbt'),
            {'height': 1,
             'resolution': 0.02},
            output_path=path.abspath('/home/francesco/Documents/Master-Thesis/core/simulation/env/webots/krock/krock2_ros/worlds/{}.wbt'.format(self.map_name)),
            agent_callbacks=[RosBagSaver(self.bags_dir, topics=['pose'])]
        )

        spawn_points = self.spawn_strategy(self.n)

        n_sim_bar = tqdm.tqdm(range(self.n), leave=False)

        for i in n_sim_bar:
            spawn_point = spawn_points[i]
            if i > len(spawn_points): spawn_point = random.choice(spawn_points)

            env.reset(pose=spawn_points2webots_pose(spawn_point, env))

            if i % 20 == 0:
                n_sim_bar.set_description('Reanimate robot')
                env.reanimate()

            elapsed = 0
            start = time.time()

            while elapsed <= (int(self.max_time)):
                obs, r, done, _ = env.step(env.GO_FORWARD)
                elapsed = time.time() - start

                if done: break

            # we want to store after each each spawn
            # map_name = self.name
            file_name = '{}-{}-{}'.format(self.map_name, self.height, time.time())
            env.agent.die(env, file_name)

            temp = pd.DataFrame(data={'filename': [file_name],
                                      'map': [self.map_name],
                                      'height': [self.height]})

            if self.meta is None:
                try:
                    self.meta = pd.read_csv('{}/meta.csv'.format(self.bags_dir), index_col=[0])
                    self.meta = pd.concat([self.meta, temp])
                    self.meta = self.meta.reset_index(drop=True)
                except FileNotFoundError:
                    self.meta = temp
            else:
                self.meta = pd.concat([self.meta, temp])
                self.meta = self.meta.reset_index(drop=True)
            self.meta.to_csv('{}/meta.csv'.format(self.bags_dir))


        # @classmethod
        # def from_config(cls, config, maps_dir, *args, **kwargs):
        #     simulations = []
        #
        #     for map_name, config in config.items():
        #         cls(maps_dir + map_name + '.png',
        #             config.n,
        #             config.height)
        #     return cls()
#
# rocks1, rocks2, rocks3 = get_rocks((513, 513))
# #
