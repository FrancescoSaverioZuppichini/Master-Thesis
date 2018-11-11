import rospy
import time
import numpy as np
import gym

from sensor_msgs.msg import Joy, Image
from webots import WebotsWorld
from webots.krock import Krock
from os import path
from gym import spaces
from webots.WebotsWorld import WebotsWorld
from cv_bridge import CvBridge, CvBridgeError

import cv2
import matplotlib.pyplot as plt

from simulation import SimulationException
from simulation.callbacks import OutOfMap

rospy.init_node("traversability_simulation")

class KrockWebotsEnv(gym.Env):
    metadata = { 'render_modes' : ['human']}

    def __init__(self):
        self.w = WebotsWorld.from_file(
            path.abspath('/home/francesco/Documents/Master-Thesis/core/webots/krock/krock2_ros/worlds/krock2_camera.wbt'))
        self.w()

        self.krock = Krock()
        self.krock(self.w)

        self.action_space =  spaces.Dict({
            'frontal_freq': spaces.Box(low=-1.0, high=1.0, shape=(), dtype=np.float),
            'lateral_freq': spaces.Box(low=-1.0, high=1.0, shape=(), dtype=np.float)
        })

        self.observation_space = spaces.Dict({
            'sensors': spaces.Dict({
                'position': spaces.Box(low=-5, high=5, shape=(3,)),
                'orientation': spaces.Box(low=-1, high=1, shape=(3,)),
                'front_cam': spaces.Tuple((
                    spaces.Box(low=0, high=1, shape=(600, 800, 3)),
                    spaces.Box(low=0, high=1, shape=(600, 800, 3))
                )),
            })
        })

        self.bridge = CvBridge()

        self.last_frame = None

        self.out_of_map = OutOfMap()

    def reset(self):
        self.w.spawn(self.krock)
        self.krock.stop()

    def make_obs_from_agent_state(self, agent):
        pose = agent.state['pose'].pose
        parse_pose = lambda p: [p.x, p.y, p.z]

        front_cam = None

        if 'frontal_camera' in agent.state:
            front_cam_msg = agent.state['frontal_camera']
            front_cam = self.bridge.imgmsg_to_cv2(front_cam_msg, "bgr8")

        obs = {
            'sensors': {
                'position': np.array(parse_pose(pose.position)),
                'orientation': np.array(parse_pose(pose.orientation)),
                'front_cam': front_cam
            }
        }

        return obs

    def step(self, action):
        self.krock.move(gait=1,
                        frontal_freq=action['frontal_freq'],
                        lateral_freq=action['lateral_freq'],
                        manual_mode=True)

        self.krock.sleep()

        obs = self.make_obs_from_agent_state(self.krock)

        self.last_frame = obs['sensors']['front_cam']

        return obs, 0,  self.done, {}

    @property
    def is_out_of_map(self):
        is_out_of_map = False

        try:
            self.out_of_map.tick(None, self.w, self.krock)
        except SimulationException:
            is_out_of_map = True

        return is_out_of_map

    def is_get_stuck(self):
        return False

    @property
    def done(self):
        return self.is_out_of_map

    def render(self, mode='human'):
        if self.last_frame is not None:
            cv2.imshow('env', self.last_frame)
            cv2.waitKey(1)


env = KrockWebotsEnv()

GO_FORWARD = {
            'frontal_freq': 1,
            'lateral_freq': 0
}

for _ in range(100):
    env.reset()
    for _ in range(100000):
        env.render()
        # action = env.action_space.sample()
        obs, r, done, _ = env.step(GO_FORWARD)

        if done:
            break

env.reset()
