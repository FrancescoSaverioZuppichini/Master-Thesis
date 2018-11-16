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

from simulation.conditions import  *

from webots.WebotsEnv import WebotsEnv

import pprint

class SimulationEnv(gym.Env):
    def __init__(self, world, agent, stop):
        self.world, self.agent, self.stop = world, agent, stop

class KrockWebotsEnv(WebotsEnv, Krock):
    metadata = {'render_modes': ['human']}

    GO_FORWARD = {
        'frontal_freq': 1,
        'lateral_freq': 0
    }

    STOP = {
        'frontal_freq': 0,
        'lateral_freq': 0
    }

    def __init__(self, world_path, agent_callbacks=[], *args, **kwargs):
        super().__init__(world_path, *args, **kwargs)

        self.agent = Krock()
        self.agent.add_callbacks(agent_callbacks)
        self.agent()

        self.action_space = spaces.Dict({
            'frontal_freq': spaces.Box(low=-1.0, high=1.0, shape=(), dtype=np.float),
            'lateral_freq': spaces.Box(low=-1.0, high=1.0, shape=(), dtype=np.float)
        })

        self.observation_space = spaces.Dict({
            'sensors': spaces.Dict({
                'position': spaces.Dict({
                    'x': spaces.Box(low=self.x[0], high=self.x[1], shape=(1,), dtype=np.float),
                    'y': spaces.Box(low=self.y[0], high=self.y[1], shape=(1,), dtype=np.float),
                    'z': spaces.Box(low=-0, high=self.z, shape=(1,), dtype=np.float),

                }),
                'orientation': spaces.Dict({
                    'x': spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float),
                    'y': spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float),
                    'z': spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float),
                    'w': spaces.Box(low=-np.pi, high=np.pi, shape=(1,), dtype=np.float),

                }),
                'front_cam': spaces.Tuple((
                    spaces.Box(low=0, high=1, shape=(600, 800, 3), dtype=np.int),
                    spaces.Box(low=0, high=1, shape=(600, 800, 3), dtype=np.int)
                )),
            })
        })

        self.bridge = CvBridge()

        self.last_frame = None

    def __call__(self, *args, **kwargs):
        super()()


    def make_obs_from_agent_state(self, agent):
        """
        Convert the ROS msg store in the agent state to the correct JSON
        representation according to the observation space
        :param agent:
        :return:
        """
        pose = agent.state['pose'].pose

        front_cam = None

        # due to ros lag, it may happened that we have to still receive a camera img
        if 'frontal_camera' in agent.state:
            front_cam_msg = agent.state['frontal_camera']
            front_cam = self.bridge.imgmsg_to_cv2(front_cam_msg, "bgr8")
            front_cam = cv2.cvtColor(front_cam, cv2.COLOR_RGB2GRAY)

        obs = {
            'sensors': {
                'position': {
                    'x': pose.position.x,
                    'y': pose.position.y,
                    'z': pose.position.z
                },
                'orientation': {
                    'x': pose.orientation.x,
                    'y': pose.orientation.y,
                    'z': pose.orientation.z,
                    'w': pose.orientation.w

                },
                'front_cam': front_cam
            }
        }

        return obs

    def step(self, action):
        """
        This function moves the krock, then it waits for krock.sleep time and get the last
        information in the krock state.
        :param action:
        :return:
        """
        self.agent.move(gait=1,
                        frontal_freq=action['frontal_freq'],
                        lateral_freq=action['lateral_freq'],
                        manual_mode=True)

        self.agent.sleep()
        # self.agent.stop()

        obs = self.make_obs_from_agent_state(self.agent)
        # the last frame will be used in the `.render` function
        self.last_frame = obs['sensors']['front_cam']

        done = False

        if self.should_stop != None: done = self.should_stop(self)

        return obs, 0, done, {}

    def render(self, mode='human'):
        if mode == 'human':
            if self.last_frame is not None:
                cv2.imshow('env', self.last_frame)
                cv2.waitKey(1)


    def reset(self, hard=True):
        self.agent.stop()
        if hard:
            self.agent()
            self.spawn(self.agent)
            self.agent.sleep()

        self.should_stop = IfOneFalseOf([IsNotStuck(n_last=50), IsInside()])

        return self.make_obs_from_agent_state(self.agent)
