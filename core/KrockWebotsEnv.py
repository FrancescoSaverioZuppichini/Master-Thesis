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
import pprint

rospy.init_node("traversability_simulation")


class KrockWebotsEnv(gym.Env):
    metadata = {'render_modes': ['human']}

    GO_FORWARD = {
        'frontal_freq': 1,
        'lateral_freq': 0
    }

    def __init__(self):
        self.world = WebotsWorld.from_file(
            path.abspath('/home/francesco/Documents/Master-Thesis/core/webots/krock/krock2_ros/worlds/bumps3.wbt'))
        self.world()

        self.agent = Krock()
        self.agent(self.world)

        self.action_space = spaces.Dict({
            'frontal_freq': spaces.Box(low=-1.0, high=1.0, shape=(), dtype=np.float),
            'lateral_freq': spaces.Box(low=-1.0, high=1.0, shape=(), dtype=np.float)
        })

        self.observation_space = spaces.Dict({
            'sensors': spaces.Dict({
                'position': spaces.Dict({
                    'x': spaces.Box(low=self.world.x[0], high=self.world.x[1], shape=(1,), dtype=np.float),
                    'y': spaces.Box(low=self.world.y[0], high=self.world.y[1], shape=(1,), dtype=np.float),
                    'z': spaces.Box(low=-0, high=self.world.z, shape=(1,), dtype=np.float),

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

        self.done = IfOneFalseOf([IsInside(), IsNotStuck()])

    def reset(self):
        self.world.spawn(self.agent)
        self.agent.stop()
        return self.make_obs_from_agent_state(self.agent)

    def make_obs_from_agent_state(self, agent):
        pose = agent.state['pose'].pose
        parse_pose = lambda p: [p.x, p.y, p.z]

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
        self.agent.stop()

        obs = self.make_obs_from_agent_state(self.agent)
        # the last frame will be used in the `.render` function
        self.last_frame = obs['sensors']['front_cam']

        return obs, 0, self.done(self), {}

    def render(self, mode='human'):
        if self.last_frame is not None:
            cv2.imshow('env', self.last_frame)
            cv2.waitKey(1)


env = KrockWebotsEnv()
obs = env.observation_space

for _ in range(1):
    env.reset()

    for _ in range(1000):
        env.render()
        # action = env.action_space.sample()
        obs, r, done, _ = env.step(env.GO_FORWARD)
        # pprint.pprint(obs)
        print(done)
        if done:
            break

env.reset()
