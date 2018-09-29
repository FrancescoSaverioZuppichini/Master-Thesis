import rospy
import numpy as np

from geometry_msgs.msg import Point, Pose, Quaternion, Twist, Wrench, PoseStamped, TwistStamped
from nav_msgs.msg import Odometry
from webots_ros.msg import Int8Stamped, Float64ArrayStamped
from sensor_msgs.msg import JointState, Joy
from std_msgs.msg import String, Header
from tf import transformations

from agent import RospyAgent

class Krock(RospyAgent):
    BASE_TOPIC = '/krock'
    POSE_SUB = '{}/pose'.format(BASE_TOPIC)
    GAIT_CHOOSER = '{}/gait_chooser'.format(BASE_TOPIC)
    MANUAL_CONTROL = '{}/manual_control_input'.format(BASE_TOPIC)
    STATUS = '{}/status'.format(BASE_TOPIC)
    SPAWN = '{}/spawn_pose'.format(BASE_TOPIC)
    TOUCH_SENSOR = '{}/touch_sensors'.format(BASE_TOPIC)
    TORQUES_FEEDBACK = '{}/torques_feedback'.format(BASE_TOPIC)

    def init_publishers(self):
        return {
            'joy' : rospy.Publisher('/joy', Joy, queue_size=1),
            'gait': rospy.Publisher(self.GAIT_CHOOSER, Int8Stamped, queue_size=1),
            'manual_control': rospy.Publisher(self.MANUAL_CONTROL, Float64ArrayStamped, queue_size=1),
            'status': rospy.Publisher(self.STATUS, String, queue_size=1),
            'spawn': rospy.Publisher(self.SPAWN, PoseStamped, queue_size=1)
        }

    def init_subscribers(self):
        rospy.Subscriber(self.POSE_SUB, PoseStamped, self.callback_pose)
        rospy.Subscriber(self.TOUCH_SENSOR, Float64ArrayStamped, self.callback_touch_sensors)
        rospy.Subscriber(self.TORQUES_FEEDBACK, Float64ArrayStamped, self.callback_torques_feedback)

    def callback_pose(self, data):
        self.state['pose'] = data

    def callback_touch_sensors(self, data):
        pass

    def callback_torques_feedback(self, data):
        pass

    def move(self, gait, frontal_freq, lateral_freq, manual_mode=False):
        mode = int(manual_mode)
        msg = Float64ArrayStamped(data=[mode, gait, frontal_freq, lateral_freq])
        self.publishers['manual_control'].publish(msg)

    def spawn(self, pos=None):
        pos = pos if pos != None else PoseStamped(pose = generate_random_pose())
        self.publishers['spawn'].publish(pos)

    def on_shut_down(self):
        self.notify('on_shut_down')
        pass
