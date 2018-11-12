import rospy
import numpy as np

from geometry_msgs.msg import PoseStamped
from webots_ros.msg import Int8Stamped, Float64ArrayStamped
from sensor_msgs.msg import Joy, Image
from std_msgs.msg import String

from agent import RospyAgent
from utils.webots2ros import Supervisor

from cv_bridge import CvBridge


class Krock(RospyAgent, Supervisor):
    BASE_TOPIC = '/krock'
    # pub
    GAIT_CHOOSER = '{}/gait_chooser'.format(BASE_TOPIC)
    MANUAL_CONTROL = '{}/manual_control_input'.format(BASE_TOPIC)
    STATUS = '{}/status'.format(BASE_TOPIC)
    SPAWN = '{}/spawn_pose'.format(BASE_TOPIC)
    # sub
    POSE_SUB = '{}/pose'.format(BASE_TOPIC)
    TOUCH_SENSOR = '{}/touch_sensors'.format(BASE_TOPIC)
    TORQUES_FEEDBACK = '{}/torques_feedback'.format(BASE_TOPIC)
    FRONTAL_CAMERA = '{}/front_camera/image'.format(BASE_TOPIC)

    def __init__(self):
        super().__init__()
        self.name = self.BASE_TOPIC
        self.get_robot_node()

    def init_publishers(self):
        return {
            'joy': rospy.Publisher('/joy', Joy, queue_size=1),
            'gait': rospy.Publisher(self.GAIT_CHOOSER, Int8Stamped, queue_size=1),
            'manual_control': rospy.Publisher(self.MANUAL_CONTROL, Float64ArrayStamped, queue_size=1),
            'status': rospy.Publisher(self.STATUS, String, queue_size=1),
            'spawn': rospy.Publisher(self.SPAWN, PoseStamped, queue_size=1)
        }

    def init_subscribers(self):
        return {
            'pose': rospy.Subscriber(self.POSE_SUB, PoseStamped, self.callback_pose),
            'touch_sensor': rospy.Subscriber(self.TOUCH_SENSOR, Float64ArrayStamped, self.callback_touch_sensors),
            'toques_feedback': rospy.Subscriber(self.TORQUES_FEEDBACK, Float64ArrayStamped,
                                                self.callback_torques_feedback),
            'frontal_camera': rospy.Subscriber(self.FRONTAL_CAMERA, Image, self.callbacks_frontal_camera)
        }

    def callback_pose(self, data):
        self.state['pose'] = data

    def callback_touch_sensors(self, data):
        self.state['touch_sensors'] = data

    def callback_torques_feedback(self, data):
        self.state['torques_feedback'] = data

    def callbacks_frontal_camera(self, data):
        self.state['frontal_camera'] = data

    def move(self, gait, frontal_freq, lateral_freq, manual_mode=False):
        mode = int(manual_mode)
        msg = Float64ArrayStamped(data=[mode, gait, frontal_freq, lateral_freq])
        self.publishers['manual_control'].publish(msg)

    def spawn(self, world, pos=None, *args, **kwargs):
        pos = world.random_position if pos == None else pos

        self.set_robot_position(x=pos.position.x,
                                y=pos.position.z,
                                z=pos.position.y)

        self.set_robot_orientation(x=pos.orientation.x,
                                   y=pos.orientation.z,
                                   z=pos.orientation.y,
                                   w=pos.orientation.w)

        self.reset_node_physics(self.robot_node)

    def stop(self):
        self.move(gait=1,
                  frontal_freq=0,
                  lateral_freq=0,
                  manual_mode=True)

    def act(self, sim, world, *args, **kwargs):
        self.move(gait=1,
                  frontal_freq=1.0,
                  lateral_freq=0,
                  manual_mode=True)
