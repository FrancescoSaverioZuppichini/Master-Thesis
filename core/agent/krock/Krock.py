import rospy
import numpy as np
import time

from geometry_msgs.msg import Point, Pose, Quaternion, Twist, Wrench, PoseStamped, TwistStamped
from nav_msgs.msg import Odometry
from webots_ros.msg import Int8Stamped, Float64ArrayStamped
from sensor_msgs.msg import JointState, Joy
from std_msgs.msg import String, Header
from tf import transformations

from agent import RospyAgent
from utils.WebotsServiceTools import Supervisor


class Krock(RospyAgent, Supervisor):
    BASE_TOPIC = '/krock'
    # sub
    POSE_SUB = '{}/pose'.format(BASE_TOPIC)
    TOUCH_SENSOR = '{}/touch_sensors'.format(BASE_TOPIC)
    TORQUES_FEEDBACK = '{}/torques_feedback'.format(BASE_TOPIC)
    # pub
    GAIT_CHOOSER = '{}/gait_chooser'.format(BASE_TOPIC)
    MANUAL_CONTROL = '{}/manual_control_input'.format(BASE_TOPIC)
    STATUS = '{}/status'.format(BASE_TOPIC)
    SPAWN = '{}/spawn_pose'.format(BASE_TOPIC)

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
        rospy.Subscriber(self.POSE_SUB, PoseStamped, self.callback_pose)
        rospy.Subscriber(self.TOUCH_SENSOR, Float64ArrayStamped, self.callback_touch_sensors)
        rospy.Subscriber(self.TORQUES_FEEDBACK, Float64ArrayStamped, self.callback_torques_feedback)

    def callback_pose(self, data):
        self.state['pose'] = data

    def callback_touch_sensors(self, data):
        self.state['touch_sensors'] = data

    def callback_torques_feedback(self, data):
        self.state['torques_feedback'] = data

    def move(self, gait, frontal_freq, lateral_freq, manual_mode=False):
        mode = int(manual_mode)
        msg = Float64ArrayStamped(data=[mode, gait, frontal_freq, lateral_freq])
        self.publishers['manual_control'].publish(msg)

    def spawn(self, pos=None):
        pos = generate_random_pose() if pos == None else pos

        # res = self.reset_simulation_physics()
        res = self.reset_simulation()
        # res = self.reload_world()

        self.set_robot_position(x=pos.position.x,
                                y=pos.position.z,
                                z=pos.position.y,
                                )

        self.set_robot_orientation(x=pos.orientation.x,
                                   y=pos.orientation.z,
                                   z=pos.orientation.y,
                                   w=pos.orientation.w)

    def die(self):
        self.notify('on_shut_down')


map_max_x = 5.0 - 0.3
map_max_y = 5.0 - 0.3
map_max_z = 0.5


# TODO this stuff should go in utils or in simlation class
def generate_random_pose():
    # x,y (z will be fixed as the max_hm_z so that the robot will drop down), gamma as orientation
    rx = np.random.uniform(-map_max_x, map_max_x)
    ry = np.random.uniform(-map_max_y, map_max_y)
    random_pose = Pose()
    random_pose.position.x = rx
    random_pose.position.y = ry
    random_pose.position.z = map_max_z * 1.0  # spawn on the air
    qto = transformations.quaternion_from_euler(0, 0, 2 * np.pi * np.random.uniform(0, 1), axes='sxyz')
    random_pose.orientation.x = qto[0]
    random_pose.orientation.y = qto[1]
    random_pose.orientation.z = qto[2]
    random_pose.orientation.w = qto[3]
    return random_pose
