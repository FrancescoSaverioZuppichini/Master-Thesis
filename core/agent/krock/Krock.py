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
        # rospy.loginfo('pose received')

    def callback_touch_sensors(self, data):
        self.state['touch_sensors'] = data
        # rospy.loginfo('touch_sensors received')

    def callback_torques_feedback(self, data):
        self.state['torques_feedback'] = data
        # rospy.loginfo('torques_feedback received')

    def move(self, gait, frontal_freq, lateral_freq, manual_mode=False):
        mode = int(manual_mode)
        msg = Float64ArrayStamped(data=[mode, gait, frontal_freq, lateral_freq])
        self.publishers['manual_control'].publish(msg)

    def spawn(self, pos=None):
        pos = PoseStamped(pose=generate_random_pose()) if pos == None else pos
        self.publishers['spawn'].publish(pos)

    def on_shut_down(self):
        self.notify('on_shut_down')

map_max_x = 5.0
map_max_y = 5.0
map_max_z = 1.0
# TODO this stuff should go in utils or in simlation class
def generate_random_pose():
    # x,y (z will be fixed as the max_hm_z so that the robot will drop down), gamma as orientation
    rn = np.random.random_sample((3,))
    random_pose = Pose()
    random_pose.position.x = 2 * map_max_x *rn[0] - map_max_x
    random_pose.position.y = 2 * map_max_x *rn[1] - map_max_x
    random_pose.position.z = map_max_z * 1.0 # spawn on the air
    qto = transformations.quaternion_from_euler(0, 0, 2*np.pi * rn[1], axes='sxyz')
    random_pose.orientation.x = qto[0]
    random_pose.orientation.y = qto[1]
    random_pose.orientation.z = qto[2]
    random_pose.orientation.w = qto[3]
    return random_pose