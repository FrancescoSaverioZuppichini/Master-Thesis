import rospy

from geometry_msgs.msg import Point, Pose, Quaternion, Twist, Wrench, PoseStamped, TwistStamped
from nav_msgs.msg import Odometry
from webots_ros.msg import Int8Stamped
from sensor_msgs.msg import JointState, Joy
from std_msgs.msg import String, Header
import numpy as np
from agent import RospyAgent

class Krock(RospyAgent):
    BASE_TOPIC = '/krock'
    POSE_SUB = '{}/pose'.format(BASE_TOPIC)
    GAIT_CHOOSER = '{}/gait_chooser'.format(BASE_TOPIC)

    def init_publishers(self):
        return {
            'joy' : rospy.Publisher('/joy', Joy, queue_size=1),
            'gait': rospy.Publisher(self.GAIT_CHOOSER, Int8Stamped, queue_size=1)

        }

    def init_subscribers(self):
        rospy.Subscriber(self.POSE_SUB, PoseStamped, self.callback_pose)
        rospy.Subscriber('/joy', Joy, self.callback_joy)

    def callback_joy(self, data):
        self.state['joy'] = data


    def callback_pose(self, data):
        self.state['pose'] = data
        # rospy.loginfo("Pos /data received")
        # self.state['pose'] = data

        return 0

    def move(self, gait=1):
        # self.publishers['gait'].publish(Int8Stamped(data=gait))
        buttons = np.zeros(11, dtype=np.int)
        # buttons[1] = 1
        # buttons[2] = 0

        msg = Joy(axes=[0,-0.5,0.5,0],
                  buttons=list(buttons))

        self.publishers['joy'].publish(msg)
        # print('pub')

    def on_shut_down(self):
        self.notify('on_shut_down')
        pass