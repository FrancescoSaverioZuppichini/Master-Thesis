import rospy
from geometry_msgs.msg import Point, Pose, Quaternion, Twist, Wrench, PoseStamped, TwistStamped
from nav_msgs.msg import Odometry
from webots_ros.msg import Int8Stamped
from sensor_msgs.msg import JointState, Joy
from std_msgs.msg import String, Header
import numpy as np
import rosbag

class Callbackable():
    def set_callbacks(self, callbacks):
        self.callbacks = callbacks

    def add_callback(self, callback):
        self.callbacks.append(callback)

    def notify(self, event_name, *args, **kwargs):
        for hook in self.callbacks:
            getattr(hook, event_name)(*args, **kwargs)

class Agent():
    def __init__(self):
        self.state = {}

    def spawn(self, pos):
        pass

    def move(self, *args, **kwargs):
        pass

    def stop(self):
        pass

class RospyAgent(Agent, Callbackable):
    def __init__(self, rate=None):
        super().__init__()
        self.rate = rospy.Rate(hz=10) if rate == None else rate
        self.state = AgentState(self)
        self.set_callbacks([])

    def __call__(self, *args, **kwargs):
        self.subscribers = self.init_subscribers()
        self.publishers = self.init_publishers()

    def init_publishers(self):
        return {}

    def init_subscribers(self):
        return {}

class AgentCallback():
    def on_state_change(self, key, value):
        pass
    def on_subscribe(self, topic, data):
        pass

    def on_publish(self, topic, data):
        pass

    def on_shut_down(self):
        pass

class RosbagCallback(AgentCallback):
    def __init__(self, save_dir, topics=None):
        self.bag = rosbag.Bag(save_dir, 'w')
        self.topics = topics

    def on_state_change(self, key, value):
        store = True
        if self.topics != None: store = key in self.topics
        if store: self.bag.write(key, value)

    def on_shut_down(self):
        self.bag.close()

class AgentState(dict):
    def __init__(self, agent: RospyAgent):
        super().__init__()
        self.agent = agent

    def __setitem__(self, key, value):
        super().__setitem__(key, value)
        self.agent.notify('on_state_change', key, value)

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

rospy.init_node("record_single_trajectory")

r = rospy.Rate(hz=10)

k = Krock()

k.add_callback(RosbagCallback('./data.bag'))
k()

while not rospy.is_shutdown():
    k.move(2)
    r.sleep()

rospy.on_shutdown(k.on_shut_down)
