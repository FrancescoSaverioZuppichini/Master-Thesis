import roslib
import rospy
from geometry_msgs.msg import PoseStamped
from tf.transformations import euler_from_quaternion

import tf

def quaternion2euler(msg):
    br = tf.TransformBroadcaster()
    print(msg)
    br.sendTransform(msg.pose.position,
                     euler_from_quaternion(msg.pose.orientation),
                     rospy.Time.now(),
                     'krock',
                     "world")

if __name__ == '__main__':
    rospy.init_node('turtle_tf_broadcaster')
    rospy.Subscriber('/krock/pose',
                     PoseStamped,
                     quaternion2euler)
    rospy.spin()