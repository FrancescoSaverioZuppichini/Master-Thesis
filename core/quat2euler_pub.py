import rospy
import tf

from geometry_msgs.msg import PoseStamped
rospy.init_node('quat2euler')


def quat2eul(msg, _):
    br = tf.TransformBroadcaster()
    br.sendTransform()

rospy.Subscriber('/krock/pose', PoseStamped, quat2eul)

rospy.spin()