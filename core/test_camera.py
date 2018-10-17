import rospy

from sensor_msgs.msg import Image
from webots_ros.srv import *
from webots_ros.msg import Float64ArrayStamped

FRONTAL_CAMERA = '/krock/front_camera/image_throttle'
MANUAL_CONTROL = '/krock/manual_control_input'

rospy.init_node("test_camera")


def camera_cb(data):
    rospy.loginfo('camera data received.')

mover = rospy.Publisher(MANUAL_CONTROL, Float64ArrayStamped, queue_size=1)

try:
    request_enable = rospy.ServiceProxy('/krock/front_camera/enable', set_int)
    res = request_enable(1)
    rospy.loginfo(res)
except rospy.ServiceException as e:
    rospy.logerr(e)

rospy.Subscriber(FRONTAL_CAMERA, Image, camera_cb)

nap = rospy.Rate(10)

while True:
    data = Float64ArrayStamped(data=[1, 1, 1.0, 0])
    mover.publish(data)
    nap.sleep()
