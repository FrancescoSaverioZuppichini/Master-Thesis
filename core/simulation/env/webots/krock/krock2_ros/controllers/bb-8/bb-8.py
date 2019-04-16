"""my controller description."""

from controller import *
import rospy
from geometry_msgs.msg import PoseStamped
from controller import Node
from tf.transformations import quaternion_from_euler
from math import sin, cos

robot = Supervisor()

node = robot.getFromDef('head')

print(node)
body_yaw_motor = robot.getMotor('body yaw motor')
body_pitch_motor = robot.getMotor('body pitch motor')
head_yaw_motor = robot.getMotor("head yaw motor");

timestep = int(robot.getBasicTimeStep())

rospy.init_node('bb8', anonymous=True)
rate = rospy.Rate(10) # 10hz

body_yaw_motor.setPosition(999999999999999999999)
body_pitch_motor.setPosition(999999999999999999999999)
head_yaw_motor.setPosition(99999999999999999999)
pub = rospy.Publisher('bb8/pose', PoseStamped, queue_size=10)

body_yaw_motor.setVelocity(0)
body_pitch_motor.setVelocity(0)
head_yaw_motor.setVelocity(0)

while(robot.step(timestep) != -1):
    head_yaw_motor.setVelocity(0)
    body_yaw_motor.setVelocity(0)
    body_pitch_motor.setVelocity(0)
    
    node_pos = node.getPosition()
    orientation_values = node.getField('rotation').getSFRotation()
    pose = PoseStamped()
    pose.header.stamp = rospy.Time.now()
    pose.header.frame_id = 'map'
    pose.pose.position.x = node_pos[0]
    pose.pose.position.y = node_pos[2]
    pose.pose.position.z = node_pos[1]
    
    # print(orientation_values)
    
    a = orientation_values[3];

    pose.pose.orientation.x =  sin(a/2)* orientation_values[0]
    pose.pose.orientation.y =  sin(a/2)* - orientation_values[2]
    pose.pose.orientation.z =  sin(a/2)* orientation_values[1]
    pose.pose.orientation.w = cos(a/2)
  
  
    # pose.pose.orientation.x =  orientation_values[0]
    # pose.pose.orientation.y =  orientation_values[1]
    # pose.pose.orientation.z =  orientation_values[2]
    # pose.pose.orientation.w = a
          
    pub.publish(pose)
    # rate.sleep()


    # head_yaw_motor.setVelocity(4.0)