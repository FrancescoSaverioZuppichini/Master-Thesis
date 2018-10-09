#!/usr/bin/env python
#
# implementation of service calls using the supervisor API of webots
#
import sys
import rospy
import time
import numpy as np
import math

# API reference for supervisor services exposed from webots
# https://www.cyberbotics.com/doc/reference/supervisor

# These message type definitions are needed to request or set values
# to some webots services. These requirements are found in the API
# reference.
from geometry_msgs.msg import Quaternion, Vector3

# These service are need for calling and receiving services
# If a new service type is needed, verify the API reference and add it
# here. I have included such files in the srv folder of this package
# If the srv is not found, verify that it appears in the service
# generation segment in the CMakeLists.txt of this package
from webots_ros.srv import get_bool, get_int, get_uint64, set_int, node_get_field, field_get_node, field_get_vec3f, node_get_position, node_get_orientation, field_set_vec3f, field_set_rotation

class ServiceTools:
    def __init__(self, model_name):
        # constants of the robot
        self.model_name = model_name

    def get_world_node(self):
        service = self.model_name+'/supervisor/get_root'
        rospy.loginfo("Waiting for service %s", service)
        rospy.wait_for_service(service)
        try:
            get_world = rospy.ServiceProxy(service, get_uint64)
            self.world = get_world()
            print (self.world)
        except rospy.ServiceException as e:
            print ("Service call failed: ", e)

    def get_robot_node(self):
        service = self.model_name+'/supervisor/get_self'
        rospy.loginfo("Waiting for service %s", service)
        rospy.wait_for_service(service)
        try:
            get_self = rospy.ServiceProxy(service, get_uint64)
            self.robot_node = get_self()
            print (self.robot_node)
        except rospy.ServiceException as e:
            print ("Service call failed: ", e)

    def get_robot_position(self):
        service = self.model_name+'/supervisor/node/get_position'
        rospy.loginfo("Waiting for service %s", service)
        rospy.wait_for_service(service)
        try:
            request_position = rospy.ServiceProxy(service, node_get_position)
            position = request_position(self.robot_node.value)
            print (position)

        except rospy.ServiceException as e:
            print ("Service call failed: ", e)

    def get_robot_orientation(self):
        service = self.model_name+'/supervisor/node/get_orientation'
        rospy.loginfo("Waiting for service %s", service)
        rospy.wait_for_service(service)
        try:
            request_position = rospy.ServiceProxy(service, node_get_orientation)
            orientation = request_position(self.robot_node.value)
            print (orientation)

        except rospy.ServiceException as e:
            print ("Service call failed: ", e)

    def set_robot_position(self):
        service = self.model_name+'/supervisor/node/get_field'
        rospy.loginfo("Waiting for service %s", service)
        rospy.wait_for_service(service)
        try:
            request_field = rospy.ServiceProxy(service, node_get_field)
            field = request_field(self.robot_node.value, 'translation')
            print (field)

            other_service = self.model_name+'/supervisor/field/set_vec3f'
            rospy.loginfo("Waiting for service %s", other_service)
            rospy.wait_for_service(other_service)
            try:
                set_field = rospy.ServiceProxy(other_service, field_set_vec3f)
                new_position = Vector3()
                # webots position, also called translation is different from ROS
                # x -> forward, z-> right, y -> upward
                new_position.x = 0.1
                new_position.y = 0.3
                new_position.z = 0.0
                resp = set_field(field.field, 0, new_position)
                print (resp)
            except rospy.ServiceException as e:
                print ("Service call failed: ", e)

        except rospy.ServiceException as e:

            print ("Service call failed: ", e)

    def set_robot_orientation(self):
        service = self.model_name+'/supervisor/node/get_field'
        rospy.loginfo("Waiting for service %s", service)
        rospy.wait_for_service(service)
        try:
            request_field = rospy.ServiceProxy(service, node_get_field)
            field = request_field(self.robot_node.value, 'rotation')
            print (field)

            other_service = self.model_name+'/supervisor/field/set_rotation'
            rospy.loginfo("Waiting for service %s", other_service)
            rospy.wait_for_service(other_service)
            try:
                set_field = rospy.ServiceProxy(other_service, field_set_rotation)
                new_orientation = Quaternion()
                # remember that the rotation in webots is different from
                # rotation in ROS. Webots x,y,z is in m and w is in rad
                # (yaw)
                new_orientation.x = 0.0
                new_orientation.y = 1.0
                new_orientation.z = 0.0
                new_orientation.w = 0.1

                resp = set_field(field.field, 0, new_orientation)
                print (resp)
            except rospy.ServiceException as e:
                print("Service call failed: ", e)

        except rospy.ServiceException as e:
            print("Service call failed: ", e)

    def enable_front_camera(self):
        # this is a serivce call that will enable the camera at the fron
        # of the robot. Once it is enabled, the webots controller will
        # publish the image data in /self.model_name/front_camera/image
        service = self.model_name+'/front_camera/enable'
        rospy.loginfo("Waiting for service %s", service)
        rospy.wait_for_service(service)
        try:
            request_enable = rospy.ServiceProxy(service, set_int)
            answer = request_enable(1) # use 0 for disabling
            print (answer)
        except rospy.ServiceException as e:
                print ("Service call failed: ", e)

if __name__ == "__main__":
    model_name = "krock"
    if len(sys.argv) == 2:
        model_name = sys.argv[1]
    print ("model name: ", model_name)
    rospy.init_node("webots_supervisor_tools")

    tools = ServiceTools(model_name)

    # these two calls are needed to know the node id of the world and the
    # robot
    tools.get_world_node()
    tools.get_robot_node()

    # examples on how to use the webots expose services to query
    tools.get_robot_position()
    tools.get_robot_orientation()

    # and set field values
    tools.set_robot_position()
    tools.set_robot_orientation()

    tools.enable_front_camera()
