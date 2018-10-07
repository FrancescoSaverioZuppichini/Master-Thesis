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
from utils.ros import ros_service
# These service are need for calling and receiving services
# If a new service type is needed, verify the API reference and add it
# here. I have included such files in the srv folder of this package
# If the srv is not found, verify that it appears in the service
# generation segment in the CMakeLists.txt of this package
from webots_ros.srv import get_bool, get_int, get_uint64, set_int, node_get_field, field_get_node, field_get_vec3f, \
    node_get_position, node_get_orientation, field_set_vec3f, field_set_rotation, set_string


class Supervisor:
    """
    Interface to access all the services exposed by the Webots Supervisor class.
    This class can be inherited to automatically access all the methods,
    or it can be instantiated to be used and shared/
    """
    name = None

    @staticmethod
    def get_service(service, type):
        res = None
        try:
            res = rospy.ServiceProxy(service, type)
        except rospy.ServiceException as e:
            rospy.logerr(e)
        finally:
            return res

    def get_world_node(self):
        service = self.name + '/supervisor/get_root'
        world = self.get_service(service, get_uint64)
        return world.get_world()

    def load_world(self, world_name):
        service = self.name + '/supervisor/world_load'
        set_world = self.get_service(service, set_string)
        res = set_world(world_name)
        return res

    def reload_world(self):
        service = self.name + '/supervisor/world_reload'
        res = self.get_service(service, get_uint64)
        return res

    def reset_simulation(self):
        service = self.name + '/supervisor/simulation_reset'
        res = self.get_service(service, get_uint64)
        return res

    def reset_simulation_physics(self):
        service = self.name + '/supervisor/simulation_reset_physics'
        res = self.get_service(service, get_uint64)
        return res

    def get_robot_node(self):
        service = self.name + '/supervisor/get_self'
        res = self.get_service(service, get_uint64)
        self.robot_node = res()
        return self.robot_node

    def get_robot_position(self):
        service = self.name + '/supervisor/node/get_position'
        req_pos = self.get_service(service, node_get_position)
        pos = req_pos(self.robot_node.value)
        return pos

    def get_robot_orientation(self):
        service = self.name + '/supervisor/node/get_orientation'
        req_orient = self.get_service(service, node_get_orientation)
        orient = req_orient(self.robot_node.value)
        return orient

    def set_robot_position(self, x, y, z):
        service = self.name + '/supervisor/node/get_field'

        req_field = self.get_service(service, node_get_field)
        field = req_field(self.robot_node.value, 'translation')

        service = self.name + '/supervisor/field/set_vec3f'
        set_pos = self.get_service(service, field_set_vec3f)
        pos_res = set_pos(field.field, 0, Vector3(x, y, z))

        return pos_res

    def set_robot_orientation(self, x, y, z, w):
        service = self.name + '/supervisor/node/get_field'

        req_field = self.get_service(service, node_get_field)
        field = req_field(self.robot_node.value, 'rotation')

        service = self.name + '/supervisor/field/set_rotation'
        set_rot = self.get_service(service, field_set_rotation)
        rot_res = set_rot(field.field, 0, Quaternion(x, y, x, w))

        return rot_res
