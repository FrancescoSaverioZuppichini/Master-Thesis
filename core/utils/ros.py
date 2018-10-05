import rospy

def ros_service(func, service, *args, **kwargs):
    rospy.wait_for_service(service)
    try:
        return func(service, *args, **kwargs)
    except rospy.ServiceException as e:
        print ("Service call failed: ", e)
