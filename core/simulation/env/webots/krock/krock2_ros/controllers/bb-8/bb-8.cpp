#include <webots/Robot.hpp>
#include <webots/Motor.hpp>
#include <webots/Device.hpp>

#include "ros/ros.h"

#include <std_msgs/Header.h>
#include <geometry_msgs/PoseStamped.h>

// All the webots classes are defined in the "webots" namespace
using namespace std;
using namespace webots;

int main(int argc, char **argv)
{
  // create the Robot instance.
  Robot *robot = new Robot();
  
  ros::init(argc, argv, "bb8");
  ros::NodeHandle node;


  Motor *body_yaw_motor = robot -> getMotor("body yaw motor");
  Motor *body_pitch_motor = robot -> getMotor("body pitch motor");
  Motor *head_yaw_motor = robot -> getMotor("head yaw motor");

  // get the time step of the current world.
  int timeStep = (int) robot->getBasicTimeStep();


   
  body_pitch_motor->setPosition(999999999999999);
  body_yaw_motor->setPosition(999999999999999);
  head_yaw_motor->setPosition(999999999999999);
  
  body_pitch_motor -> setVelocity(0.0);
  body_yaw_motor->setVelocity(0.0);
  head_yaw_motor->setVelocity(0.0);
  double yaw_speed = 0.0;
  double pitch_speed = 0.0;
  const double max_speed = 2.0;
  // Main loop:
  // - perform simulation steps until Webots is stopping the controller
  while (robot->step(timeStep) != -1) {
    body_pitch_motor->setVelocity(max_speed);

  };

  // Enter here exit cleanup code.

  delete robot;
  return 0;
}
