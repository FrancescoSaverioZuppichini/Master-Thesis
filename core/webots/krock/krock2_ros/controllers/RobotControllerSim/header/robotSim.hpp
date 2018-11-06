#ifndef ROBOTSIM_HPP
#define ROBOTSIM_HPP

#include <iostream>

#include <vector>
#include <webots/Supervisor.hpp>
#include <webots/Accelerometer.hpp>
#include <webots/Compass.hpp>
#include <webots/GPS.hpp>
#include <webots/Motor.hpp>
#include <webots/TouchSensor.hpp>
#include <webots/Node.hpp>
#include <webots/Field.hpp>
#include <webots/PositionSensor.hpp>
#include <webots/InertialUnit.hpp>


#define N_TOUCH_SENSORS    4
#define NUM_MOTORS   18

using namespace std;


class RobotSim : public webots::Supervisor{


  public:
    //================== public variables ===============================================
    //webots::Servo *servo[NUM_MOTORS], *linservo[3];
    vector<webots::Motor*> rm_motor;
    vector<webots::PositionSensor*> ps_motor;
    webots::InertialUnit *imu;
    webots::Accelerometer *acc;
    webots::Compass *compass;
    webots::GPS *gps_fgird, *gps_hgird;
    webots::Camera *camera;
    webots::TouchSensor *touch_sensor[N_TOUCH_SENSORS];
    webots::TouchSensor *touch_sensor_spine[12];
    webots::Node *fgirdle, *FL_marker, *FR_marker, *HL_marker, *HR_marker, *roboDef, *tsdefFL, *tsdefFR, *tsdefHL, *tsdefHR;
    const double *compassData, *gpsDataFgird, *gpsDataHgird, *imuData, *accData, *ts_fl, *ts_fr, *ts_hl, *ts_hr, *rotMat, *posFL, *posFR, *posHL, *posHR;
    webots::Field *roboRot, *roboPos;
    double t_total;

    webots::Node *supportPolyDEF;
    
    //================== public functions ===============================================
    RobotSim(int TIME_STEP); // constructor
    void setAngles(double*);
    void torques(double*, int*);
    void getPositionTorques(double *d_posture, double *d_torques);
    void GetPosition(double *gpsData1, double *gpsData2);
    void GetIMU(double *imuData_i);
    void ReadTouchSensors(double *ts_data);
    void killSimulation();
    void setPositionRotation(double *p, double *r);
    void setPositionOfRobot(double *p);
    void GetCompass(double *data_i);
    void GetFeetGPS(double *FL_feet_gpos, double *FR_feet_gpos, double *HL_feet_gpos, double *HR_feet_gpos);
    void setServoMaxForce(double *force);
  private:
    //================== private variables ===============================================




};


#endif