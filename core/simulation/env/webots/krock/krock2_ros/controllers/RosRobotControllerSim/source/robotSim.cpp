#include "robotSim.hpp"

extern double get_timestamp();

/* RobotSim constructor */
RobotSim :: RobotSim(int TIME_STEP)
{
    cout << "RobotSim constructor"<<endl;

    gps_fgird=new webots::GPS("gps_fgirdle");
    gps_fgird->enable(TIME_STEP);
    gps_hgird=new webots::GPS("gps_hgirdle");
    gps_hgird->enable(TIME_STEP);

    imu = new webots::InertialUnit("IMU");
    imu->enable(TIME_STEP);

    gpsDataFgird=gps_fgird->getValues();
    gpsDataHgird=gps_hgird->getValues();
    imuData = imu->getRollPitchYaw();

    //
    // SUPERVISOR_NEEDED?
    //
    // get the robot node to get the translantion and rotation
    // is it possible to do it without the superviosr capabilities?
    roboDef=getFromDef("ROBOT");

    roboPos=roboDef->getField("translation");
    roboRot=roboDef->getField("rotation");


    vector<string> RM_NAMES =
    {
        "FLpitch_motor", "FLyaw_motor", "FLroll_motor", "FLknee_motor",
        "FRpitch_motor", "FRyaw_motor", "FRroll_motor", "FRknee_motor",
        "HLpitch_motor", "HLyaw_motor", "HLroll_motor", "HLknee_motor",
        "HRpitch_motor", "HRyaw_motor", "HRroll_motor", "HRknee_motor",
        "spine1_motor", "spine2_motor",
        "neck1_motor", "tail1_motor", "tail2_motor"
    };

    vector<string> PS_NAMES =
    {
        "FLpitch_sensor", "FLyaw_sensor", "FLroll_sensor", "FLknee_sensor",
        "FRpitch_sensor", "FRyaw_sensor", "FRroll_sensor", "FRknee_sensor",
        "HLpitch_sensor", "HLyaw_sensor", "HLroll_sensor", "HLknee_sensor",
        "HRpitch_sensor", "HRyaw_sensor", "HRroll_sensor", "HRknee_sensor",
        "spine1_sensor", "spine2_sensor",
        "neck1_sensor", "tail1_sensor", "tail2_sensor"
    };

    const char *TOUCH_SENSOR_NAMES[N_TOUCH_SENSORS] =
    {
    "fl_touch", "fr_touch", "hl_touch", "hr_touch",
    };

    rm_motor.resize(NUM_MOTORS);
    ps_motor.resize(NUM_MOTORS);
    // get the servos
    cout << "connecting to motors" << endl;
    for(int i=0;i<NUM_MOTORS;i++)
    {
        //rm_motor[i].getMotor(RM_NAMES[i]);
        rm_motor[i] = webots::Robot::getMotor(RM_NAMES[i]);
        ps_motor[i] = webots::Robot::getPositionSensor(PS_NAMES[i]);
        ps_motor[i]->enable(TIME_STEP);
        rm_motor[i]->enableTorqueFeedback(TIME_STEP);
    }
    cout << "motors collected" << endl;

    for(int i=0;i<N_TOUCH_SENSORS;i++){
        touch_sensor[i] = new webots::TouchSensor(TOUCH_SENSOR_NAMES[i]);
        touch_sensor[i]->enable(TIME_STEP);
    }

    tsdefFL=getFromDef("TS_FL");
    tsdefFR=getFromDef("TS_FR");
    tsdefHL=getFromDef("TS_HL");
    tsdefHR=getFromDef("TS_HR");

    // Enable Camera
    cout << "setting camera" << endl;
    camera = new webots::Camera("front_camera");
    camera->enable(TIME_STEP);

}

/* Sets the position of all servos to a table of angles in radians */
void
RobotSim :: setAngles(double *table)
{
    for(int i=0; i<NUM_MOTORS;  i++){
        rm_motor[i]->setPosition(table[i]);
    }
}

/* Sets the torques of all servos to a table of torques */
void
RobotSim :: torques(double *table, int *ids)
{
        for(int i=0; i<NUM_MOTORS;  i++){
            if(ids[i]){
                //servo[i]->setForce(table[i]);
            }
        }
}

/* Reads positions, torques   */
void
RobotSim::getPositionTorques(double *d_posture, double *d_torques)
{
    for(int i=0;i<NUM_MOTORS;i++)
    {
        d_posture[i]=ps_motor[i]->getValue();
        d_torques[i]=rm_motor[i]->getForceFeedback();
    }
}

/* Reads positions, torques and IMU data */
void
RobotSim::GetPosition(double *gpsDataFgird_i, double *gpsDataHgird_i)
{
    gpsDataFgird_i[0]=gpsDataFgird[0];
    gpsDataFgird_i[1]=gpsDataFgird[1];
    gpsDataFgird_i[2]=gpsDataFgird[2];

    gpsDataHgird_i[0]=gpsDataHgird[0];
    gpsDataHgird_i[1]=gpsDataHgird[1];
    gpsDataHgird_i[2]=gpsDataHgird[2];
}

/* Reads IMU, torques and IMU data */
void
RobotSim::GetIMU(double *imuData_i)
{
    imuData_i[0]=imuData[0];
    imuData_i[1]=imuData[1];
    imuData_i[2]=imuData[2];

}


/* Reads touch sensor data */
void
RobotSim::ReadTouchSensors(double *ts_data)
{

    ts_fl=touch_sensor[0]->getValues();
    ts_fr=touch_sensor[1]->getValues();
    ts_hl=touch_sensor[2]->getValues();
    ts_hr=touch_sensor[3]->getValues();
    /*for(int i=0; i<3; i++){
        ts_data[i]=ts_fl[i];
        ts_data[i+3]=ts_fr[i];
        ts_data[i+6]=ts_hl[i];
        ts_data[i+9]=ts_hr[i];
    }*/

    ts_data[0]=ts_fl[0];
    ts_data[1]=-ts_fl[2];
    ts_data[2]=ts_fl[1];

    ts_data[0+3]=ts_fr[0];
    ts_data[1+3]=-ts_fr[2];
    ts_data[2+3]=ts_fr[1];

    ts_data[0+6]=ts_hl[0];
    ts_data[1+6]=-ts_hl[2];
    ts_data[2+6]=ts_hl[1];

    ts_data[0+9]=ts_hr[0];
    ts_data[1+9]=-ts_hr[2];
    ts_data[2+9]=ts_hr[1];
}

/* Get camera image*/
// Theses functions are not needed because webot ROS controller exposes
// the image as a topic

/*
int RobotSim::getCameraWidth(){
  return camera->getWidth();
}

int RobotSim::getCameraHeight(){
  return camera->getHeight();
}

void RobotSim::getCameraImage(unsigned char *image){
  const unsigned char *tmp;
  tmp = camera->getImage();
  memcpy(image, tmp, (camera->getWidth())*(camera->getHeight())*4);
}
*/

/* Quits simulation */
void
RobotSim::killSimulation()
{
    simulationQuit(0);
}



//
// SUPERVISOR_NEEDED?
//
// get the robot node to set the translantion and rotation
// is it possible to do it without the superviosr capabilities?


/* Sets position ond orientation of the Robot */
void
RobotSim::setPositionRotation(double *p, double *r)
{
    roboPos->setSFVec3f((const double*)p);
    roboRot->setSFRotation(r);
}

/* Sets position ond orientation of the Robot */
void
RobotSim::setPositionOfRobot(double *p)
{
    roboPos->setSFVec3f((const double*)p);
}



/* Reads compass data */
void
RobotSim::GetCompass(double *data)
{
    data[0]=compassData[0];
    data[1]=compassData[1];
    data[2]=compassData[2];
}



/* Reads compass data */
void
RobotSim::GetFeetGPS(double *FL_feet_gpos, double *FR_feet_gpos, double *HL_feet_gpos, double *HR_feet_gpos)
{
    static const double *FL_feet_gposC=tsdefFL->getPosition();
    static const double *FR_feet_gposC=tsdefFR->getPosition();
    static const double *HL_feet_gposC=tsdefHL->getPosition();
    static const double *HR_feet_gposC=tsdefHR->getPosition();
   // cout<<FL_feet_gpos[0]<<FL_feet_gpos[1]<<FL_feet_gpos[2]<<endl;

    for(int i=0; i<3; i++){
        FL_feet_gpos[i]=FL_feet_gposC[i];
        FR_feet_gpos[i]=FR_feet_gposC[i];
        HL_feet_gpos[i]=HL_feet_gposC[i];
        HR_feet_gpos[i]=HR_feet_gposC[i];
    }
}
