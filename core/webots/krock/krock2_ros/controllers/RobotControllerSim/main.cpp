#include <webots/Supervisor.hpp>
#include <iostream>
#include <ctime>
#include <fstream>
#include <sys/time.h>
#include <vector>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string.h>
#include <stdlib.h>
#include "controller.hpp"
#include "robotSim.hpp"

#define TIME_STEP   10    //ms
#define FREQUENCY   0.5   //Hz
#define NUM_MOTORS   18

using namespace std;
using namespace webots;


//############################ MAIN PROGRAM ###########################################################

int main(int argc, char **argv)
{
    cout<<"MAIN STARTS"<<endl;
    double dt = TIME_STEP/1000., t=0;
    double table_p[30], table_t[30];

    //======================================================//
    RobotSim robotSim(TIME_STEP);
    cout<<"ROBOTSIM CREATED"<<endl;

    
    Controller controller(FREQUENCY, "./gaits/angles_normal.txt");
    cout<<"CONTROLLER CREATED"<<endl;

    
    double angref[NUM_MOTORS], feedbackAngles[NUM_MOTORS], feedbackTorques[NUM_MOTORS], rollPitchYaw[3], gpsDataFront[3], gpsDataHind[3];
    ofstream gpsLog("./logData/gpsLog.txt");
    ofstream rpyLog("./logData/rpyLog.txt");
    ofstream anglesLog("./logData/anglesLog.txt");
    ofstream torquesLog("./logData/torquesLog.txt");
//=============================  LOOP  =============================================
    while(robotSim.step(TIME_STEP) != -1) {


        // controller calls
        controller.setTimeStep(dt);
        controller.runStep();
        controller.getAngles(angref);

        // send new joint angles to the robot
        robotSim.setAngles(angref);

        // get feedback from the robot
        robotSim.GetIMU(rollPitchYaw);
        robotSim.GetPosition(gpsDataFront, gpsDataHind);
        robotSim.getPositionTorques(feedbackAngles, feedbackTorques);


        // LOG DATA
        gpsLog << t << "\t";
        for(int i=0; i<3; i++){
            gpsLog << gpsDataFront[i] << "\t";
        }
        gpsLog << endl;

        rpyLog << t << "\t";
        for(int i=0; i<3; i++){
            rpyLog << rollPitchYaw[i] << "\t";
        }
        rpyLog << endl;

        anglesLog << t << "\t";
        for(int i=0; i<NUM_MOTORS; i++){
            anglesLog << feedbackAngles[i] << "\t";
        }
        anglesLog << endl;

        torquesLog << t << "\t";
        for(int i=0; i<NUM_MOTORS; i++){
            torquesLog << feedbackTorques[i] << "\t";
        }
        torquesLog << endl;

        
        t+=dt;
    }


  return 0;
}
















