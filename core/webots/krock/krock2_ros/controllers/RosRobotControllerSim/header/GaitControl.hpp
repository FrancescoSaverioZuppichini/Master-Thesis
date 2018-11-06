#ifndef CONTROLLER_HPP
#define CONTROLLER_HPP



#include <fstream>
#include <vector>
#include <string>
#include <iostream>
#include <fstream>
#include <sstream>
#include <math.h>       /* fmod */


#include <webots_ros/get_float.h>
#include <webots_ros/get_int.h>
#include <webots_ros/set_bool.h>
#include <webots_ros/set_float.h>
#include <webots_ros/set_int.h>

#include "Ros.hpp"

#define NUM_LINES   1000
#define NUM_MOTORS   18


using namespace std;

class GaitControl{


  public:
    //================== public variables ===============================================



    //================== public functions ===============================================
    GaitControl(double frequency, string anglesFile);
    void setTimeStep(double time_step);
    void runStep();
    void getAngles(double *angRef);
    // for "manual" control
    void runStep(const double freq_left, const double freq_right);
    void getAnglesManual(double *angRef);

  private:
    //================== private variables ===============================================

    double anglesData[NUM_LINES][NUM_MOTORS];
    double freq;
    double t, dt;
    int index;
    // for "manual" control
    int index_left,
        index_right,
        index_tail;
    //================== private functions ===============================================


};




int readFileWithLineSkipping(ifstream& inputfile, stringstream& file);


#endif
