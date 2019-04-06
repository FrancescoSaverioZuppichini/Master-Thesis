#ifndef CONTROLLER_HPP
#define CONTROLLER_HPP



#include <fstream>
#include <vector>
#include <string>
#include <iostream>
#include <fstream>
#include <sstream>
#include <math.h>       /* fmod */

#define NUM_LINES   1000
#define NUM_MOTORS   18


using namespace std;

class Controller{


  public:
    //================== public variables ===============================================
    


    //================== public functions ===============================================
    Controller(double frequency, string anglesFile);
    void setTimeStep(double time_step);
    void runStep();
    void getAngles(double *angRef);

  private:
    //================== private variables ===============================================
    
    double anglesData[NUM_LINES][NUM_MOTORS];
    double freq;
    double t, dt;
    int index;
    //================== private functions ===============================================

    
};




int readFileWithLineSkipping(ifstream& inputfile, stringstream& file);


#endif
