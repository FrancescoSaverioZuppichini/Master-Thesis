#include "GaitControl.hpp"


// ---------------------------------------------- FUNCTIONS ----------------------------------------------------------
/* GaitControl constructor */
GaitControl :: GaitControl(double frequency, string anglesFile)
{
    // read angles file
    stringstream stringstream_file;
    ifstream file_angles;
    file_angles.open(anglesFile);
    if(file_angles.is_open()){
        cout << "reading angles" << endl;
        stringstream_file.str(string());
        readFileWithLineSkipping(file_angles, stringstream_file);
        for(int i=0; i<NUM_LINES; i++){
            for(int j=0; j<NUM_MOTORS; j++){
                stringstream_file >> anglesData[i][j];
            }
        }

    }

    freq = frequency;
    t=0;
}

/* Sets time step */
void
GaitControl :: setTimeStep(double time_step)
{
    dt=time_step;
    return;
}

/* Updates angles for all joints - MAIN FUNCTION */
void
GaitControl :: runStep()
{
    t=t+dt;

    index = floor(fmod(t,1/freq)*freq*NUM_LINES);

}

/* Writes angles calculated by runStep function into a table - interface with Pleurobot class */
void
GaitControl :: getAngles(double *angRef)
{

    for(int i=0; i<NUM_MOTORS; i++){
        angRef[i] = anglesData[index][i];
    }
}

/* Update angles for left/right joints using frequency values
   (intended for "manual" turning behaviour).
   NOTE: remember to cap or discretize freq posible values to avoid
   unexpected behaviours*/

void
GaitControl :: runStep(const double freq_left, const double freq_right)
{
    t=t+dt;

    if (freq_left>0)
        index_left = floor(fmod(t,1/freq_left)*freq_left*NUM_LINES);
    else if (freq_left<0)
        index_left = NUM_LINES - floor(fmod(t,1/-freq_left)*-freq_left*NUM_LINES);;

    if (freq_right>0)
        index_right = floor(fmod(t,1/freq_right)*freq_right*NUM_LINES);
    else if (freq_right<0)
        index_right = NUM_LINES - floor(fmod(t,1/-freq_right)*-freq_right*NUM_LINES);;

    // tail joint angles, how to choose them?

    index_tail = (index_left < index_right)? index_left: index_right;

    //cout << "L: ("<< freq_left <<")= " << index_left << " R("<< freq_right << ")= " << index_right << endl;

}

/* Get angles calculated by runStep manual function into a table */
void
GaitControl :: getAnglesManual(double *angRef)
{
    // Assuming that first 2 set of values (each set 4 entries) from the angles
    // file correspond to the front legs, following 2 set to rear legs and
    // last 2 entries to the tail.
    const int LEGS=4;
    const int JOINTSPLEG=4;
    for (int leg =0; leg < LEGS; leg++){
        for (int j =0; j< JOINTSPLEG; j++){
            if (leg%2 == 0)
                angRef[(leg*JOINTSPLEG)+j] = anglesData[index_left][(leg*JOINTSPLEG)+j];
            else
                angRef[(leg*JOINTSPLEG)+j] = anglesData[index_right][(leg*JOINTSPLEG)+j];
        }
    }
    // tail
    for(int i=(LEGS*JOINTSPLEG)-1; i<NUM_MOTORS; i++){
        angRef[i] = anglesData[index_tail][i];
    }
}












int
readFileWithLineSkipping(ifstream& inputfile, stringstream& file){
    string line;
    cout << "ckecking input file" << endl;
    //file.str(std::string());
    int linenum=0;
    cout << "starting the loop" << endl;
    cout << inputfile.is_open() << endl;

    while (!inputfile.eof()){
        getline(inputfile,line);

        //line.erase(line.begin(), find_if(line.begin(), line.end(), not1(ptr_fun<int, int>(isspace))));
        if (line.length() == 0 || (line[0] == '#') || (line[0] == ';')){
            //continue;
        }
        else
            file << line << "\n";
            linenum++;
        }
    cout << linenum << endl;
    return linenum-1;
}
