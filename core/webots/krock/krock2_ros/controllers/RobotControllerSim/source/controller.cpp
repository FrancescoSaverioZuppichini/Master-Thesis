#include "controller.hpp"



// ---------------------------------------------- FUNCTIONS ----------------------------------------------------------
/* Controller constructor */
Controller :: Controller(double frequency, string anglesFile)
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
Controller :: setTimeStep(double time_step)
{
    dt=time_step;
    return;
}

/* Updates angles for all joints - MAIN FUNCTION */
void
Controller :: runStep()
{
    t=t+dt;

    index = floor(fmod(t,1/freq)*freq*NUM_LINES);

}



/* Writes angles calculated by runStep function into a table - interface with Pleurobot class */
void
Controller :: getAngles(double *angRef)
{   

    for(int i=0; i<NUM_MOTORS; i++){
        angRef[i] = anglesData[index][i]; 
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
    return linenum-1;
}