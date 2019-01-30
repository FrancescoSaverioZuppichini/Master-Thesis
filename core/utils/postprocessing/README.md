# Post Processing

### Getting Started

We used [pypeln](https://github.com/cgarciae/pypeln) a 
*python library for creating concurrent data pipelines* to create the pipeline. 

The pipeline does the follow

- open all the `bags` files and convert them to `pandas` `Dataframe` and set the index on the timestamp
- decorate the dataframes by:
    - converting the timestamp to the time relative for the simulation (e.g. first column timestamp -> 0.0s, last column = 20.0)
    - converting the pose's orientation quaternion to euler 
    - calculate the advancement
    - label each row
- from the dataframes it creates a path and store it in `<OUTPUT_DIR>/<LABEL>/<DATAFRAME_TIME_STAMP>-<CURRENT_TIME>.png`


### Usage
Edit `config.py` with the correct path to the bags file and the outputs folder that you desire. 
Then run `main.py` to create the paths from the bag files.

TODO
- [ ] command line
