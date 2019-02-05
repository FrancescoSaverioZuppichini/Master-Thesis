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
You can import it as a python module
```
from utils.postprocessing import PostProcessingPipeline

p_p_pip = PostProcessingPipeline()
p_p_pip('./bags')

```

Or you can run `main.py` directly. You will need to change  `config.py` with the correct path to the bags file and the outputs folder that you desire.

TODO
- [x] create a whole class to do everything in one shot
- [ ] command line
