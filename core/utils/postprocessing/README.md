# Post Processing

### Usage
To use the command line, run

```
python3 main.py --help
```

To use in your code you will need to import the handlers, `BagsHandler, DataFrameHandler, PatchesHandler`,
 as well as `PostProcessingConfig`, in your file. 
All these classes inherit from `Handler` and implements the single chain of responsability pattern. It follows a basic example

```python
import glob
from postprocessing import *

config = PostProcessingConfig(base_dir='/foo/data',
                                   maps_folder='/foo/maps',
                                   patch_size=92,
                                   advancement_th=0.12,
                                   skip_every=12,
                                   time_window=125,
                                   name='patches')

patches_h = PatchesHandler(config=config)
df_h = DataFrameHandler(successor=patches_h, config=config)
b_h = BagsHandler(config=config, successor=df_h)
# the bags handler needs a list of bags file path 
bags = glob.glob('{}/**/*.bag'.format(config.bags_dir))

list(b_h(bags))
```

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

TODO
- [x] command line
