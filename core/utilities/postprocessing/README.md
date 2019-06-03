# Post Processing

### Usage


Example

Assuming you have this directory structure

```
├── test
│   ├── bags


Running the following script

```python3
post = PostProcessing(root='./test/',
                      maps_dir='<PATH_TO_HEIGHTMAP',
                      advancement=1, 
                      time_window=150)

post()
```

Will result in

```
├── test
│   ├── bags
│   ├── csvs
|   |── 150 
│       ├── csvs
│       └── patches
```
Where `csvs` is the folder with the parsed bag files that are cached to speed up the next calls. `150` is the folder with the output for the given time window. It contains the dataframes that with all the information for each patch and the actually patches. The dataframes contain the `height`, the `advancement`, the coordinates and the filename for each patch.


We used [pypeln](https://github.com/cgarciae/pypeln) a 
*python library for creating concurrent data pipelines* to create the pipeline. 

The pipeline does the follow

- opens all the `bags` files and convert them to `pandas` `Dataframe` and set the index on the timestamp
- decorates the dataframes by:
    - converting the timestamp to the time relative for the simulation (e.g. first column timestamp -> 0.0s, last column = 20.0)
    - converting the pose's orientation quaternion to euler 
    - calculate the advancement
    - clear the data
- creates a dataframe with the `advancement` and the extracted `patch`

TODO
- [ ] command line
