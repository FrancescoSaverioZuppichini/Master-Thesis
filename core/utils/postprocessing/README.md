#Post Processing

### Getting Started

Edit `config.py` with the correct path to the bags file and the outputs folder that you desire. 
Then run `main.py`

This package provides the following pipeline utilities

- from the stored bags file creates a csv for each of them 
- from the csvs creates new csvs which the training data for the model

All the files will be stored mimic the `BAG_FOLDER` directories tree. E.g. if a bag file is in

`bags/foo/foo.bag` the csvs will be stored in `csvs/foo/foo.csv` and `dataset/foo/foo.csv`