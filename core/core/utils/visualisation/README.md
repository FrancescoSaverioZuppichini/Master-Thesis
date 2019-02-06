# Visualisation

This package contains functions to generate meaningful plots to better understand the data gather from the simulation

## Example

```python
import pandas as pd

from utils.visualisation import show_advancement, show_trajectory

from utils.postprocessing.utils import read_image
from utils.postprocessing.config import Config

df = pd.read_csv('/home/francesco/Desktop/carino/vaevictis/data/dataset/bars1/1548510453.5639887.csv')
hm = read_image('/home/francesco/Desktop/carino/vaevictis/data/maps/bars1.png')

show_advancement(df, hm, Config)

show_trajectory(df.iterrows(), hm)
```

## TODO
- [ ] add pictures to the README