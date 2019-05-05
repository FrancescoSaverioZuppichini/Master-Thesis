
# Pipeline
This packages provide a set of tiny yet useful function to create functional pipelines.
## Example


```python
import pandas as pd
import numpy as np

from pipeline import Compose

df = pd.DataFrame(np.random.randint(0,100,size=(100, 4)), columns=list('ABCD'))

def merge_df(df):
    df['A'] = df['A'] + df['B'] 
    return df

class OnlyHead():
    def __init__(self, n):
        self.n = n
    def __call__(self, df):
        return df.head(self.n)

pip = Compose([
    lambda x: x[x > 10],
    lambda x: x.drop(columns=['C', 'D']),
    merge_df,
    OnlyHead(5)
])

pip(df)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>A</th>
      <th>B</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>115.0</td>
      <td>32.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>165.0</td>
      <td>67.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>138.0</td>
      <td>85.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>4</th>
      <td>111.0</td>
      <td>90.0</td>
    </tr>
  </tbody>
</table>
</div>



## TODO
- [ ] getting started section
    - [ ] For each
    - [ ] multi thread
    - [ ] Combine
    - [ ] Merge
