import pandas as pd


df = pd.DataFrame(data={'foo' : [1]})

df.to_csv('/home/francesco/Desktop/test.csv')

df = pd.read_csv('/home/francesco/Desktop/test.csv', index_col=[0])
df = pd.concat([df, pd.DataFrame(data={'foo' : [1]})], sort=True)
df = df.reset_index(drop=True)

df.to_csv('/home/francesco/Desktop/test.csv')
