import quandl
import pandas as pd

df = quandl.get('WIKI/GOOGL')
df.to_csv('wiki-googl.csv', sep=',', index=True)