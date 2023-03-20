#from pandas import pd
import pandas as pd
df= pd.read_csv('EUROSTOXX50_2023_Dataset.csv') #, index_cd=0)
df.fillna(axis=0,method='ffill',inplace=True)
#print(df)
print(df['Date'])
print(df['AD.AS'])

from DateTime import DateTime
y = DateTime('Mar 7, 2016')
print(y)
df['AD.AS'].index(29.735000)
cioa
#from pandas import pd
import pandas as pd
df= pd.read_csv('EUROSTOXX50_2023_Dataset.csv') #, index_cd=0)
df.fillna(axis=0,method='ffill',inplace=True)
#print(df)
print(df['Date'])
print(df['AD.AS'])

from DateTime import DateTime
y = DateTime('Mar 7, 2013')
print(y)
df['AD.AS'].index(29.735000)
