#from pandas import pd
import pandas as pd
df= pd.read_csv('EUROSTOXX50_2023_Dataset.csv') #, index_cd=0)
df.fillna(axis=0,method='ffill',inplace=True)
#print(df)
print(df['Date'])
print(df['AD.AS'])

from DateTime import DateTime
y = DateTime('2013-01-03')
#x= DateTime(df['Date'])
#print(x)
a=df['Date']
p=a[1]
#print(a[1])
#ind=df['Date'].index(p)
#print(ind)
df.drop(index=p, inplace=True)




