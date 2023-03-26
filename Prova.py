import numpy as np
import pandas as pd
from utilities_prova import AnalyticalNormalMeasures
from utilities import HSMeasurements
from utilities import WHSMeasurements
from utilities import PrincCompAnalysis
from utilities import plausibilityCheck

#0
df= pd.read_csv('EUROSTOXX50_2023_Dataset.csv')
df.fillna(axis=0,method='ffill',inplace=True) #filling the nan values with the previous value

#df_3y=df[df['Date']>='2016-03-18' and df['Date']<='2019-03-20']

#3 years estimation
# df_3y=df[df['Date']<='2019-03-20']
# df_3y=df_3y[df_3y['Date']>='2016-03-18']
#print(df_3y)

#Ptf with Adidas, Allianz, Munich Re and L’Oréal
selected_columns=['AD.AS','ALVG.DE','MUVGn.DE','OREP.PA']
df0=df[selected_columns]

df0=df0[df0['Date']>='2016-03-18']
df0=df0[df0['Date']<='2019-03-20']
# df0=np.array([df_3y['AD.AS'].values,
#                  df_3y['ALVG.DE'].values,
#                  df_3y['MUVGn.DE'].values,
#                  df_3y['OREP.PA'].values]).T
# df1=np.array([df_3y['AD.AS'],df_3y['ALVG.DE'],df_3y['MUVGn.DE'],df_3y['OREP.PA']])
# print(df1)
portfolioValue0=10**7
alpha0=0.95
weights0=1/4*np.ones([4,1]) #equally weighted ptf
riskMeasureTimeIntervalInDay=1
#df0 is a 770x4 matrix
returns0=np.log(df0[1:,:]/df0[:-1,:]) # df0[1:,:] selects all rows except the first row
                                      # df0[:-1,:] selects all rows except the last row
#returns0 will be a 769x4 matrix
Var0,ES0=AnalyticalNormalMeasures(alpha0, weights0, portfolioValue0, riskMeasureTimeIntervalInDay, returns0)
print('Var0=',Var0,'ES0=',ES0)

##1. Case study:
alpha=0.99
#ptf A
