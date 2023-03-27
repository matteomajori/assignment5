import numpy as np
import pandas as pd
from utilities_prova import AnalyticalNormalMeasures
from utilities_prova import HSMeasurements
from utilities_prova import WHSMeasurements
from utilities import PrincCompAnalysis
from utilities_prova import plausibilityCheck

#0
df= pd.read_csv('EUROSTOXX50_2023_Dataset.csv')
df.fillna(axis=0,method='ffill',inplace=True) #filling the nan values with the previous value

#df_3y=df[df['Date']>='2016-03-18' and df['Date']<='2019-03-20']

#3 years estimation
df_3y=df[df['Date']<='2019-03-20']
df_3y=df_3y[df_3y['Date']>='2016-03-18']
#print(df_3y)
df_3y = df_3y.set_index('Date')
#Ptf with Adidas, Allianz, Munich Re and L’Oréal
selected_columns0=['AD.AS','ALVG.DE','MUVGn.DE','OREP.PA']
df0=np.array(df_3y[selected_columns0])

# df0=df0[df0['Date']>='2016-03-18']
# df0=df0[df0['Date']<='2019-03-20']
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
# Portfolio with: Total (25K shares), Danone (20K Shares), Sanofi (20K Shares), Volkswagen (10K Shares)
selected_columns1a=['TTEF.PA','DANO.PA','SASY.PA','VOWG_p.DE']
df1a=np.array(df_3y[selected_columns1a])
returns1a=np.log(df1a[1:,:]/df1a[:-1,:]) #769x4
#number of shares for each company
shares = 10**3 * np.array([25, 20, 20, 10])
#prices of the stocks at 20 march 2019
#prices=np.array([50.36,68.96,78.9858496,143.92])
prices=np.array(df_3y[selected_columns1a].loc['2019-03-20'])
#print(prices)

position_val=shares*prices #4x1
portfolioValue=np.sum(position_val)
weights1a=position_val/portfolioValue #4x1
Var1a,ES1a=HSMeasurements(returns1a, alpha, weights1a, portfolioValue, riskMeasureTimeIntervalInDay)
print('Var1a=',Var1a,'ES1a=',ES1a)

#Statistical Bootstrap
M=200
#n=np.size(returns)/4
# Rand_simulation=np.random.randint(1,n-1,M)
Rand_simulation = np.random.randint(0, len(returns1a), size=M)
returns1a2=returns1a[Rand_simulation,:] #we select the corresponding set of returns
VaR1a2,ES1a2=HSMeasurements(returns1a2,alpha,weights1a,portfolioValue,riskMeasureTimeIntervalInDay)
print('VaR1a2=',VaR1a2,'ES1a2=',ES1a2)

VaR_check1a = plausibilityCheck(returns1a, weights1a, alpha, portfolioValue, riskMeasureTimeIntervalInDay)
print('VaR_check1a=', VaR_check1a)

#ptf B
#Portfolio with: Adidas, Airbus, BBVA, BMW and Schneider
selected_columns1b=['AD.AS','AIR.PA','BBVA.MC','BMWG.DE','SCHN.PA']
df1b=np.array(df_3y[selected_columns1b])
returns1b=np.log(df1b[1:,:]/df1b[:-1,:]) #769x5

lambd = 0.97
weights1b = 1/5*np.ones([5, 1]) #equally weighted ptf
VaR1b, ES1b = WHSMeasurements(returns1b, alpha, lambd, weights1b, portfolioValue, riskMeasureTimeIntervalInDay)
print('VaR1b=', VaR1b, 'ES1b=', ES1b)

VaR_check1b = plausibilityCheck(returns1b, weights1b, alpha, portfolioValue, riskMeasureTimeIntervalInDay)
print('VaR_check1b=', VaR_check1b)

## 2
