#from pandas import pd
import math
import numpy as np
import pandas as pd
from utilities import AnalyticalNormalMeasures
from utilities import HSMeasurements
from utilities import WHSMeasurements
from utilities import PrincCompAnalysis
from utilities import plausibilityCheck
import random

random.seed(10)

##0. Exercise. Variance-covariance method for VaR & ES in linear portfolio: a simple example of data mining
#import utilities
df= pd.read_csv('EUROSTOXX50_2023_Dataset.csv', index_col=False)
df.fillna(axis=0,method='ffill',inplace=True)

from DateTime import DateTime
df_3y=df[df['Date']<='2019-03-20']
df_3y=df_3y[df_3y['Date']>='2016-03-18']

#matrix with 4 rows (one for each company), with daily value on the columns
#portfolio with Adidas, Allianz, Munich Re and L’Oréal
# Values=np.array([date_3y['AD.AS'].values,
#                    date_3y['ALVG.DE'].values,
#                    date_3y['MUVGn.DE'].values,
#                    date_3y['OREP.PA'].values])
#Ptf with Adidas, Allianz, Munich Re and L’Oréal
selected_columns0=['AD.AS','ALVG.DE','MUVGn.DE','OREP.PA']
df0=np.array(df_3y[selected_columns0]).T


alpha=0.95
weights=1/4*np.ones([1,4]) #equally weighted
riskMeasureTimeIntervalInDay=1
portfolioValue=10**7
returns0=np.log(df0[:,1:]/df0[:,0:-1])
Var0,ES0=AnalyticalNormalMeasures(alpha, weights, portfolioValue, riskMeasureTimeIntervalInDay, returns0)
print('Var0=',Var0,'ES0=',ES0)


##1. Case study: Historical (HS & WHS) Simulation, Bootstrap and PCA for VaR & ES in a linear portfolio

alpha = 0.99
#Point A)
#matrix with 3 rows (one for each company), with daily value on the columns
# Values1a=np.array([date_3y['TTEF.PA'].values,
#                  date_3y['DANO.PA'].values,
#                  date_3y['SASY.PA'].values,
#                  date_3y['VOWG_p.DE'].values])

selected_columns1a=['TTEF.PA','DANO.PA','SASY.PA','VOWG_p.DE']
df1a=np.array(df_3y[selected_columns1a]).T
returns1a=np.log(df1a[:,1:]/df1a[:,0:-1])

#number of shares for each company
shares = 10**3 * np.array([25, 20, 20, 10])
#prices of the stocks at 20 march 2019: Total,Danone,Sanofi,Volkswagen
prices=np.array([50.36,68.96,78.9858496,143.92])
position_val=shares*prices
portfolioValue=np.sum(position_val)
weights1a=position_val/portfolioValue
Var1a,ES1a=HSMeasurements(returns1a, alpha, weights1a, portfolioValue, riskMeasureTimeIntervalInDay)
print('Var1a=',Var1a,'ES1a=',ES1a)

#Statistical Bootstrap
M=200
n=np.size(returns1a)/4
Rand_simulation=np.random.randint(1,n,M)
returns1s=returns1a[:,Rand_simulation]
VaR1a2,ES1a2=HSMeasurements(returns1s,alpha,weights1a,portfolioValue,riskMeasureTimeIntervalInDay)
print('VaR1a2=',VaR1a2,'ES1a2=',ES1a2)

VaR_check1a = plausibilityCheck(returns1a, weights1a, alpha, portfolioValue, riskMeasureTimeIntervalInDay)
print('VaR_check1a=', VaR_check1a)


#Point B)
#matrix with 5 rows (one for each company), with daily value on the columns
# Values1b = np.array([date_3y['AD.AS'].values,
#                  date_3y['AIR.PA'].values,
#                  date_3y['BBVA.MC'].values,
#                  date_3y['BMWG.DE'].values,
#                  date_3y['SCHN.PA'].values])

selected_columns1b=['AD.AS','AIR.PA','BBVA.MC','BMWG.DE','SCHN.PA']
df1b=np.array(df_3y[selected_columns1b]).T
returns1b = np.log(df1b[:, 1:]/df1b[:, 0:-1])
lambd = 0.97
weights1b = 1/5*np.ones([1, 5])
VaR1b, ES1b = WHSMeasurements(returns1b, alpha, lambd, weights1b, portfolioValue, riskMeasureTimeIntervalInDay)
print('VaR1b=', VaR1b, 'ES1b=', ES1b)

VaR_check1b = plausibilityCheck(returns1b, weights1b, alpha, portfolioValue, riskMeasureTimeIntervalInDay)
print('VaR_check1b=', VaR_check1b)


#Point C)
# Values1c=np.array([date_3y['ABI.BR'].values,
#                  date_3y['AD.AS'].values,
#                  date_3y['ADSGn.DE'].values,
#                  date_3y['AIR.PA'].values,
#                  date_3y['AIRP.PA'].values,
#                  date_3y['ALVG.DE'].values,
#                  date_3y['ASML.AS'].values,
#                  date_3y['AXAF.PA'].values,
#                  date_3y['BASFn.DE'].values,
#                  date_3y['BAYGn.DE'].values,
#                  date_3y['BBVA.MC'].values,
#                  date_3y['BMWG.DE'].values,
#                  date_3y['BNPP.PA'].values,
#                  date_3y['CRH.I'].values,
#                  date_3y['DANO.PA'].values,
#                  date_3y['DB1Gn.DE'].values,
#                  date_3y['DPWGn.DE'].values,
#                  date_3y['DTEGn.DE'].values,
#                  date_3y['ENEI.MI'].values,
#                  date_3y['ENI.MI'].values])

selected_columns1c=['ABI.BR','AD.AS','ADSGn.DE','AIR.PA','AIRP.PA','ALVG.DE','ASML.AS','AXAF.PA','BASFn.DE',
                    'BAYGn.DE','BBVA.MC','BMWG.DE','BNPP.PA','CRH.I','DANO.PA','DB1Gn.DE','DPWGn.DE','DTEGn.DE','ENEI.MI','ENI.MI']
df1c=np.array(df_3y[selected_columns1c]).T
returns1c=np.log(df1c[:,1:]/df1c[:,0:-1])
yearlyMeanReturns=np.zeros([3,20])

yearlyMeanReturns[0,:]=returns1c[:,0:258].mean(1) #mean of the returns of the first year for each company
yearlyMeanReturns[1,:]=returns1c[:,259:513].mean(1) #mean of the returns of the second year for each company
yearlyMeanReturns[2,:]=returns1c[:,514:769].mean(1) #mean of the returns of the third year for each company

#metodo yearlyMeanReturns vettore 20x1
#yearlyMeanReturns=returns.mean(1)

#calcolo portfoliovalue
#prices of the stocks at 20 march 2019: Total,Danone,Sanofi,Volkswagen
#prices=np.array([73.18,23.545,209.8,117.86,93.09919217355464,198.92,166.18,22.675,67.5,63.0,5.347610093999999,71.98,44.19,27.82,68.96,114.5,29.43,15.665,5.544,15.776])
#portfolioValue=200000

#yearlyMeanReturns=yearlyMeanReturns*256

yearlyCovariance=np.zeros([20,20])
yearlyCovariance=np.cov(returns1c) #*256

weights1c= 1/20*np.ones([1, 20])  #equally weighted portfolio
H=10
numberOfPrincipalComponents=6

ES1c, VaR1c = PrincCompAnalysis(yearlyCovariance, yearlyMeanReturns, weights1c, H, alpha,numberOfPrincipalComponents, portfolioValue)
print('VaR1c=', VaR1c, 'ES1c=', ES1c)


#Plausibility Check
VaR_check1c = plausibilityCheck(returns1c, weights1c, alpha, portfolioValue, riskMeasureTimeIntervalInDay)
print('VaR_check1c=', VaR_check1c)