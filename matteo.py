#from pandas import pd
import math
import numpy as np
import pandas as pd
from utilities import AnalyticalNormalMeasures
from utilities import HSMeasurements
from utilities import WHSMeasurements
import random

random.seed(10)

##0. Exercise. Variance-covariance method for VaR & ES in linear portfolio: a simple example of data mining
#import utilities
df= pd.read_csv('EUROSTOXX50_2023_Dataset.csv', index_col=False)
df.fillna(axis=0,method='ffill',inplace=True)

from DateTime import DateTime
date3y=df['Date'][823:1593].values

#matrix with 4 rows (one for each company), with daily value on the columns
Values=np.array([df['AD.AS'][823:1593].values,
                 df['ALVG.DE'][823:1593].values,
                 df['MUVGn.DE'][823:1593].values,
                 df['OREP.PA'][823:1593].values])


alpha=0.95
weights=1/4*np.ones([1,4])
riskMeasureTimeIntervalInDay=1
portfolioValue=10000000
returns=np.log(Values[:,2:]/Values[:,1:-1])
Var0,ES0=AnalyticalNormalMeasures(alpha, weights, portfolioValue, riskMeasureTimeIntervalInDay, returns)
print('Var0=',Var0,'ES0=',ES0)


##1. Case study: Historical (HS & WHS) Simulation, Bootstrap and PCA for VaR & ES in a linear portfolio

alpha = 0.99
#Point A)
#matrix with 3 rows (one for each company), with daily value on the columns
Values=np.array([df['TTEF.PA'][823:1593].values,
                 df['DANO.PA'][823:1593].values,
                 df['SASY.PA'][823:1593].values,
                 df['VOWG_p.DE'][823:1593].values])
returns=np.log(Values[:,2:]/Values[:,1:-1])
#number of shares for each stock
shares = 10**3 * np.array([25, 20, 20, 10])
#prices of the stocks at 20 march 2019: Total,Danone,Sanofi,Volkswagen
prices=np.array([50.36,68.96,78.9858496,143.92])
position_val=shares*prices
portfolioValue=np.sum(position_val)
weights1a=position_val/portfolioValue
Var1a,ES1a=HSMeasurements(returns, alpha, weights1a, portfolioValue, riskMeasureTimeIntervalInDay)
print('Var1a=',Var1a,'ES1a=',ES1a)

#Statistical Bootstrap
M=200
n=np.size(returns)/4
Rand_simulation=np.random.randint(1,n-1,M)
returns1b=returns[:,Rand_simulation]
VaR1a2,ES1a2=HSMeasurements(returns1b,alpha,weights1a,portfolioValue,riskMeasureTimeIntervalInDay)
print('VaR1a2=',VaR1a2,'ES1a2=',ES1a2)




#Point B)
#matrix with 5 rows (one for each company), with daily value on the columns
Values = np.array([df['AD.AS'][823:1593].values,
                 df['AIR.PA'][823:1593].values,
                 df['BBVA.MC'][823:1593].values,
                 df['BMWG.DE'][823:1593].values,
                 df['SCHN.PA'][823:1593].values])
returns = np.log(Values[:, 2:]/Values[:, 1:-1])
lambd = 0.97
weights1b = 1/5*np.ones([1, 5])
VaR1b, ES1b = WHSMeasurements(returns, alpha, lambd, weights1b, portfolioValue, riskMeasureTimeIntervalInDay)
print('VaR1b=', VaR1b, 'ES1b=', ES1b)


#Point C)
Values=np.array([df['ABI.BR'][823:1593].values,
                 df['AD.AS'][823:1593].values,
                 df['ADSGn.DE'][823:1593].values,
                 df['AIR.PA'][823:1593].values,
                 df['AIRP.PA'][823:1593].values,
                 df['ALVG.DE'][823:1593].values,
                 df['ASML.AS'][823:1593].values,
                 df['AXAF.PA'][823:1593].values,
                 df['BASFn.DE'][823:1593].values,
                 df['BAYGn.DE'][823:1593].values,
                 df['BBVA.MC'][823:1593].values,
                 df['BMWG.DE'][823:1593].values,
                 df['BNPP.PA'][823:1593].values,
                 df['CRH.I'][823:1593].values,
                 df['DANO.PA'][823:1593].values,
                 df['DB1Gn.DE'][823:1593].values,
                 df['DPWGn.DE'][823:1593].values,
                 df['DTEGn.DE'][823:1593].values,
                 df['ENEI.MI'][823:1593].values,
                 df['ENI.MI'][823:1593].values])
returns=np.log(Values[:,2:]/Values[:,1:-1])
yearlyMeanReturns=np.zeros([3,20])
yearlyMeanReturns[0,:]=returns[:,0:258].mean(1) #mean of the returns of the first year for each company
yearlyMeanReturns[1,:]=returns[:,259:513].mean(1) #mean of the returns of the second year for each company
yearlyMeanReturns[2,:]=returns[:,514:769].mean(1) #mean of the returns of the third year for each company
yearlyCovariance=np.cov(returns)

weights= 1/20*np.ones([1, 20])
H=
numberOfPrincipalComponents=6

ES, VaR = PrincCompAnalysis(yearlyCovariance, yearlyMeanReturns, weights, H, alpha,numberOfPrincipalComponents, portfolioValue)

