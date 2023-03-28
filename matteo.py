#from pandas import pd
import math
import numpy as np
import pandas as pd
from utilities import AnalyticalNormalMeasures
from utilities import HSMeasurements
from utilities import WHSMeasurements
from utilities import PrincCompAnalysis
from utilities import plausibilityCheck
from utilities import FullMonteCarloVaR
import random

random.seed(134)

##0. Exercise. Variance-covariance method for VaR & ES in linear portfolio: a simple example of data mining
#import utilities
df= pd.read_csv('EUROSTOXX50_2023_Dataset.csv', index_col=False)
df.fillna(axis=0,method='ffill',inplace=True)

from DateTime import DateTime
df_3y=df[df['Date']<='2019-03-20']
df_3y=df_3y[df_3y['Date']>='2016-03-21']

#matrix with 4 rows (one for each company), with daily value on the columns
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
#Portfolio with: Total (25K shares), Danone (20K Shares), Sanofi (20K Shares), Volkswagen (10K Shares)
selected_columns1a=['TTEF.PA','DANO.PA','SASY.PA','VOWG_p.DE']
df1a=np.array(df_3y[selected_columns1a]).T #matrix with 3 rows (one for each company), with daily value on the columns
returns1a=np.log(df1a[:,1:]/df1a[:,0:-1])

#number of shares for each company
shares = 10**3 * np.array([25, 20, 20, 10])
#prices of the stocks at 20 march 2019: Total,Danone,Sanofi,Volkswagen
#prices=np.array([50.36,68.96,78.9858496,143.92])
df_3y = df_3y.set_index('Date')
prices=np.array(df_3y[selected_columns1a].loc['2019-03-20'])
position_val=shares*prices
portfolioValue=np.sum(position_val)
weights1a=position_val/portfolioValue
Var1a,ES1a=HSMeasurements(returns1a, alpha, weights1a, portfolioValue, riskMeasureTimeIntervalInDay)
print('Var1a=',Var1a,'ES1a=',ES1a)

#Statistical Bootstrap
M=200
n=np.size(returns1a)/4
Rand_simulation=np.random.randint(0,n,size=M)
returns1s=returns1a[:,Rand_simulation]
VaR1a2,ES1a2=HSMeasurements(returns1s,alpha,weights1a,portfolioValue,riskMeasureTimeIntervalInDay)
print('VaR1a2=',VaR1a2,'ES1a2=',ES1a2)

VaR_check1a = plausibilityCheck(returns1a, weights1a, alpha, portfolioValue, riskMeasureTimeIntervalInDay)
print('VaR_check1a=', VaR_check1a)


#Point B)
#Portfolio with: Adidas, Airbus, BBVA, BMW and Schneider
selected_columns1b=['ADSGn.DE','AIR.PA','BBVA.MC','BMWG.DE','SCHN.PA']
df1b=np.array(df_3y[selected_columns1b]).T #matrix with 5 rows (one for each company), with daily value on the columns
returns1b = np.log(df1b[:, 1:]/df1b[:, 0:-1])
lambd = 0.97
weights1b = 1/5*np.ones([1, 5])
VaR1b, ES1b = WHSMeasurements(returns1b, alpha, lambd, weights1b, portfolioValue, riskMeasureTimeIntervalInDay)
print('VaR1b=', VaR1b, 'ES1b=', ES1b)

VaR_check1b = plausibilityCheck(returns1b, weights1b, alpha, portfolioValue, riskMeasureTimeIntervalInDay)
print('VaR_check1b=', VaR_check1b)


#Point C)
#Portfolio with shares of the first 20 companies in the provided csv file “_indexes.csv”
#(leave out Adyen - due to missing data - and take Eni, the 21st, instead)
selected_columns1c=['ABI.BR','AD.AS','ADSGn.DE','AIR.PA','AIRP.PA','ALVG.DE','ASML.AS','AXAF.PA','BASFn.DE',
                    'BAYGn.DE','BBVA.MC','BMWG.DE','BNPP.PA','CRH.I','DANO.PA','DB1Gn.DE','DPWGn.DE','DTEGn.DE','ENEI.MI','ENI.MI']
df1c=np.array(df_3y[selected_columns1c]).T
returns1c=np.log(df1c[:,1:]/df1c[:,0:-1])
yearlyMeanReturns=np.zeros([3,20])

#metodo yearlyMeanReturns vettore 20x1
yearlyMeanReturns=returns1c.mean(1)*256


#calcolo portfoliovalue
#prices of the stocks at 20 march 2019: Total,Danone,Sanofi,Volkswagen
#prices=np.array([73.18,23.545,209.8,117.86,93.09919217355464,198.92,166.18,22.675,67.5,63.0,5.347610093999999,71.98,44.19,27.82,68.96,114.5,29.43,15.665,5.544,15.776])
#portfolioValue=200000

yearlyCovariance=np.zeros([20,20])
yearlyCovariance=np.cov(returns1c)*256

weights1c= 1/20*np.ones(20)  #equally weighted portfolio
H=10
numberOfPrincipalComponents=6
# numberOfPrincipalComponents=0
n=range(1,7)
ES1c=np.zeros((len(n),1))
VaR1c=np.zeros((len(n),1))
for i in n:
    VaR1c[i-1],ES1c[i-1]  = PrincCompAnalysis(yearlyCovariance, yearlyMeanReturns, weights1c, H, alpha, i , portfolioValue)

print('VaR1c=', VaR1c, 'ES1c=', ES1c)

#Plausibility Check
VaR_check1c = plausibilityCheck(returns1c, weights1c, alpha, portfolioValue, riskMeasureTimeIntervalInDay)
print('VaR_check1c=', VaR_check1c)

##3. Case study:
dates=pd.read_csv('dates.csv')
dates=dates['DATES']
discounts=pd.read_csv('discounts.csv')
discounts=discounts['DISCOUNTS']

df_2y=df[df['Date']<='2023-01-31']
df_2y=df_2y[df_2y['Date']>='2021-02-01']
df3=np.array(df_2y['VNAn.DE']).T
returns3=np.log(df3[1:]/df3[:-1])

alpha=0.99
strike=25
volatility=15.4/100
dividend=3.1/100 #dividend yield
notional_2=25870000
df = df.set_index('Date')
stockPrice=df['VNAn.DE'].loc['2023-01-31'] #25.87
numberOfShares=notional_2/stockPrice
numberOfPuts=numberOfShares
rate=2.391334809477566/100
timeToMaturityInYears=0.175342465753425 #yearfrac('31-Gen-2023','05-Apr-2023',3)
riskMeasureTimeIntervalInYears=10/256
NumberOfDaysPerYears=256
VaR = FullMonteCarloVaR(returns3, numberOfShares, numberOfPuts, stockPrice, strike, rate, dividend,volatility,
                        timeToMaturityInYears, riskMeasureTimeIntervalInYears, alpha,NumberOfDaysPerYears)
print('FullMCVar=',VaR_FullMC)

#
VaR_deltaNorm = DeltaNormalVaR(returns2, numberOfShares, numberOfPuts, stockPrice, strike, rate, dividend,volatility, timeToMaturityInYears, riskMeasureTimeIntervalInYears, alpha, NumberOfDaysPerYears)
print('VaR_deltaNorm=',VaR_deltaNorm)


## 3. Case study: Pricing in presence of counterparty risk
Notional=50*10**6
volatility= 25/100

