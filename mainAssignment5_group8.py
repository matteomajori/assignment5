#from pandas import pd
import math
import numpy as np
import pandas as pd
from utilities import *
import random
import matplotlib.pyplot as plt


##0. Exercise. Variance-covariance method for VaR & ES in linear portfolio: a simple example of data mining
#import the dataset and fill the missing returns with the previous one
df= pd.read_csv('EUROSTOXX50_2023_Dataset.csv', index_col=False)
df.fillna(axis=0,method='ffill',inplace=True)

from DateTime import DateTime
#select the date of interest
df_3y=df[df['Date']<='2019-03-20']
df_3y=df_3y[df_3y['Date']>='2016-03-21']
#matrix with 4 rows (one for each company), with daily value on the columns
#Ptf with Adidas, Allianz, Munich Re and L’Oréal
selected_columns0=['ADSGn.DE','ALVG.DE','MUVGn.DE','OREP.PA']
df0=np.array(df_3y[selected_columns0]).T


alpha=0.95
weights=1/4*np.ones([1,4]) #equally weighted
riskMeasureTimeIntervalInDay=1
portfolioValue=10**7
#compute the logreturns
returns0=np.log(df0[:,1:]/df0[:,0:-1])
#VaR and ES using Variance/Covariance method
Var0,ES0=AnalyticalNormalMeasures(alpha, weights, portfolioValue, riskMeasureTimeIntervalInDay, returns0)
print('Var0=',Var0,'ES0=',ES0)


##1. Case study: Historical (HS & WHS) Simulation, Bootstrap and PCA for VaR & ES in a linear portfolio

alpha = 0.99
#Point A)
#Portfolio with: Total (25K shares), Danone (20K Shares), Sanofi (20K Shares), Volkswagen (10K Shares)
selected_columns1a=['TTEF.PA','DANO.PA','SASY.PA','VOWG_p.DE']
df1a=np.array(df_3y[selected_columns1a]).T #matrix with 3 rows (one for each company), with daily value on the columns
#compute the logreturns
returns1a=np.log(df1a[:,1:]/df1a[:,0:-1])

#number of shares for each company: Total,Danone,Sanofi,Volkswagen
shares = 10**3 * np.array([25, 20, 20, 10])
#dates of interest
df_3y = df_3y.set_index('Date')
#prices of the stocks at 20 March 2019: Total,Danone,Sanofi,Volkswagen
prices=np.array(df_3y[selected_columns1a].loc['2019-03-20'])
#value of a single position
position_val=shares*prices
#Compute the value of the portfolio as sum of the single positions
portfolioValue=np.sum(position_val)
#weight for each position
weights1a=position_val/portfolioValue
#compute VaR and ES with Historical Simulation method
Var1a,ES1a=HSMeasurements(returns1a, alpha, weights1a, portfolioValue, riskMeasureTimeIntervalInDay)
print('Var1a=',Var1a,'ES1a=',ES1a)

#Statistical Bootstrap
M=200
n=np.size(returns1a)/4
#set seed
np.random.seed(50)
#simulate random indexes
Rand_simulation=np.random.randint(0,n,size=M)
#select random returns
returns1s=returns1a[:,Rand_simulation]
# Apply HS method with the selected random returns
VaR1a2,ES1a2=HSMeasurements(returns1s,alpha,weights1a,portfolioValue,riskMeasureTimeIntervalInDay)
print('VaR1a2=',VaR1a2,'ES1a2=',ES1a2)
#Plausibility Check
VaR_check1a = plausibilityCheck(returns1a, weights1a, alpha, portfolioValue, riskMeasureTimeIntervalInDay)
print('VaR_check1a=', VaR_check1a)


#Point B)
portfolioValue=10**7
#Portfolio with: Adidas, Airbus, BBVA, BMW and Schneider
selected_columns1b=['ADSGn.DE','AIR.PA','BBVA.MC','BMWG.DE','SCHN.PA']
#select the date of interest
df1b=np.array(df_3y[selected_columns1b]).T #matrix with 5 rows (one for each company), with daily value on the columns
#compute logreturns
returns1b = np.log(df1b[:, 1:]/df1b[:, 0:-1])
lambd = 0.97
weights1b = 1/5*np.ones([1, 5])
#compute VaR and ES using Weighted Historical Simulation method
VaR1b, ES1b = WHSMeasurements(returns1b, alpha, lambd, weights1b, portfolioValue, riskMeasureTimeIntervalInDay)
print('VaR1b=', VaR1b, 'ES1b=', ES1b)
#Plausibility Check for WHS
VaR_check1b = plausibilityCheck(returns1b, weights1b, alpha, portfolioValue, riskMeasureTimeIntervalInDay)
print('VaR_check1b=', VaR_check1b)


#Point C)
#Portfolio with shares of the first 20 companies in the provided csv file “_indexes.csv”
#(leave out Adyen - due to missing data - and take Eni, the 21st, instead)
selected_columns1c=['ABI.BR','AD.AS','ADSGn.DE','AIR.PA','AIRP.PA','ALVG.DE','ASML.AS','AXAF.PA','BASFn.DE',
                    'BAYGn.DE','BBVA.MC','BMWG.DE','BNPP.PA','CRH.I','DANO.PA','DB1Gn.DE','DPWGn.DE','DTEGn.DE','ENEI.MI','ENI.MI']
#select the date of interest
df1c=np.array(df_3y[selected_columns1c]).T
#compute logreturns
returns1c=np.log(df1c[:,1:]/df1c[:,0:-1])
yearlyMeanReturns=np.zeros([3,20])

#metodo yearlyMeanReturns array 20x1
yearlyMeanReturns=returns1c.mean(1)*256
yearlyCovariance=np.zeros([20,20])
#covariance matrix
yearlyCovariance=np.cov(returns1c)*256

weights1c= 1/20*np.ones(20)  #equally weighted portfolio
#time lag
H=10/256
n=range(1,21) #we use a greater range to plot the error vs Analytical measure
ES1c=np.zeros((len(n),1))
VaR1c=np.zeros((len(n),1))
#compute VaR and Es with Principal Component Analysis changing the number of components n=1:6
for i in n:
    VaR1c[i-1],ES1c[i-1]  = PrincCompAnalysis(yearlyCovariance, yearlyMeanReturns, weights1c, H, alpha, i , portfolioValue)
#print the first 6 components as asked in the exercise
print('VaR1c=', VaR1c[0:6], '\n' ,'ES1c=', ES1c[0:6])
#Plot of the PrincCompAnalysis and the exact result from exercise 0
VaR1C_correct, ES1C_correct = AnalyticalNormalMeasures(alpha, weights1c, portfolioValue, 10, returns1c)
#compute the VaR error using PCA
err_VaR = abs((VaR1c - VaR1C_correct) / VaR1C_correct)
#compute the ES error using PCA
err_ES = abs((ES1c - ES1C_correct) / ES1C_correct)
x=np.linspace(1,20,20)
fig = plt.figure()
plt.plot(x, err_VaR[:,0],label='VaR_Error')
plt.plot(x, err_ES[:,0], label='ES_Error')
plt.title('PCA errors')
plt.xlabel('Number of Principal Components')
plt.ylabel('Error')
plt.legend()
plt.show()
fig.savefig('PCA errors.png')


#Plausibility Check
VaR_check1c = plausibilityCheck(returns1c, weights1c, alpha, portfolioValue, 10)
print('VaR_check1c=', VaR_check1c)

##2. FullMonteCarlo, DeltaNormal and DeltaGamma methods for a portfolio of Vonovia stocks + put options
dates=pd.read_csv('dates.csv')
dates=dates['DATES']
discounts=pd.read_csv('discounts.csv')
discounts=discounts['DISCOUNTS']
#select date of interest
df_2y=df[df['Date']<='2023-01-31']
df_2y=df_2y[df_2y['Date']>='2021-02-01']
df2=np.array(df_2y['VNAn.DE']).T
#compute logreturns
returns2=np.log(df2[1:]/df2[:-1])

alpha=0.99
strike=25
volatility=15.4/100
dividend=3.1/100 #dividend yield
notional_2=25870000
df = df.set_index('Date')
#select Vonovia stock price at 31/01/2023
stockPrice=df['VNAn.DE'].loc['2023-01-31']
#Compute number of shares of Vonovia
numberOfShares=notional_2/stockPrice
#number of puts is the same
numberOfPuts=numberOfShares
#Rate imported from boostrap at 31/01/2023
rate=2.391334809477566/100
timeToMaturityInYears=0.175342465753425 #yearfrac('31-Gen-2023','05-Apr-2023',3)
#time-lag
riskMeasureTimeIntervalInYears=10/256
#number of working days in a year
NumberOfDaysPerYears=256
#VaR computed using FullMonteCarlo method
VaR_FullMC = FullMonteCarloVaR(returns2, numberOfShares, numberOfPuts, stockPrice, strike, rate, dividend,volatility,
                        timeToMaturityInYears, riskMeasureTimeIntervalInYears, alpha,NumberOfDaysPerYears)
print('FullMCVar=',VaR_FullMC)

#VaR using DeltaNormal method
VaR_deltaNorm = DeltaNormalVaR(returns2, numberOfShares, numberOfPuts, stockPrice, strike, rate, dividend,volatility, timeToMaturityInYears, riskMeasureTimeIntervalInYears, alpha, NumberOfDaysPerYears)
print('VaR_deltaNorm=',VaR_deltaNorm)
#Improvement of DeltaNormal method->DeltaGamma method
VaR_DeltaGammaNorm=DeltaGammaNormal(returns2, numberOfShares, numberOfPuts, stockPrice, strike, rate, dividend,
                     volatility, timeToMaturityInYears, riskMeasureTimeIntervalInYears, alpha, NumberOfDaysPerYears)
print('VaR_deltaGammaNorm=',VaR_DeltaGammaNorm)

## 3. Case study: Pricing in presence of counterparty risk
#Pricing the Cliquet option
#price of the stock of ISP at 02-02-2023 = 2.455
StockPrice=df['ISP.MI'].loc['2023-02-02']
volatility= 25/100
notional=50*10**6
#from assignment 3
recovery=40/100
#survival probability at payment dates of Cliquet option (every year from 02-02-2023 to 02-02-2027
SurvProbOnCliquet=np.array([1,0.99501246882793,0.988947643459277,0.982200758595402,0.975067029599282])
#risk-free rates
rates = np.array([0.031568541419429,0.031450996290326,0.029721790673799,0.028659033079144])
#discounts factor at payment dates
discounts=np.concatenate((np.array([0.968924542714243]),np.array(discounts)[12:15]))

#compute correct cliquet price and cliquet price at which ISP would try to sell it
cliquet_price, cliquet_riskfree_price = CliquetPrice(volatility,StockPrice,SurvProbOnCliquet,discounts,rates,recovery)
cliquet_price=cliquet_price*notional
cliquet_riskfree_price=cliquet_riskfree_price*notional
print('cliquet_price',cliquet_price,'cliquet_riskfree_price',cliquet_riskfree_price)
