from DateTime import DateTime
#from pandas import pd
import math as mt
import numpy as np
from numpy.linalg import eig
import scipy as sc
from scipy.stats import norm
import pandas as pd

def printer(x):
    print(x)

##point 0
def AnalyticalNormalMeasures(alpha, weights, portfolioValue, riskMeasureTimeIntervalInDay, returns):

    #Covariance matrix
    Cov=riskMeasureTimeIntervalInDay*(np.cov(returns))
    #mean of the normal distribution
    Mean=riskMeasureTimeIntervalInDay*(-np.sum(weights*returns.mean(1)))#(-np.dot(returns.mean(1),weights))
    #standard deviation of portfolio
    Standard_Dev=np.sqrt(np.sum(np.dot(weights,Cov)*weights))
    #Compute VaR
    VaR = portfolioValue * (Mean + Standard_Dev * norm.ppf(alpha))

    #Compute ES
    ES = portfolioValue * (Mean + Standard_Dev * norm.pdf(norm.ppf(alpha)) / (1 - alpha))
    return VaR, ES




##point 1
##a
def HSMeasurements(returns, alpha, weights, portfolioValue, riskMeasureTimeIntervalInDay):
    #vector of losses
    Loss=-portfolioValue*np.dot(returns.T,weights.T)
    #Order them in descending order
    Loss_desce=np.sort(Loss)[::-1]
    n=len(Loss_desce)
    VaR = riskMeasureTimeIntervalInDay*Loss_desce[mt.floor(n*(1-alpha))]
    ES = riskMeasureTimeIntervalInDay*np.mean(Loss_desce[0:mt.floor(n*(1-alpha))])

    return VaR,ES
##b
def WHSMeasurements(returns, alpha, lambd, weights, portfolioValue, riskMeasureTimeIntervalInDay):
    n=np.size(returns)/4
    #normalized factor
    C=(1-lambd)/(1-lambd**n)
    #weights exponentially decreasing
    w = C*lambd**(np.linspace(0, int(n) - 1, num=int(n)))

    Loss = -portfolioValue*np.dot(returns.T,weights.T).T
    Loss_desce = np.sort(Loss).T[::-1].T
    index_w = np.argsort(Loss)[::-1]
    w_desce = w[index_w]
    #We look for i_star: the largest value s.t. sum(w_i, i=1,..,i_star)<=1-alpha
    i = 1
    while np.sum(w_desce[0,0:i]) <= 1-alpha:
        i=i+1
    i_star=i-1

    VaR = riskMeasureTimeIntervalInDay*Loss_desce[0,i_star]
    ES = riskMeasureTimeIntervalInDay*(np.sum(w_desce[0,1:i_star].T*Loss_desce[0,1:i_star])/np.sum(w_desce[0,1:i_star]))
    return VaR, ES
##c

def PrincCompAnalysis(yearlyCovariance, yearlyMeanReturns, weights, H, alpha, numberOfPrincipalComponents, portfolioValue):
    eval, evect = np.linalg.eig(yearlyCovariance)

    #order eigenvalues in descending way
    eval_desce = np.sort(eval)[::-1]
    eval_index = np.argsort(eval)[::-1]
    evect=evect[:,eval_index]
    yearlyMeanReturns=yearlyMeanReturns[:,eval_index]

    #reduced form portfolio
    mu_hat =np.dot(evect.T , yearlyMeanReturns.T)
    w_hat = np.dot(evect.T , weights.T)

    #computing mean and variance of the reduced ptf up to K
    k=numberOfPrincipalComponents
    Mean_reduced = np.sum(mu_hat[1:k]*w_hat[1: k]) *H  #mu_red * delta
    Sigma_reduced = np.sqrt(np.sum(eval_desce[1:k]*(w_hat[1:k]**2))*H) #sqrt(sigma_red*delta)


    VaR = portfolioValue * (Mean_reduced+Sigma_reduced * norm.ppf(alpha)) #mu_red * delta +  sigma_red * sqrt(delta) * VaR_std
    ES = portfolioValue * (Mean_reduced+Sigma_reduced*  norm.pdf(norm.ppf(alpha)) / (1 - alpha)) #mu_red * delta +   sqrt(sigma_red *delta) * ES_std

    return VaR, ES

def plausibilityCheck(returns, portfolioWeights, alpha, portfolioValue, riskMeasureTimeIntervalInDay):
    # estimation of the order of magnitude of portfolio VaR
    C = np.corrcoef(returns.T,rowvar=False) #correlation matrix
    u=np.percentile(returns,alpha*100) #upper quantile
    l=np.percentile(returns,(1-alpha)*100) #lower quantile

    sVaR = portfolioWeights * (abs(l) + abs(u)) / 2  #signed-VaR
    VaR = np.sqrt(np.sum(np.dot(sVaR,C)*sVaR))*portfolioValue

    return VaR


    #samples = bootstrapStatistical(numberOfSamplesToBootstrap, returns)


#[ES, VaR] = WHSMeasurements(returns, alpha, lambda, weights, portfolioValue,riskMeasureTimeIntervalInDay)


#[ES, VaR] = PrincCompAnalysis(yearlyCovariance, yearlyMeanReturns, weights, H, alpha,numberOfPrincipalComponents, portfolioValue)


#VaR = plausibilityCheck(returns, portfolioWeights, alpha, portfolioValue, riskMeasureTimeIntervalInDay)


#VaR = FullMonteCarloVaR(logReturns, numberOfShares, numberOfPuts, stockPrice, strike, rate, dividend,volatility, timeToMaturityInYears, riskMeasureTimeIntervalInYears, alpha,NumberOfDaysPerYears)


#VaR = DeltaNormalVaR(logReturns, numberOfShares, numberOfPuts, stockPrice, strike, rate, dividend,volatility, timeToMaturityInYears, riskMeasureTimeIntervalInYears, alpha, NumberOfDaysPerYears)

#777