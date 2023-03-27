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
    n=len(returns.T)
    #normalized factor
    C=(1-lambd)/(1-lambd**n)
    #weights exponentially decreasing
    w = C*lambd**(np.arange(0, n, 1))

    Loss = -portfolioValue*np.dot(returns.T,weights.T).T
    Loss_desce = np.sort(Loss).T[::-1].T
    index_w = np.argsort(Loss).T[::-1]

    # lambdas_sorted=np.zeros((len(Loss_desce),1))
    # for i in range(len(Loss_desce)):
    #     # we order the weights of the WHS following the order of the losses
    #     lambdas_sorted[i] = w[Loss == Loss_desce[0,i]]
    #

    w_desce = w[index_w].T
    #We look for i_star: the largest value s.t. sum(w_i, i=1,..,i_star)<=1-alpha
    # i = 1
    # while np.sum(w_desce[0,0:i]) <= 1-alpha:
    #     i=i+1
    # i_star=i-1
    i_star = np.where(np.cumsum(w_desce) <= 1 - alpha)[0][-1]+1

    VaR = riskMeasureTimeIntervalInDay*Loss_desce[0,i_star]
    ES = riskMeasureTimeIntervalInDay*(np.sum(w_desce[0,1:i_star].T*Loss_desce[0,1:i_star])/np.sum(w_desce[0,1:i_star]))
    return VaR, ES
##c

def PrincCompAnalysis(yearlyCovariance, yearlyMeanReturns, weights, H, alpha, numberOfPrincipalComponents, portfolioValue):
    eval, evect = np.linalg.eig(yearlyCovariance)

    #order eigenvalues in descending way
    eval_desce = np.sort(eval).T[::-1]
    eval_index = np.argsort(eval).T[::-1]
    #reordering the eigenvectors
    evect=evect[:,eval_index]
    yearlyMeanReturns=yearlyMeanReturns[eval_index]

    #reduced form portfolio
    mu_hat =np.dot(evect , yearlyMeanReturns)
    w_hat = np.dot(evect , weights.T)

    #computing mean and variance of the reduced ptf up to K
    k=numberOfPrincipalComponents
    Mean_reduced = -np.sum(mu_hat[0:k]*w_hat[0: k]) *H  #mu_red * delta
    Sigma_reduced = np.sqrt(np.sum(eval_desce[0:k]*np.square(w_hat[0:k]))*H) #sqrt(sigma_red*delta)


    VaR = portfolioValue * (Mean_reduced+Sigma_reduced * norm.ppf(alpha)) #mu_red * delta +  sigma_red * sqrt(delta) * VaR_std
    ES = portfolioValue * (Mean_reduced+Sigma_reduced*  norm.pdf(norm.ppf(alpha)) / (1 - alpha)) #mu_red * delta +   sqrt(sigma_red *delta) * ES_std

    return VaR, ES

def plausibilityCheck(returns, portfolioWeights, alpha, portfolioValue, riskMeasureTimeIntervalInDay):
    # estimation of the order of magnitude of portfolio VaR
    C = np.corrcoef(returns.T,rowvar=False) #correlation matrix
    #u=np.percentile(returns,alpha*100) #upper quantile
    u = np.quantile(returns, alpha) #upper quantile
    #l=np.percentile(returns,(1-alpha)*100) #lower quantile
    l = np.quantile(returns, (1 - alpha)) #lower quantile

    sVaR = portfolioWeights * (abs(l) + abs(u)) / 2  #signed-VaR
    VaR = np.sqrt(np.sum(np.dot(sVaR,C)*sVaR))*portfolioValue

    return VaR

def FullMonteCarloVaR(logReturns, numberOfShares, numberOfPuts, stockPrice, strike, rate, dividend, volatility,
                          timeToMaturityInYears, riskMeasureTimeIntervalInYears, alpha, NumberOfDaysPerYears):
    M = 10 ** 6
    # compute d1,d2
    d1 = (np.log(stockPrice / strike) + (rate + volatility ** 2 / 2.) * timeToMaturityInYears) / (
            volatility * np.sqrt(timeToMaturityInYears))
    d2 = d1 - volatility * np.sqrt(timeToMaturityInYears)
    call_price = stockPrice * norm.cdf(d1) - strike * np.exp(-rate * timeToMaturityInYears) * norm.cdf(d2)
    put_price = strike * np.exp(-rate * timeToMaturityInYears) - stockPrice * call_price
    # Random indexes: to check
    n = np.size(logReturns)
    Rand_simulation = np.random.randint(1, n, M)
    rand_returns = logReturns[:, Rand_simulation]
    stockPrice_new = stockPrice * np.exp(rand_returns)
    # compute put and call price at next step
    call_price_new = stockPrice_new * norm.cdf(d1) - strike * np.exp(-rate * timeToMaturityInYears) * norm.cdf(d2)
    put_price_new = strike * np.exp(-rate * timeToMaturityInYears) - stockPrice_new * call_price_new
    # Loss using MonteCarlo
    Loss = numberOfPuts * (-put_price_new + put_price) + numberOfShares * (-stockPrice_new + stockPrice)
    # compute VaR
    # the delta should be in days?
    VaR = (riskMeasureTimeIntervalInYears * np.percentile(Loss, 100 * alpha))
    return VaR

#samples = bootstrapStatistical(numberOfSamplesToBootstrap, returns)


#[ES, VaR] = WHSMeasurements(returns, alpha, lambda, weights, portfolioValue,riskMeasureTimeIntervalInDay)


#[ES, VaR] = PrincCompAnalysis(yearlyCovariance, yearlyMeanReturns, weights, H, alpha,numberOfPrincipalComponents, portfolioValue)


#VaR = plausibilityCheck(returns, portfolioWeights, alpha, portfolioValue, riskMeasureTimeIntervalInDay)


#VaR = FullMonteCarloVaR(logReturns, numberOfShares, numberOfPuts, stockPrice, strike, rate, dividend,volatility, timeToMaturityInYears, riskMeasureTimeIntervalInYears, alpha,NumberOfDaysPerYears)


#VaR = DeltaNormalVaR(logReturns, numberOfShares, numberOfPuts, stockPrice, strike, rate, dividend,volatility, timeToMaturityInYears, riskMeasureTimeIntervalInYears, alpha, NumberOfDaysPerYears)
