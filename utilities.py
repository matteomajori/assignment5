import random

import numpy.random
from DateTime import DateTime
#from pandas import pd
import math as mt
import numpy as np
from numpy.linalg import eig
import scipy as sc
from scipy.stats import norm
import math
import pandas as pd

##point 0
def AnalyticalNormalMeasures(alpha, weights, portfolioValue, riskMeasureTimeIntervalInDay, returns):
    #Computes VaR and ES using VarianceCovariance method

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
    # Computes VaR and ES using Historical Simulation method
    #vector of losses
    Loss=-portfolioValue*np.dot(returns.T,weights.T)
    #Order them in descending order
    Loss_desce=np.sort(Loss)[::-1]
    n=len(Loss_desce)
    #Compute VaR
    VaR = riskMeasureTimeIntervalInDay*Loss_desce[mt.floor(n*(1-alpha))]
    #Compute ES
    ES = riskMeasureTimeIntervalInDay*np.mean(Loss_desce[0:mt.floor(n*(1-alpha))])

    return VaR,ES
##b
def WHSMeasurements(returns, alpha, lambd, weights, portfolioValue, riskMeasureTimeIntervalInDay):
    # Computes VaR and ES using Weighted Historical Simulation method
    n=len(returns.T)
    #normalized factor
    C=(1-lambd)/(1-lambd**n)
    #weights exponentially decreasing in the past
    w = C*lambd**(np.arange(n-1, -1, -1))
    #loss of the portfolio
    Loss = -portfolioValue*np.dot(returns.T,weights.T).T
    #we sort the loss in decreasing order
    Loss_desce = np.sort(Loss).T[::-1].T
    #find the indexes of the descending loss
    index_w = np.argsort(Loss).T[::-1]
    #order the weights with the correspondent loss
    w_desce = w[index_w].T
    #We look for i_star: the largest value s.t. sum(w_i, i=1,..,i_star)<=1-alpsha
    i_star = np.where(np.cumsum(w_desce) <= 1 - alpha)[0][-1]+1
    #compute VaR
    VaR = riskMeasureTimeIntervalInDay*Loss_desce[0,i_star]
    #compute ES
    ES = riskMeasureTimeIntervalInDay*(np.sum(w_desce[0,0:i_star]*Loss_desce[0,0:i_star])/np.sum(w_desce[0,0:i_star]))
    return VaR, ES
##c
def PrincCompAnalysis(yearlyCovariance, yearlyMeanReturns, weights, H, alpha, numberOfPrincipalComponents, portfolioValue):
    eval, evect = np.linalg.eig(yearlyCovariance)
    #Compute VaR and ES using the Principal Component Analysis method
    #order eigenvalues in descending way
    eval_desce = np.sort(eval).T[::-1]
    #find the indexes of the descending loss
    eval_index = np.argsort(eval).T[::-1]
    #reordering the eigenvectors
    evect=evect[:,eval_index]
    #reordering the returns using the index
    yearlyMeanReturns=yearlyMeanReturns[eval_index]

    #define reduced form portfolio
    mu_hat =np.dot(evect.T , yearlyMeanReturns.T)
    w_hat = np.dot(evect.T , weights.T)

    #computing mean and variance of the reduced ptf up to number of components K
    k=numberOfPrincipalComponents
    #compute mean and variance of the reduced portfolio
    Mean_reduced = -np.sum(mu_hat[0:k]*w_hat[0: k]) *H  #mu_red * delta
    Sigma_reduced = np.sqrt(np.sum(eval_desce[0:k]*np.square(w_hat[0:k]))*H) #sqrt(sigma_red*delta)
    #compute VaR
    VaR = portfolioValue * (Mean_reduced+Sigma_reduced * norm.ppf(alpha)) #mu_red * delta +  sigma_red * sqrt(delta) * VaR_std
    #compute ES
    ES = portfolioValue * (Mean_reduced+Sigma_reduced*  norm.pdf(norm.ppf(alpha)) / (1 - alpha)) #mu_red * delta +   sqrt(sigma_red *delta) * ES_std

    return VaR, ES

def plausibilityCheck(returns, portfolioWeights, alpha, portfolioValue, riskMeasureTimeIntervalInDay):
    # Estimation of the order of magnitude of portfolio VaR
    C = riskMeasureTimeIntervalInDay*np.corrcoef(returns.T,rowvar=False) #correlation matrix

    u = np.quantile(returns, alpha, axis=1) #upper quantile along axis 1 (i.e row-wise)
    l = np.quantile(returns, (1 - alpha), axis=1) #lower quantile along axis 1 (i.e row-wise)
    #compute sensitivities
    sens=-portfolioValue * portfolioWeights
    #compute sign VaR as sVaR = portfolioValue* portfolioWeights * (abs(l) + abs(u))/ 2
    sVaR = sens * (abs(l) + abs(u))/ 2  #signed-VaR
    #compute VaR
    VaR = np.sqrt(np.sum(np.dot(sVaR,C)*sVaR)) #*portfolioValue

    return VaR

def FullMonteCarloVaR(logReturns, numberOfShares, numberOfPuts, stockPrice, strike, rate, dividend, volatility,
                          timeToMaturityInYears, riskMeasureTimeIntervalInYears, alpha, NumberOfDaysPerYears):
    #compute VaR using Full MOnteCarlo Valuation method
    M = 10 ** 6
    put_price = PutPrice(rate,stockPrice,strike,dividend,volatility,timeToMaturityInYears)
    # Random indexes: to check
    n = len(logReturns.T)
    rand_simulation = np.random.randint(1, n, M)
    rand_returns = logReturns[rand_simulation]
    #oppure
    #mean=mean(logReturns.T)
    #stdev = np.sqrt(np.cov(returns))
    #rand_returns = mean+stdev*rand_simulation

    stockPrice_new = stockPrice * np.exp(rand_returns) #stockPrice_new = stockPrice * np.exp(logReturns)
    timeToMat_New=timeToMaturityInYears-riskMeasureTimeIntervalInYears
    put_price_new = PutPrice(rate, stockPrice_new, strike, dividend, volatility, timeToMat_New)
    # Loss using MonteCarlo
    Loss = numberOfPuts * (-put_price_new + put_price) + numberOfShares * (-stockPrice_new + stockPrice)
    # compute VaR
    # we call time lag the delta of the VaR
    time_lag=riskMeasureTimeIntervalInYears*NumberOfDaysPerYears
    VaR = np.sqrt(time_lag) * np.quantile(Loss, alpha)
    return VaR

def DeltaNormalVaR(logReturns, numberOfShares, numberOfPuts, stockPrice, strike, rate, dividend,
                     volatility, timeToMaturityInYears, riskMeasureTimeIntervalInYears, alpha, NumberOfDaysPerYears):
    M = 10 ** 6
    n = len(logReturns)
    # simulate a random set of dimension M of integer number between 1 and n
    Rand_simulation = np.random.randint(1, n, M)
    # select the corresponding returns
    rand_returns = logReturns[Rand_simulation]
    timeToMat_New = timeToMaturityInYears - riskMeasureTimeIntervalInYears
    d1 = (np.log(stockPrice / strike) + ( rate - dividend + volatility ** 2 / 2.) * timeToMat_New) / (
            volatility * np.sqrt(timeToMat_New))

    delta_put=-np.exp(-dividend * timeToMat_New) * norm.cdf(-d1)

    #compute the price of the stock at t + time_lag using the random set of returns
    stockPrice_new = stockPrice * np.exp(rand_returns)
    Loss = -(numberOfShares + delta_put * numberOfPuts)*stockPrice_new* rand_returns
    #VaR
    # we call time lag the delta of the VaR
    time_lag = int(riskMeasureTimeIntervalInYears * NumberOfDaysPerYears)
    VaR = np.sqrt(time_lag)* np.percentile(Loss, 100 * alpha)
    return VaR

def DeltaGammaNormal(logReturns, numberOfShares, numberOfPuts, stockPrice, strike, rate, dividend,
                     volatility, timeToMaturityInYears, riskMeasureTimeIntervalInYears, alpha, NumberOfDaysPerYears):
    M = 10 ** 6
    n = len(logReturns)
    #simulate a random set of dimension M of integer number between 1 and n
    Rand_simulation = np.random.randint(1, n, M)
    #select the corresponding returns
    rand_returns = logReturns[Rand_simulation]

    d1 = (np.log(stockPrice / strike) + (rate - dividend + volatility ** 2 / 2.) * timeToMaturityInYears) / (
            volatility * np.sqrt(timeToMaturityInYears))
    delta_put = -np.exp(-dividend * timeToMaturityInYears) * norm.cdf(-d1)

    gamma_put = np.exp(-dividend * timeToMaturityInYears) * norm.pdf(d1)/(stockPrice*volatility*np.sqrt(timeToMaturityInYears))

    #compute the price of the stock at t + time_lag using the random set of returns
    stockPrice_new = stockPrice * np.exp(rand_returns)
    #sensitivities
    delta_sensitivities = numberOfShares * stockPrice_new + delta_put * stockPrice_new  * numberOfPuts
    gamma_sensitivities = numberOfPuts * stockPrice_new**2 *gamma_put
    #Loss as deltanormal case + the gamma factor
    Loss = -delta_sensitivities* rand_returns - (gamma_sensitivities * rand_returns**2 )/2
    # we call time lag the delta of the VaR
    time_lag =int(riskMeasureTimeIntervalInYears * NumberOfDaysPerYears)
    VaR = np.sqrt(time_lag) * np.percentile(Loss, 100 * alpha)
    return VaR


def PutPrice(rate,stockPrice,strike,dividend,volatility,timeToMaturityInYears):
    d1 = (np.log(stockPrice / strike) + ( rate - dividend + volatility ** 2 / 2.) * timeToMaturityInYears) / (
            volatility * np.sqrt(timeToMaturityInYears))
    d2 = d1 - volatility * np.sqrt(timeToMaturityInYears)
    F=stockPrice * np.exp((rate -dividend) * timeToMaturityInYears)

    #price = np.exp(-rate * timeToMaturityInYears) * (F * norm.cdf(-d1) - strike * norm.cdf(-d2))
    price = np.exp(-rate * timeToMaturityInYears) * (-F * norm.cdf(-d1) + strike * norm.cdf(-d2))
    return price


def CliquetPrice_numerical(volatility,StockPrice,SurvProb,discounts,rates,recovery):
    M = 10 ** 6
    #random set of realization of standard normal for every year
    u = np.random.standard_normal((M,4))
    #inizializations
    s=np.zeros((M,4))
    payoff=np.zeros((M,4))
    #compute S_ti and the payoff of the cliquet

    #simulated stock at the first year using the first column of the random set
    s[:,0]= StockPrice *  np.exp(rates[0] -0.5 * volatility**2 + volatility*u[:,0])
    #cumulative sum over years
    u_cumsum=np.cumsum(u,axis=1)
    #simulated stock at the others years using the cumulative sum
    s = StockPrice * np.exp((rates - 0.5 * volatility ** 2) * np.arange(1,5) + volatility * u_cumsum)
    #compute payoff as the positive part of the difference of the stock and the strike using the stock at the previous year
    payoff[:, 0] = np.maximum(s[:, 0] - StockPrice, 0)
    payoff[:,1:] = np.maximum(s[:, 1:] - s[:, :-1], 0)
    
    #compute the correct price of the cliquet
    Cliquet = np.sum((np.mean(payoff,axis=0)*discounts)*SurvProb[1:]) #+recovery * np.sum(np.mean(payoff,axis=0)*discounts*(SurvProb[:- 1]-SurvProb[1:]))
    #compute the price with no risk of default
    Cliquet_riskfree = np.sum((np.mean(payoff,axis=0)*discounts))

    return Cliquet, Cliquet_riskfree
