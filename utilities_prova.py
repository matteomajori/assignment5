from DateTime import DateTime
#from pandas import pd
import math as mt
import numpy as np
from numpy.linalg import eig
import scipy as sc
from scipy.stats import norm
import pandas as pd

##point 0
def AnalyticalNormalMeasures(alpha, weights, portfolioValue, riskMeasureTimeIntervalInDay, returns):
    #weights 4x1
    #returns 769x4
    #Covariance matrix
    Cov=riskMeasureTimeIntervalInDay*(np.cov(returns.T)) #4x4

    mu=np.mean(returns, axis=0) #media sulle colonne dei returns mu=1x4
    Mean=riskMeasureTimeIntervalInDay*np.dot(weights.T,-mu)#(-np.dot(returns.mean(1),weights))
    #standard deviation of portfolio
    Standard_Dev=np.sqrt(np.dot(np.dot(weights.T,Cov),weights))
    #Compute VaR
    VaR = portfolioValue * (Mean + Standard_Dev * norm.ppf(alpha))

    #Compute ES
    ES = portfolioValue * (Mean + Standard_Dev * norm.pdf(norm.ppf(alpha)) / (1 - alpha))
    return VaR, ES

#1a
def HSMeasurements(returns, alpha, weights, portfolioValue, riskMeasureTimeIntervalInDay):
    #vector of losses
    Loss=-portfolioValue*np.dot(returns,weights)
    #Order them in descending order
    Loss_desce=np.sort(Loss)[::-1]
    n=len(Loss_desce)
    VaR = riskMeasureTimeIntervalInDay*Loss_desce[int(n*(1-alpha))]
    ES = riskMeasureTimeIntervalInDay*np.mean(Loss_desce[:int(n*(1-alpha))])

    return VaR,ES
#1b
def WHSMeasurements(returns, alpha, lambd, weights, portfolioValue, riskMeasureTimeIntervalInDay):
    #weights 5x1, returns 769x5
    #n=np.size(returns)
    n=returns.shape[0] #number of rows of the returns
    #normalized factor
    C=(1-lambd)/(1-lambd**n)
    #weights exponentially decreasing
    #w = C*lambd**(np.linspace(0, int(n) - 1, num=int(n)))
    w = C * lambd ** (np.arange(n-1,-1,-1))
    for i in range(n):
        w1[i]=C*lambd**i

    Loss = -portfolioValue*np.dot(returns,weights).T
    Loss_desce = np.sort(Loss).T[::-1].T
    index_w = np.argsort(Loss).T[::-1].T
    w_desce = w[index_w]
    # indices = np.argsort(Loss, kind='mergesort')
    # Loss_desce = Loss[indices]
    #We look for i_star: the largest value s.t. sum(w_i, i=1,..,i_star)<=1-alpha
    # i = 1
    # while np.sum(w_desce[0,0:i]) <= 1-alpha:
    #     i=i+1
    # i_star=i-1
    # Trova l'ultimo indice i tale che cum_w[i] <= 1 - alpha
    i_star = np.where(np.cumsum(w_desce )<= 1 - alpha)[0][-1]

    VaR = riskMeasureTimeIntervalInDay*Loss_desce[0,i_star]
    ES = riskMeasureTimeIntervalInDay*(np.sum(w_desce[0,0:i_star]*Loss_desce[0,0:i_star])/np.sum(w_desce[0,0:i_star]))
    return VaR, ES


def plausibilityCheck(returns, portfolioWeights, alpha, portfolioValue, riskMeasureTimeIntervalInDay):
    # estimation of the order of magnitude of portfolio VaR
    C = np.corrcoef(returns,rowvar=False) #correlation matrix
    #u=np.percentile(returns,alpha*100) #upper quantile
    u = np.quantile(returns, alpha) #upper quantile
    #l=np.percentile(returns,(1-alpha)*100) #lower quantile
    l = np.quantile(returns, (1 - alpha)) #lower quantile

    sVaR = portfolioWeights * (abs(l) + abs(u)) / 2  #signed-VaR
    VaR = np.sqrt(np.sum(np.dot(sVaR.T,C)*sVaR.T))*portfolioValue

    return VaR
