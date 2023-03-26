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




