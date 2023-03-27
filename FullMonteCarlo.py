

def FullMonteCarloVaR(logReturns, numberOfShares, numberOfPuts, stockPrice, strike, rate, dividend,volatility, timeToMaturityInYears, riskMeasureTimeIntervalInYears, alpha,NumberOfDaysPerYears)
    M=10**6
    #compute d1,d2
    d1=(np.log(stockPrice / strike) + (rate + volatility** 2 / 2.) * timeToMaturityInYears) / (volatility * sqrt(timeToMaturityInYears))
    d2=d1 - volatility * np.sqrt(timeToMaturityInYears)
    call_price = stockPrice*norm.cdf(d1)-strike*np.exp(-rate*timeToMaturityInYears)*norm.cdf(d2)
    put_price = strike * np.exp(-rate * timeToMaturityInYears) - stockPrice * call_price
    #Random indexes: to check
    n = np.size(logReturns)
    Rand_simulation = np.random.randint(1, n - 1, M)
    rand_returns = logReturns[:,Rand_simulation]
    stockPrice_new = stockPrice*np.exp(rand_returns)
    #compute put and call price at next step
    call_price = stockPrice_new * norm.cdf(d1) - strike * np.exp(-rate * timeToMaturityInYears) * norm.cdf(d2)
    put_price_new = strike * np.exp(-rate * timeToMaturityInYears) - stockPrice_new * call_price
    #Loss using MonteCarlo
    Loss = numberOfPuts * (-put_price_new+put_price) + numberOfShares*(-stockPrice_new + stockPrice)
    #compute VaR
    #the delta should be in days?
    VaR=(riskMeasureTimeIntervalInYears * np.percentile(Loss, 100*alpha))

    return VaR