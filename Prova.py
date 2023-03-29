def PrincCompAnalysis(yearlyCovariance, yearlyMeanReturns, weights, H, alpha, numberOfPrincipalComponents,
                      portfolioValue):
    # spectral decomposition of the variance covariance matrix
    eigenvalues, eigenvectors = linalg.eig(yearlyCovariance)
    gamma = np.zeros((len(eigenvalues), len(eigenvalues)))
    # we order the set of eigenvalues
    eigenvalues_sorted = np.sort(eigenvalues)
    weights_sorted = weights
    mean_sorted = yearlyMeanReturns
    for i in range(len(eigenvalues_sorted)):
        # We order the eigenvectors, the weights in the portfolio and the mean vector following the eigenvalues' order
        gamma[:, i] = eigenvectors[eigenvalues == eigenvalues_sorted[i]]
        weights_sorted[i] = weights[eigenvalues == eigenvalues_sorted[i]]
        mean_sorted[i] = yearlyMeanReturns[eigenvalues == eigenvalues_sorted[i]]
        # Projected weights
    weights_hat = gamma.T.dot(weights_sorted)
    # Projected mean vector
    mean_hat = gamma.T.dot(mean_sorted)
    # reduced standard deviation
    sigma_red = (H * (weights_hat[0:numberOfPrincipalComponents] ** 2).T.dot(eigenvalues_sorted[0:numberOfPrincipalComponents])) ** (1 / 2)
    # reduced mean
    mean_red = H * sum(mean_hat[0:numberOfPrincipalComponents] * weights_hat[0:numberOfPrincipalComponents])
    # VaR and ES with the usual formulas
    VaR = float(portfolioValue * (mean_red + sigma_red * st.norm.ppf(alpha)))
    ES = float(portfolioValue * (mean_red + sigma_red * st.norm.pdf(st.norm.ppf(alpha)) / (1 - alpha)))
    return ES, VaR
