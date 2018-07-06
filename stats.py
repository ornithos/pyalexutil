import numpy as np
from scipy import special
from pyalexutil.manipulate import copy_tri

def t_fit(X, dof=3.5, iter=200, eps=1e-6):
    '''t_fit
    --- snaffled from Gist (cdipaolo/9dd6794a4f0a2889ef60a5effa419093)

    Estimates the mean and covariance of the dataset
    X (rows are datapoints) assuming they come from a
    student t likelihood with no priors and dof degrees
    of freedom using the EM algorithm.
    Implementation based on the algorithm detailed in Murphy
    Section 11.4.5 (page 362).
    :param X: dataset
    :type  X: np.array[n,d]
    :param dof: degrees of freedom for likelihood
    :type  dof: float > 2
    :param iter: maximum EM iterations
    :type  iter: int
    :param eps: tolerance for EM convergence
    :type  eps: float
    :return: estimated covariance, estimated mean, list of
             objectives at each iteration.
    :rtype: np.array[d,d], np.array[d], list[float]
    '''
    # initialize parameters
    D = X.shape[1]
    N = X.shape[0]
    cov = np.cov(X,rowvar=False)
    mean = X.mean(axis=0)
    mu = X - mean[None,:]
    delta = np.einsum('ij,ij->i', mu, np.linalg.solve(cov,mu.T).T)
    z = (dof + D) / (dof + delta)
    obj = [
        -N*np.linalg.slogdet(cov)[1]/2 - (z*delta).sum()/2 \
        -N*special.gammaln(dof/2) + N*dof*np.log(dof/2)/2 + dof*(np.log(z)-z).sum()/2
    ]

    # iterate
    for i in range(iter):
        # M step
        mean = (X * z[:,None]).sum(axis=0).reshape(-1,1) / z.sum()
        mu = X - mean.squeeze()[None,:]
        cov = np.einsum('ij,ik->jk', mu, mu * z[:,None])/N

        # E step
        delta = (mu * np.linalg.solve(cov,mu.T).T).sum(axis=1)
        delta = np.einsum('ij,ij->i', mu, np.linalg.solve(cov,mu.T).T)
        z = (dof + D) / (dof + delta)

        # store objective
        obj.append(
            -N*np.linalg.slogdet(cov)[1]/2 - (z*delta).sum()/2 \
            -N*special.gammaln(dof/2) + N*dof*np.log(dof/2)/2 + dof*(np.log(z)-z).sum()/2
        )

        if np.abs(obj[-1] - obj[-2]) < eps:
            break
    return cov, mean.squeeze(), obj


def build_ar1_cov_matrix(T, rho, sigma0, sigma):
    marginal_var = [sigma0 ** 2]
    for ii in range(1, T):
        marginal_var.append(marginal_var[-1] * rho ** 2 + sigma ** 2)

    M = np.diag(marginal_var)
    for ii in range(T - 1):
        M[ii, ii + 1:] = M[ii, ii] * (rho ** np.arange(1, T - ii))

    return axu.manipulate.copy_tri(M)


def build_ar1_prec_matrix(T, rho, sigma0, sigma):
    tmpdiag = np.concatenate((np.array([rho ** 2 / sigma ** 2 + 1 / sigma0 ** 2]),
                              np.repeat((1 + rho ** 2) / sigma ** 2, T - 1)))
    tmpdiag[-1] = 1 / sigma ** 2
    M = np.diag(tmpdiag)
    if T <= 1:
        return M

    subdiag = np.repeat(-rho / sigma ** 2, T - 1)
    M.ravel()[1:(T - 1) * T:T + 1] = subdiag  # upper superdiagonal
    M.ravel()[T:T * T:T + 1] = subdiag  # lower subdiagonal
    return M