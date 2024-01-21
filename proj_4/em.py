"""Mixture model for matrix completion"""
from typing import Tuple
import numpy as np
from scipy.special import logsumexp
from common import GaussianMixture


def estep(X: np.ndarray, mixture: GaussianMixture) -> Tuple[np.ndarray, float]:
    """E-step: Softly assigns each datapoint to a gaussian component

    Args:
        X: (n, d) array holding the data, with incomplete entries (set to 0)
        mixture: the current gaussian mixture

    Returns:
        np.ndarray: (n, K) array holding the soft counts
            for all components for all examples
        float: log-likelihood of the assignment

    """
    l = 0
    n , d = X.shape
    K , _ = mixture.mu.shape
    post = np.zeros((n,K))

    mv_gauss = lambda x_n , mu_k , var_k , dim : ((2 * np.pi * var_k)**(-dim/2)) * np.exp(-0.5 * (np.linalg.norm(x_n - mu_k)**2) / var_k)
    log_gauss = lambda x_n , mu_k , var_k , dim : (-dim/2) * (np.log(2) + np.log(np.pi) + np.log(var_k)) + (-0.5 * (np.linalg.norm(x_n - mu_k)**2) / var_k)

    for u in range(n):
        C_u = np.array([i for i in range(d) if X[u,i] != 0])
        X_C = np.array([X[u,i] for i in C_u])
        C = len(C_u)
        H = d - C
        for j in range(K):
            p = mixture.p[j]
            var = mixture.var[j]
            mu = mixture.mu[j,:]
            mu_C = np.array([mu[i] for i in C_u])

            #f = np.log(p + 1e-16) + np.log(mv_gauss(X_C,mu_C,var,C))
            f = np.log(p + 1e-16) + log_gauss(X_C, mu_C, var , C)
            post[u,j] = f
            #l += f
        J = logsumexp(post[u,:])
        post[u,:] = post[u,:] - J #log posteriors l(j|u) for a given X_C^(u)
        post[u,:] = np.exp(post[u,:])#posteriors p(j|u) for a given X_C^(u)
        #total_prob = np.sum(post[u,:])
        l += np.sum(post[u,:] * J)


    #raise NotImplementedError
    return post , l


def mstep(X: np.ndarray, post: np.ndarray, mixture: GaussianMixture,
          min_variance: float = .25) -> GaussianMixture:
    """M-step: Updates the gaussian mixture by maximizing the log-likelihood
    of the weighted dataset

    Args:
        X: (n, d) array holding the data, with incomplete entries (set to 0)
        post: (n, K) array holding the soft counts
            for all components for all examples
        mixture: the current gaussian mixture
        min_variance: the minimum variance for each gaussian

    Returns:
        GaussianMixture: the new gaussian mixture
    """
    #Get Constants
    n , d = X.shape
    _ , K = post.shape
    Z = np.sign(X)

    #compute mu's
    #mu = np.dot(post.T,X)
    #mu = np.divide(mu,np.sum(np.dot(post.T,Z),0))
    mu = mixture.mu
    for j in range(K):
        for l in range(d):
            mu_hat = np.multiply(post[:,j],X[:,l])
            p_hat = np.multiply(post[:,j],Z[:,l])
            mu_hat = np.sum(mu_hat)
            p_hat = np.sum(p_hat)
            if p_hat > 1:
                mu[j,l] = mu_hat/p_hat
    #compute p's
    p = np.sum(post,0) / n

    #compute var's
    '''
    var = np.zeros((n,K))
    p_star = np.zeros((n,K))
    for u in range(n):
        C_u = np.array([i for i in range(d) if X[u,i] != 0])
        C = C_u.size
        p_hat = post[u,:]
        p_star[u,:] = C * p_hat
        X_u = np.array([X[u,i] for i in C_u])
        for j in range(K):
            mu_u = np.array([mu[j,i] for i in C_u])
            var[u,j] = post[u,j] * (np.linalg.norm(X_u - mu_u)**2)
    var = np.sum(var,0)
    p_star = np.sum(p_star,0)
    var = np.divide(var,p_star)

    min_var = np.ones(var.shape) * min_variance
    var = np.maximum(var,min_var)
    '''
    var = np.zeros(K)
    for j in range(K):
        num = 0
        den = 0
        for u in range(n):
            C_u = np.array([i for i in range(d) if X[u,i] != 0])
            C = C_u.size
            X_u = np.array([X[u,i] for i in C_u])
            mu_u = np.array([mu[j,i] for i in C_u])
            num += (post[u,j] * np.linalg.norm(X_u - mu_u)**2)
            den += (C * post[u,j])
        var[j] = (num/den)
    min_var = np.ones(var.shape) * min_variance
    var = np.maximum(var,min_var)

    #raise NotImplementedError

    return GaussianMixture(mu , var , p)
    #return mu , var , p , Z


def run(X: np.ndarray, mixture: GaussianMixture,
        post: np.ndarray) -> Tuple[GaussianMixture, np.ndarray, float]:
    """Runs the mixture model

    Args:
        X: (n, d) array holding the data
        post: (n, K) array holding the soft counts
            for all components for all examples

    Returns:
        GaussianMixture: the new gaussian mixture
        np.ndarray: (n, K) array holding the soft counts
            for all components for all examples
        float: log-likelihood of the current assignment
    """
    loss = 0
    estimate = estep(X,mixture)
    mx = mixture
    new_loss = estimate[1]
    while np.abs(loss - new_loss) > (10**-6 * np.abs(new_loss)):
        mx = mstep(X,estimate[0],mx)
        loss = new_loss
        estimate = estep(X,mx)
        new_loss = estimate[1]

    #raise NotImplementedError
    return mx,estimate[0],estimate[1]


def fill_matrix(X: np.ndarray, mixture: GaussianMixture) -> np.ndarray:
    """Fills an incomplete matrix according to a mixture model

    Args:
        X: (n, d) array of incomplete data (incomplete entries =0)
        mixture: a mixture of gaussians

    Returns
        np.ndarray: a (n, d) array with completed data
    """
    raise NotImplementedError
