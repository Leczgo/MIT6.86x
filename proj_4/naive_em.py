"""Mixture model using EM"""
from typing import Tuple
import numpy as np
from common import GaussianMixture



def estep(X: np.ndarray, mixture: GaussianMixture) -> Tuple[np.ndarray, float]:
    """E-step: Softly assigns each datapoint to a gaussian component

    Args:
        X: (n, d) array holding the data
        mixture: the current gaussian mixture

    Returns:
        np.ndarray: (n, K) array holding the soft counts
            for all components for all examples
        float: log-likelihood of the assignment
    """
    l = 0
    n , d = X.shape #get dimensions of data
    K , _ = mixture.mu.shape
    post = np.zeros((n,K))
    for i in range(n):
        #tiled_vector = np.tile(X[i,:],(K,1))
        tiled_vector = X[i,:]
        for j in range(K):
            var = mixture.var[j]
            G_prob = ((2 * np.pi * var)**(-d/2)) * np.exp(-0.5 * (np.linalg.norm(tiled_vector - mixture.mu[j,:])**2) / var)
            
            #G = np.random.multivariate_normal(mixture.mu[j,:],var * np.identity(d))
            #G_prob = G(tiled_vector)

            #prob = np.dot(np.transpose(mixture.p),G_prob)
            #l += np.log(prob[0])
            #post[i,j] = G_prob

            post[i,j] = mixture.p[j] * G_prob
        #total_prob = np.sum(post[i,:])
        total_prob = np.sum(post[i,:])
        post[i,:] = post[i,:] / total_prob
        l += np.sum(post[i,:] * np.log(total_prob))

    #l = np.log(np.sum(post))
    #for i in range(n):
    #    for j in range(K):
    #        l += mixture.p[j] * np.log(post[i,j])
    #for j in range(K):
    #    Pi = post[j]
    #    l += np.sum(np.log(Pi))


    '''Need to define log likelihood'''

    #raise NotImplementedError
    return post , l


def mstep(X: np.ndarray, post: np.ndarray) -> GaussianMixture:
    """M-step: Updates the gaussian mixture by maximizing the log-likelihood
    of the weighted dataset

    Args:
        X: (n, d) array holding the data
        post: (n, K) array holding the soft counts
            for all components for all examples

    Returns:
        GaussianMixture: the new gaussian mixture
    """
    #get constants
    n,d = X.shape
    _ , K = post.shape

    #compute mu's
    mu = np.dot(post.T,X)
    mu = np.divide(mu.T,np.sum(post,0)).T

    #compute proportions
    p = np.sum(post,0) / n

    #compute var's
    var = np.zeros((n,K))
    for j in range(K):
        for i in range(n):
            var[i,j] = post[i,j] * (np.linalg.norm(X[i,:] - mu[j,:])**2)
    
    var = np.sum(var,0)
    var = np.divide(var,p * d * n)

    #raise NotImplementedError
    return GaussianMixture(mu , var , p)


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
    new_loss = estimate[1]
    while np.abs(loss - new_loss) > (10**-6 * np.abs(new_loss)):
        mx = mstep(X,estimate[0])
        loss = new_loss
        estimate = estep(X,mx)
        new_loss = estimate[1]

    #raise NotImplementedError
    return mx,estimate[0],estimate[1]
