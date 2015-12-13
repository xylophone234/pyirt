# -*- coding:utf-8 -*-

'''
Created on 2015/1/2

@author: junchen
'''

import numpy as np


def get_grade_prob(theta, As, Bs,c = None):
    J = len(Bs)
    if J < 1:
        raise ValueError('Beta vector is empty.')
    if J != len(As):
        raise ValueError('Alpha and beta have different length')



    if c is not None:
        # if guess parameter is specified, use the two parameter IRT
        if J != 1:
            raise TypeError('Only dichotomous response can specify guess parameter')
        # TODO: make IRT parameter comparable to MNLogit
        prob_vec = [c, c + (1.0 - c) / (1 + np.exp(-(alpha * theta + beta)))]
    else:
        ps = [1]
        # otherwise use the full specification
        for i in range(J):
            p = exp(sum([As[k]*theta+Bs[k] for k in range(i+1)]))
            ps.append(p)
        G = sum(ps)
        prob_vec = [p/G for p in ps]
    
    return prob_vec




'''
Legacy code
'''
def irt_fnc(theta, beta, alpha=1.0, c=0.0):
    # beta is item difficulty
    # theta is respondent capability

    prob = c + (1.0 - c) / (1 + np.exp(-(alpha * theta + beta)))
    return prob


def log_likelihood_factor_gradient(y1, y0, theta, alpha, beta, c=0.0):
    temp = np.exp(beta + alpha * theta)
    grad = alpha * temp / (1.0 + temp) * (y1 * (1.0 - c) / (c + temp ) - y0 )

    return grad


def log_likelihood_factor_hessian(y1, y0, theta, alpha, beta, c=0.0):
    x = np.exp(beta + alpha * theta)
    # hessian = - alpha**2*(y1+y0)*temp/(1+temp)**2
    hessian = alpha ** 2 * x / (1 + x) ** 2 * (y1 * (1 - c) * (c - x ** 2) / (c + x) ** 2 - y0)

    return hessian


def log_likelihood_2PL_hessian(y1, y0, theta, alpha, beta, c=0.0):
    hessian = np.zeros((2, 2))
    x = np.exp(beta + alpha * theta)
    base = x / (1 + x) ** 2 * (y1 * (1 - c) * (c - x ** 2) / (c + x) ** 2 - y0)

    hessian = np.matrix([[1, theta], [theta, theta ** 2]]) * base

    return hessian


def logsum(logp):
    w = max(logp)
    logSump = w + np.log(sum(np.exp(logp - w)))
    return logSump
