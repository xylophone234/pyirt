# -*- coding:utf-8 -*-

'''
Created on 2015/1/2

@author: junchen
'''

import numpy as np


def get_grade_prob(theta, As, Bs, c=None):
    J = len(Bs)
    if J < 1:
        raise ValueError('Beta vector is empty.')
    if J != len(As):
        raise ValueError('Alpha and beta have different length')

    ps = [1]
    # otherwise use the full specification
    for i in range(J):
        p = np.exp(sum([As[k]*theta+Bs[k] for k in range(i+1)]))
        ps.append(p)
    G = sum(ps)
    prob_vec = [x/G for x in ps]

    # if there is guess parameter involved
    if c is not None and J == 1:
        # the 1 response is changed to
        p1 = c + (1-c)*prob_vec[1]
        p0 = 1-p1
        prob_vec = [p0, p1]
    elif c is not None and J != 1:
        raise ValueError('Polytomous response does not allow guess parameter.')
    return prob_vec


def get_conditional_loglikelihood(records, theta, item_param_dict):
    '''
    Input:
    (1) record list: (eid, grade). All the items that users did
    (2) theta: user ability parameter
    (3) item param dictionary
    '''
    
    ll = 0
    for record in records:
        eid = record[0]
        grade = record[1]
        Bs = item_param_dict[eid]['ab'][0, :]
        As = item_param_dict[eid]['ab'][1, :]
        C = item_param_dict[eid]['c']
        prob = get_grade_prob(theta, As, Bs, C)
        ll += np.log(prob[grade])
    return ll


def logsum(logp):
    w = max(logp)
    logSump = w + np.log(sum(np.exp(logp - w)))
    return logSump


def update_posterior_distribution(full_loglikelihood_array):
    '''
    Input: 
    full loglikelihood is a N*1 numpy array
    '''
    # TODO: Input check

    marginal = logsum(full_loglikelihood_array)
    posterior = np.exp(full_loglikelihood_array - marginal)
    return posterior



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



