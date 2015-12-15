'''
The model is an implementation of EM algorithm of IRT


For reference, see:
Brad Hanson, IRT Parameter Estimation using the EM Algorithm, 2000
Eiji Muraki, A Generalized Partial Credit Model: Application of an EM Algorithm, 1992


The current version only deals with unidimension theta
The current version does not constrain slop in the multinomial case
The current version does not prevent perfect prediction
The current version only deal with ordered data, e.g. [2,4,5] will be mapped as [0,1,2]

If the item has dichotomous response, allow to set the guess parameter
If the item has polytomous response, does not allow set the guess parameter

'''
import numpy as np
from scipy.stats import norm
import time

from ..utl import clib, tools, loader
from ..solver import optimizer


class IRT_MMLE_2PL(object):

    '''
    Three steps are exposed
    (1) load data
    (2) set parameter
    (3) solve
    '''

    def load_data(self, src):
        '''
        # Input:
        user id, int or string
        item id, int or string
        response: user's answer to item, int or string
        
        # Output:
        1. Element map: user(_reverse)_map, item(_reverse)_map, response(_reverse)_map
           Translate between data index and the class index.
        
        2. Relation map: user2item, item2user, user2grade_item
           Cache frequently used maps.
        '''

        # TODO: map the choice of src in loader
        if isinstance(src, basestring):  # assume a valid path 
            # if the src is file handle
            user_ids, item_ids, responses = loader.load_file_handle(src)
        else:
            # if the src is list of tuples
            user_ids, item_ids, responses = loader.load_tuples(src)
        print('Data are loaded.')

        self.data_ref = loader.data_storage(user_ids, item_ids, responses)

        self._init_item_param()
        print('Data are preprocessed.')


    def load_param(self, theta_bnds, alpha_bnds, beta_bnds):

        '''
        # Input:
        1. theta_bnds: the range value for the latent ability. The package suggests [-4,4].
        2. alpha_bnds: the range value for item discrimination. The package suggests a lower bound of 0.25 and upper bound of 2 for each response component
        3. beta_bnds: the range value for item difficulty. The package suggests a range smaller than that of theta.

        # Output:
        1. Prior distribution of theta. 
        Value grid is default to be 11. The prior distribution is normal N(0, range/4)
        2. Solver choice:
        Default solver is BFGS. If failed to converge, try L-BFGS-B
        Dichotomous solver and polytomous solver uses different log-likelihood function
        '''
        # TODO: allow for a structured constraints on alpha

        # load user item
        num_theta = 11
        self._init_user_param(theta_bnds[0], theta_bnds[1], num_theta)

        # load the solver
        boundary = {'alpha': alpha_bnds,
                    'beta': beta_bnds}

        solver_type = 'gradient'
        is_constrained = True
        max_iter = 10
        tol = 1e-3

        self._init_solver_param(is_constrained, boundary, solver_type, max_iter, tol)



    def load_guess_param(self, in_guess_param):
        '''
        Input:
        in_guess_param: dictionary index by original eid
        '''
        for out_eid, c in in_guess_param.iteritems():
            # translate eid
            eid = self.data_ref.item_map[out_eid]
            # update
            self.item_param_dict[eid]['C'] = c

    def solve_EM(self):
        # create the inner parameters
        self.theta_distr = np.zeros((self.data_ref.num_user, self.num_theta))

        # TODO: enable the stopping condition
        num_iter = 1
        self.ell_list = []
        avg_prob_t0 = 0

        while True:
            iter_start_time = time.time()
            # add in time block
            start_time = time.time()
            self._exp_step()
            print("--- E step: %f secs ---" % np.round((time.time() - start_time)))

            start_time = time.time()
            self._max_step()
            print("--- M step: %f secs ---" % np.round((time.time() - start_time)))

            self.__calc_theta()

            '''
            Exp
            '''
            # self.update_guess_param()

            # the goal is to maximize the "average" probability
            avg_prob = np.exp(self.__calc_data_likelihood() / self.data_ref.num_log)
            self.ell_list.append(avg_prob)
            print("--- all: %f secs ---" % np.round((time.time() - iter_start_time)))
            print(avg_prob)

            # if the algorithm improves, then ell > ell_t0
            if avg_prob_t0 > avg_prob:
                # TODO: needs to roll back if the likelihood decrease
                print('Likelihood descrease, stops at iteration %d.' % num_iter)
                break

            if avg_prob_t0 < avg_prob and avg_prob - avg_prob_t0 <= self.tol:
                print('EM converged at iteration %d.' % num_iter)
                break
            # update the stop condition
            avg_prob_t0 = avg_prob
            num_iter += 1

            if (num_iter > self.max_iter):
                print('EM does not converge within max iteration')
                break

    def get_item_param(self):
        # need to remap the inner id to the outer id
        return self.item_param_dict

    def get_user_param(self):
        user_param_dict = {}
        for i in xrange(self.data_ref.num_user):
            uid = self.data_ref.uid_vec[i]
            user_param_dict[uid] = self.theta_vec[i]

        return user_param_dict

    '''
    Main Routine
    '''

    def _exp_step(self):
        '''
        Basic Math:
        Take expecation of the log likelihood (L).
        Since L is linear additive, its expectation is also linear additive.
        '''
        # TODO: better math explanation
        # (1) update the posterior distribution of theta
        self.__update_theta_distr()

        # (2) marginalize
        # because of the sparsity, the expected right and wrong may not sum up
        # to the total num of items!
        self.__get_expect_count()

    def _max_step(self):
        '''
        Basic Math
            log likelihood(param_j) = sum_k(log likelihood(param_j, theta_k))
        '''
        # [A] max for item parameter
        opt_worker = optimizer.irt_2PL_Optimizer()
        # the boundary is universal
        # the boundary is set regardless of the constrained option because the
        # constrained search serves as backup for outlier cases
        opt_worker.set_bounds([self.beta_bnds, self.alpha_bnds])

        # theta value is universal
        opt_worker.set_theta(self.theta_val)

        for eid in self.data_ref.eid_vec:
            # set the initial guess as a mixture of current value and a new
            # start to avoid trap in local maximum
            initial_guess_val = (self.item_param_dict[eid]['beta'],
                                 self.item_param_dict[eid]['alpha'])

            opt_worker.set_initial_guess(initial_guess_val)
            opt_worker.set_c(self.item_param_dict[eid]['c'])

            # assemble the expected data
            j = self.data_ref.eidx[eid]
            expected_right_count = self.item_expected_right_bytheta[:, j]
            expected_wrong_count = self.item_expected_wrong_bytheta[:, j]
            input_data = [expected_right_count, expected_wrong_count]
            opt_worker.load_res_data(input_data)
            # if one wishes to inspect the model input, print the input data

            est_param = opt_worker.solve_param_mix(self.is_constrained)

            # update
            self.item_param_dict[eid]['beta'] = est_param[0]
            self.item_param_dict[eid]['alpha'] = est_param[1]

        # [B] max for theta density
        # pi = r_k/(w_k+r_k)
        r_vec = np.sum(self.item_expected_right_bytheta, axis=1)
        w_vec = np.sum(self.item_expected_wrong_bytheta, axis=1)
        self.theta_density = np.divide(r_vec, r_vec + w_vec)

    '''
    Auxuliary function
    '''


    def _init_solver_param(self, is_constrained, boundary,
                           solver_type, max_iter, tol):

        # TODO:separate bnds for dichotomous and polytomous
        # initialize bounds
        self.is_constrained = is_constrained
        self.alpha_bnds = boundary['alpha']
        self.beta_bnds = boundary['beta']
        self.solver_type = solver_type
        self.max_iter = max_iter
        self.tol = tol

        if solver_type == 'gradient' and not is_constrained:
            raise Exception('BFGS has to be constrained')

    def _init_item_param(self):

        # follow MNlogit's convention, the parameters [b,a]
        # guess parameter is not considered here
        self.item_param_dict = {}
        for eid, responses in self.data_ref.response_map.iteritems():
            J = len(responses.keys())
            # Bs and As
            Bs = np.zeros((J-1, 1))
            As = np.ones((J-1, 1))
            self.item_param_dict[eid] = {'ab': np.column_stack((Bs, As)).T,
                                         'c': None}

    def _init_user_param(self, theta_min, theta_max, num_theta):
        self.theta_val = np.linspace(theta_min, theta_max, num=num_theta)
        self.num_theta = num_theta

        # prior density normal, make it fat
        sd = (theta_max - theta_min)/2.0

        theta_prior_normal_pdf = [norm.pdf(x, scale=sd) for x in self.theta_val]
        # N*1 * 1*S array
        prior_density = [x/sum(theta_prior_normal_pdf) for x in theta_prior_normal_pdf]  # normalize
        self.theta_density = np.dot(np.ones((self.data_ref.num_user,1)), np.array(prior_density)[None,:])



    def __update_theta_distr(self):
        for i in xrange(self.data_ref.num_user):
            full_ll_array = np.log(self.theta_density[i, :])
            for s in range(self.num_theta):
                full_ll_array[s] += tools.get_conditional_loglikelihood(self.data_ref.item2user[i], self.theta_val[s],
                                                         self.item_param_dict)

            self.theta_density[i, :] = tools.update_posterior_distribution(full_ll_array)

    def __get_expect_count(self):

        self.item_expected_right_bytheta = np.zeros((self.num_theta, self.data_ref.num_item))
        self.item_expected_wrong_bytheta = np.zeros((self.num_theta, self.data_ref.num_item))

        for j in range(self.data_ref.num_item):
            eid = self.data_ref.eid_vec[j]
            # get all the users that done it right
            # get all the users that done it wrong
            right_uid_vec, wrong_uid_vec = self.data_ref.get_rwmap(eid)
            # condition on the posterior ability, what is the expected count of
            # students get it right
            # TODO: for readability, should specify the rows and columns
            self.item_expected_right_bytheta[:, j] = np.sum(self.theta_density[right_uid_vec, :], axis=0)
            self.item_expected_wrong_bytheta[:, j] = np.sum(self.theta_density[wrong_uid_vec, :], axis=0)

    def __calc_data_likelihood(self):
        # calculate the likelihood for the data set

        ell = 0
        for i in range(self.data_ref.num_user):
            uid = self.data_ref.uid_vec[i]
            theta = self.theta_vec[i]
            # find all the eid
            logs = self.data_ref.get_log(uid)
            for log in logs:
                eid = log[0]
                grade = log[1]
                alpha = self.item_param_dict[eid]['alpha']
                beta = self.item_param_dict[eid]['beta']
                c = self.item_param_dict[eid]['c']

                ell += clib.log_likelihood_2PL(grade, 1 - grade,
                                               theta, alpha, beta, c)
        return ell

    def __calc_theta(self):
        self.theta_vec = np.dot(self.theta_density, self.theta_val) 
