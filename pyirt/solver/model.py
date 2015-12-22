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
            self.item_param_dict[eid]['c'] = c

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
        # (1) update the posterior distribution of theta
        self.__update_theta_distr()

        # (2) marginalize
        # because of the sparsity, the expected right and wrong may not sum up
        # to the total num of items!
        self.__get_expect_count()

    def _max_step(self):
        '''
        # Notes on the solver
        (1) If it is dichotomous, using the modified 3PL solver
        The 3PL solver using fixed, rather than estimated guess parameter
        (2) If it is polynomous, using the modified MNlogit solver
        The MNlogit does not impose constraints on fixed effect decomposition, ie b_{jk} != b_j+c_k.
        It does not impose constrains on distinguishing power, i.e. a_{jk} != k*a_{j,1}

        Currently the alogrithm imposes block diagnoal constraints on item parameters.
        It allows the alogrithm to estimate a large quantity of item parameters without running into the problem of inverting a giantic matrix
        The price is slower convergence rate and more boundary conditions. 
        '''

        for eid in range(self.data_ref.num_item):
            #TODO: get rid of the keys step
            if len(self.data_ref.response_map[eid].keys()) > 2:
                est_param = self.__solve_MNLogit(eid)
            else:
                est_param = self.__solve_3PL(eid)

            self.item_param_dict[eid]['ab'] = est_param

        # [B] max for theta density
        #TODO: resolve the constrained maximization problem
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
        '''
        # Input:
        (1) prior density of theta distribution
        (2) item parameters
        (3) response data

        calculate p(\theta_t,Y|param_{t+1}), then update p(theta_{t+1}|Y,param_{t+1})

        # output:
        (1) posterior distribution of theta
        '''
        for i in xrange(self.data_ref.num_user):
            full_ll_array = np.log(self.theta_density[i, :])
            for s in range(self.num_theta):
                full_ll_array[s] += tools.get_conditional_loglikelihood(self.data_ref.item2user[i], self.theta_val[s],
                                                         self.item_param_dict)

            self.theta_density[i, :] = tools.update_posterior_distribution(full_ll_array)

    def __get_expect_count(self):
        '''
        # Input:
        (1) user2grade_item : a map of users to item(j)*grade(k) grid
        (2) distribution of theta

        the expected count of student i answering item j with category k under theta l is
        p(\theta_{i,l})*Y_{j,k}
        
        Thus the population expected count of answering item j with category under theta l is 
        sum_{i} p(\theta_{i,l})*Y_{j,k}


        # Output:
        (1) expected count of response variables

        '''
        # TODO: produce more general unit test
        # generate the expected count dicitionary
        self.exp_response_data = [self.data_ref.get_response_space() for l in range(self.num_theta)]

        for l in range(self.num_theta):
            for eid, grade_map in self.data_ref.user2grade_item.items():
                for grade, user_list in grade_map.items():
                    self.exp_response_data[l][eid][grade] = self.theta_density[user_list, l].sum()


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


    def __solve_3PL(self, eid):
        '''
        # Input:
        (1) initial guess value {'ab':np array[Bs,As], 'c':val}
        (2) expected item count under theta l
        (3) theta value vector
        (4) constraints
        # Output:
        np array [beta, alpha]
        '''
        # [A] max for item parameter
        opt_worker = optimizer.irt_2PL_Optimizer()
        # the boundary is universal
        # the boundary is set regardless of the constrained option because the
        # constrained search serves as backup for outlier cases
        opt_worker.set_bounds([self.beta_bnds, self.alpha_bnds])
        # theta value is universal
        opt_worker.set_theta(self.theta_val)
        initial_guess_val = self.item_param_dict[eid]['ab'].flatten().tolist()

        opt_worker.set_initial_guess(initial_guess_val)
        if self.item_param_dict[eid]['c'] is None:
            c_val = 0.0
        else:
            c_val = self.item_param_dict[eid]['c'] 
        opt_worker.set_c(c_val)

        # assemble the expected data
        expected_right_count = [self.exp_response_data[l][eid][1] for l in range(self.num_theta)]
        expected_wrong_count = [self.exp_response_data[l][eid][0] for l in range(self.num_theta)]

        input_data = [expected_right_count, expected_wrong_count]
        opt_worker.load_res_data(input_data)
        # if one wishes to inspect the model input, print the input data

        est_param = opt_worker.solve_param_mix(self.is_constrained)
        item_param = np.array(est_param).reshape(2, 1)
        return item_param

    def __solve_MNLogit(self, eid):
        '''
        # Input:
        (1) initial guess value {'ab':np array[Bs, As]}
        (2) expected item count under theta l
        (3) theta value vector
        '''

        opt_worker = optimizer.Mirt_Optimizer()

        num_y = len(self.data_ref.response_map[eid].keys())

        # generate estimate data
        sim_ys = []
        sim_xs = []
        # TODO: improve efficience here since theta value are fixed
        # simulation works well in large sample but also quite inefficient
        for l in range(self.num_theta):
            for j in range(num_y):
                sim_cnt = int(self.exp_response_data[l][eid][j])  
                for t in range(sim_cnt):
                    sim_ys.append(j)
                    sim_xs.append([1.0, self.theta_val[l]])
        # estimate
        X = np.array(sim_xs)
        Y = np.array(sim_ys)
        import ipdb; ipdb.set_trace() # BREAKPOINT
        opt_worker.load_res_data(Y, X, J=num_y, K=2)
        item_param = opt_worker.solve_param(self.item_param_dict[eid]['ab'])
        return item_param


    def __calc_theta(self):
        self.theta_vec = np.dot(self.theta_density, self.theta_val) 
