'''
This script deals with the data format problem.

The stardard format for pyirt is ( uid,eid,result),
where uid is the idx for test taker, eid is the idx for items

It is set in this way to deal with the sparsity in the massive dataset.

'''
import numpy as np
import time
import os
import subprocess


import collections as cos

def load_tuples(data):
    '''
    # Input: python tuple (uid, eid, grade)
    # Only int is allowed in the environment
    '''
    uids = []
    eids = []
    grades = []
    if len(data) == 0:
        raise ValueError('Input is empty.')

    for log in data:
        uids.append(int(log[0]))
        eids.append(int(log[1]))
        grades.append(int(log[2]))

    return uids, eids, grades

def load_file_handle(file_path, sep=','):
    '''
    # Input file fields: uid, eid, grade. Default comma separated.
    # Only int is allowed in the environment
    '''
    uids = []
    eids = []
    grades = []

    with open(file_path) as f:
        for line in f:
            if line == '':
                continue
            uidstr, eidstr, gradestr = line.strip().split(sep)
            uids.append(int(uidstr))
            eids.append(int(eidstr))
            grades.append(int(gradestr))
    return uids, eids, grades


def map_response2grade(eids, responses):
    '''
    Map item's response to grade category
    Because multinomial is essentially rank order utility, reorder the data as 0-J
    
    Depends on init_item_user_map
    '''
    #TODO: deal with perfect prediction
    
    # reduce 
    unique_item_response_combo = set(zip(eids, responses))
    item2response = cos.defaultdict(list)
    for item_response_pair in unique_item_response_combo:
        item2response[item_response_pair[0]].append(item_response_pair[1])

    # map
    response2grade = {}
    grade2response = {}
    for eid, responses in item2response.iteritems():
        response2grade[eid] = dict((response, j) for j, response in enumerate(sorted(responses)))
        grade2response[eid] = dict((j, response) for j, response in enumerate(sorted(responses)))

    return response2grade,grade2response


def map_ids(ids):
    # map unique ids to 0-N dict
    unique_ids = list(set(ids))
    n = len(unique_ids)
    id_map = dict(zip(unique_ids, xrange(n)))
    id_reverse_map = dict(zip(xrange(n),unique_ids))

    return id_map, id_reverse_map


class data_storage(object):
    '''
    map user ids to 0-N idx (uids)
    map item ids to 0-M idx (eids)
    map response of each item to 0-J idx(grade)

    In relationship map, x2y means project x to cell of y. 
    For example, user2item = list(list(uid)), the outer list is indexed by eid
    The list index is valid after the item map.
    '''

    def __init__(self, user_ids, item_ids, responses):

        # check input
        self.num_log = len(user_ids)
        if len(item_ids) != self.num_log or len(responses) != self.num_log:
            raise ValueError('Input data are not the same length')

        # map user and item ids
        self.user_map, self.user_reverse_map = map_ids(user_ids)
        self.item_map, self.item_reverse_map = map_ids(item_ids)
        # translate
        uids = [self.user_map[x] for x in user_ids]
        eids = [self.item_map[y] for y in item_ids]
        # generate auxilary parameters, later used in _map_item_user
        self.num_user = len(self.user_map.keys())
        self.num_item = len(self.item_map.keys())
       
        # map responses
        self.response_map, self.response_reverse_map = map_response2grade(eids, responses)
        # translate
        grades = [self.response_map[eids[i]][responses[i]] for i in xrange(self.num_log)]

        # map user2item and item2user
        self._map_item_user(uids, eids, grades)

        # map user2grade*item 
        self._map_user2grade_item()


    def _map_item_user(self, uids, eids, grades):
        num_log = self.num_log
        num_user = self.num_user
        num_item = self.num_item

        # initialize
        self.item2user = [[] for x in xrange(num_user)]
        self.user2item = [[] for y in xrange(num_item)]

        for i in xrange(num_log):
            eid = eids[i]; uid = uids[i]; grade = grades[i]
            self.item2user[uid].append((eid, grade))
            self.user2item[eid].append((uid, grade))


    def _map_user2grade_item(self):
        self.user2grade_item = cos.defaultdict(dict)
        # initialize
        for eid, response_dict in self.response_map.iteritems():
            for j in response_dict.itervalues():
                self.user2grade_item[eid][j] = []

        # fill
        for eid in range(len(self.user2item)):
            for log in self.user2item[eid]:
                uid = log[0]; grade = log[1]
                self.user2grade_item[eid][grade].append(uid) 
        

'''
Legacy Code
'''
def from_matrix_to_list(indata_file, sep=',', header=False, is_uid=False):
    # assume the data takes the following format
    # (uid,) item1, item2, item3
    # (1,)   0,     1,     1
    is_skip = True
    is_init = False
    uid = 0

    result_list = []

    with open(indata_file, 'r') as f:
        for line in f:
            if is_skip and header:
                # if there is a header, skip
                is_skip = False
                continue
            segs = line.strip().split(sep)

            if len(segs) == 0:
                continue

            if not is_init:
                # calibrate item id
                if is_uid:
                    num_item = len(segs) - 1
                else:
                    num_item = len(segs)

            # parse
            for j in xrange(num_item):
                if is_uid:
                    idx = j + 1
                else:
                    idx = j
                result_list.append((uid, j, int(segs[idx])))

            # TODO: the current code is hard wired to uid starts from 0 to n
            # needs to remove the dependency
            uid += 1

    return result_list

def parse_item_paramer(item_param_dict, output_file=None):

    if output_file is not None:
        # open the file
        out_fh = open(output_file, 'w')

    sorted_eids = sorted(item_param_dict.keys())

    for eid in sorted_eids:
        param = item_param_dict[eid]
        alpha_val = np.round(param['alpha'], decimals=2)
        beta_val = np.round(param['beta'], decimals=2)
        if output_file is None:
            print eid, alpha_val, beta_val
        else:
            out_fh.write('{},{},{}\n'.format(eid, alpha_val, beta_val))
