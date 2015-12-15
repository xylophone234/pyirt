# -*- coding: utf-8 -*-

import os
import sys
RootDir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, RootDir)


import unittest
from pyirt.utl import loader

class TestMapResponse2Grade(unittest.TestCase):
    def setUp(self):
        eids = [1,1,1,2,2,2,2,2]
        full_resp = [1,0,0,0,1,3,2,1]
        self.full_data = {'eids':eids, 'responses':full_resp}
        
        gap_resp = [1,3,5,4,3,1,5,9]
        self.gap_data = {'eids':eids, 'responses':gap_resp}
        

    def test_full_response(self):
        res2grade,_ = loader.map_response2grade(self.full_data['eids'], 
                                               self.full_data['responses'])
        true_res2grade = {1:{0:0,1:1}, 2:{0:0,1:1,2:2,3:3}}
        self.assertDictEqual(res2grade,true_res2grade)

    def test_gap_response(self):
        res2grade, grade2res = loader.map_response2grade(self.gap_data['eids'], 
                                               self.gap_data['responses'])
        true_res2grade = {1:{1:0,3:1,5:2}, 2:{1:0,3:1,4:2,5:3,9:4}}
        true_grade2res = {1:{0:1,1:3,2:5}, 2:{0:1,1:3,2:4,3:5,4:9}}

        self.assertDictEqual(res2grade,true_res2grade)
        self.assertDictEqual(grade2res,true_grade2res)


class TestMapIds(unittest.TestCase):
    def setUp(self):
        self.str_ids = ['a','d','b','d']
        self.int_ids = [1,5,3,5]

    def test_map_str(self):
        id_map, id_reverse_map = loader.map_ids(self.str_ids)
        self.assertDictEqual(id_map, {'a':0,'b':1,'d':2})
        self.assertDictEqual(id_reverse_map, {0:'a',1:'b',2:'d'})

    def test_map_int(self):
        id_map, id_reverse_map = loader.map_ids(self.int_ids)
        self.assertDictEqual(id_map, {1:0,3:1,5:2})
        self.assertDictEqual(id_reverse_map, {0:1,1:3,2:5})

class TestDataStorage(unittest.TestCase):
    def setUp(self):
        # item a has three respones
        # 2: user 1
        # 4: user 2
        # 6: user 1, user 3
        # item b has two responses
        # Y: user 1
        # N: user 2
        user_ids = [1,3,1,5,3,1]
        item_ids = ['a','a','a','a','b','b']
        responses = [2,4,6,6,'N','Y']
        self.test_data = loader.data_storage(user_ids, item_ids, responses)

    def test_map(self):
        self.assertDictEqual(self.test_data.user_map, {1:0,3:1,5:2})
        self.assertDictEqual(self.test_data.item_map, {'a':0,'b':1})
        self.assertDictEqual(self.test_data.response_map, {0:{2:0,4:1,6:2},1:{'N':0,'Y':1}})

    def relation_map(self):
        self.assertDictEqual(self.test_data.user2item, {0:[0,1,2],1:[0,1]})
        self.assertDictEqual(self.test_data.item2user, {0:[0,1],1:[0,1],2:[0]})
        self.assertDictEqual(self.test_data.user2grade_item, {0:{0:[0],1:[1],2:[0,2]},1:{0:[1],1:[0]}})

        


    
       


if __name__ == '__main__':
    unittest.main()
