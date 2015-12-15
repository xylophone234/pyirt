import json
from scipy.stats import norm

import os
import sys
RootDir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, RootDir)

from pyirt.utl.tools import get_grade_prob
import numpy as np


# the standard test response data is a mixture of two and three component data

item_param = {0: {'A': [1.0], 'B': [1.0]},
              1: {'A': [1.0], 'B': [1.0], 'C': 0.25},
              2: {'A': [1.0, 2.0], 'B': [1.0, 2.0]}}

# theta are genreated by normal N(0,2)
N = 100
R = 10
thetas = [norm.rvs()*2 for x in range(N)]

# generate response variables
raw_data = []
complete_data = []
for i in range(N):
    theta = thetas[i]
    for j in range(R):
        eid = j % 3
        if eid != 1:
            probs = get_grade_prob(theta, item_param[eid]['A'], item_param[eid]['B']) 
        elif eid == 1:
            probs = get_grade_prob(theta, item_param[eid]['A'], item_param[eid]['B'], item_param[eid]['C']) 
        Y = np.random.choice(range(len(probs)), p=probs)
        raw_data.append((i, eid, Y))
        complete_data.append((i, eid, Y))

data = {'raw': raw_data, 'complete': complete_data}

json.dump(data, open(RootDir+'/data/mirt_mixture_data.json', 'w'))
