# -*- coding: utf-8 -*-

import os
import sys
RootDir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, RootDir)
from pyirt.solver.model import IRT_MMLE_2PL

import json

# load data
data = json.load(open(RootDir + '/data/mirt_mixture_data.json'))
raw_data = data['raw']

# run model
mod = IRT_MMLE_2PL()

# load data
mod.load_data(raw_data)  # pass
mod.load_guess_param({1: 0.25})
mod.load_param(theta_bnds=[-2, 2],
               alpha_bnds=[0.25, 2],
               beta_bnds=[-2, 2])

mod.solve_EM()



