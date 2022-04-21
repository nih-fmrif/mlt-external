import os
import numpy as np
import time
import argparse
import pickle

import scipy
import sklearn

from cvxopt import matrix, solvers
from numpy.linalg import norm
from datetime import datetime

import sys
sys.path.append('../')

from MF_model import MF_model


parser = argparse.ArgumentParser()
parser.add_argument('-f', type=str, help="matrix file with key 'P' and 'data_mask' ", required=True)
parser.add_argument('-d', type=int, help="embedding dimension", required=True)
parser.add_argument('-s', type=float, help="regullarization parameter", required=True)
parser.add_argument('-r', type=int, help="repeat ID", required=True)
parser.add_argument('-j', type=int, help="j-th CV ID", required=True)

parser.add_argument('-t' type=int, default=2, choices={1, 2}, help="regularization type: L1 or L2")
args = parser.parse_args()


datafile = args.f

data = np.load(datafile, allow_pickle=True)
P = data['P']
data_mask = data['data_mask']
C = data['C']
if C.ndim == 0:
    C = None

dim = args.d
sparsity = args.s
r = args.r
j = args.j

regularizer = args.t

embed_start = time.time()

mask_data = np.load("./masks/masks-{}-{}.npz".format(r,j))
mask_train = mask_data['mask_train']
mask_valid = mask_data['mask_valid']

now = datetime.now()
dt_string = now.strftime("%d%m%Y-%H%M%S")
filename = "cv-d{}-s{}-r{}-j{}-".format(dim, sparsity, r, j)+dt_string

model = MF_model(P, data_mask, dim, method='admm', C=C,
                  Wbound=(True, 1.0), regularizer=regularizer,
                  sparsity_parameter=sparsity, max_iter=500)

output = model.embed_holdout(mask_train, mask_valid)
embed_end = time.time()
embed_time = embed_end-embed_start


output_start = time.time()
np.save( "./results/"+filename+".npy", output)
output_end = time.time() 
output_time = output_end-output_start
print('Embed time: {}, Output time: {}'.format(embed_time, output_time))


