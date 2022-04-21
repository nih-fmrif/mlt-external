import numpy as np
import pandas as pd

import sys
import os
sys.path.append( os.path.abspath(os.path.join('./')) )

from sklearn.decomposition import NMF
from sklearn.impute import SimpleImputer
from _admm_MF import BCNMF
from _muNMF import muNMF

import seaborn as sns
import time
from tqdm import tqdm

from matplotlib import pyplot
from matplotlib.pyplot import *

from kneed import KneeLocator

from dataclasses import dataclass

@dataclass
class Mi_class:
    row_idx: np.ndarray
    col_idx: np.ndarray
    mask: np.ndarray
    nan_mask: np.ndarray
    M: np.ndarray
    M_nan: np.ndarray
    dataname: str
    cofounder: np.ndarray

class MF_model:
    
    def __init__(self, data_matrix, data_mask, dimension,
                 ppmi_preprocess=False,
                 method='generic',
                 regularizer=2,
                 reg_parameter=0.1,
                 Wbound=(False, 1.0),
                 Qbound=(False, 1.0),
                 max_iter=200,
                 C=None):
        
        # Algorithm to decomposition data matrix M into M = W Q'
        
        # ===input===
        # data_matrix : the data matrix
        
        # data_mask : binary matrix with same shape as the data matrix, 0=data not avaialble, 1=data available
        #             if all data entries are available, simply set data_mask = np.ones(data_matrix)
        
        # dimension : dimension of the representation
        # ppmi_preprocess : whether to normalize the data matrix by positive pointwise mutual information
        # method : two algorithms are available, which are 'generic' and 'admm'. Only 'admm' works for the Wbound and Qbound
        # reg_parameter : hyperparameter to control regularization strength. Zero corresponds to no regularization
        # Wbound : tuple for setting an upper bound of the entries in W. E.g. (True, 3.0) represents constraining entries
        #          in W to be within [0.0, 3.0]
        # Qbound : similar to Wbound
        
        # ===usage===
        # the function 'decomposition' wraps the NMF algorithm.
        # In this version :
        # - the regularization parameters are set to be equivalent for both W and Q.
        #
        # The 'BIC_dimension' is a simple function estimating the dimension of the representation using BIC.
        # For large data matrix, it is time consuming.
        # The input 'search_range' is an integer array including all possible dimensions for testing.
        # E.g. setting 'search_range = np.arange(3, 20)' will search for the optimal dimension between 3 and 20.
        # This is a simple alternative to cross-validation approach, which is more costly
        
        # ===demonstration===
        # See 'demo.py' for details
        
        self.data_matrix = data_matrix
        self.data_mask = data_mask
        
        self.ppmi_preprocess = ppmi_preprocess
        self.dimension = dimension
        self.method = method
        self.regularizer = regularizer
        self.reg_parameter = reg_parameter
        self.Wbound = Wbound
        self.Qbound = Qbound
        self.max_iter = max_iter
        
        self.C = C
        
        if self.ppmi_preprocess:
            self.data_matrix = np.nan_to_num(self.data_matrix)
            self.ppmi()
        else:
            self.P = self.data_matrix.copy()
        
        self.Pimputed = None
        
    def output_datafile(self):
        
        np.savez('./datafile.npz', P=self.data_matrix, data_mask=self.data_mask)
        
    def ppmi(self):
        
        marginal_row = self.data_matrix.sum(axis=1)
        marginal_col = self.data_matrix.sum(axis=0)
        total = marginal_col.sum()
        expected = np.outer(marginal_row, marginal_col) / total
        P = self.data_matrix / expected
        with np.errstate(divide='ignore'):
            np.log(P, out=P)
        P.clip(0.0, out=P)
        
        self.P = P
        
    def decomposition(self, verbose=True):
            
        if self.method == 'generic':
            
            if self.C is not None:
                print('cofounders are not support in generic algorithm')
            
            if ~np.all(self.data_mask == 1):
                Pnan = self.P.copy()
                Pnan[~self.data_mask.astype('bool')] = np.nan
                imp = SimpleImputer(missing_values=np.nan, strategy='mean')
                self.Pimputed = imp.fit_transform(Pnan)
                
            else:
                self.Pimputed = self.P.copy()
                
            if self.regularizer == 1:
                model = NMF(n_components=self.dimension, init='random', random_state=0, 
                            alpha_W=self.reg_parameter, 
                            alpha_H='same',
                            l1_ratio=1.0,
                            max_iter=self.max_iter)
            elif self.regularizer == 2:
                model = NMF(n_components=self.dimension, init='random', random_state=0, 
                        alpha_W=self.reg_parameter, 
                        alpha_H='same',
                        l1_ratio=0.0,
                        max_iter=self.max_iter)
            else:
                raise ValueError("Unknown regularizer.")

            W = model.fit_transform(self.Pimputed)
            Q = model.components_.T
                
        elif self.method == 'muNMF':    
            
            # Only L1 regularizer is included
            W, Q = muNMF(self.P,
                         self.dimension,
                         self.reg_parameter,
                         self.data_mask,
                         max_iter=self.max_iter)

            
        elif self.method == 'admm':
            
            Pnan = self.P.copy()
            Pnan[~self.data_mask.astype('bool')] = np.nan
            
            M_data = Mi_class(np.arange(self.P.shape[0]),
                              np.arange(self.P.shape[1]),
                              np.ones_like(self.P),
                              self.data_mask,
                              self.P,
                              Pnan,
                              'matrix',
                              self.C)
            
            clf = BCNMF(self.dimension,
                        rho=3.0,
                        tau=3.0,
                        regularizer=self.regularizer,
                        W_upperbd=self.Wbound,
                        Q_upperbd=self.Qbound,
                        W_beta=self.reg_parameter,
                        Q_beta=self.reg_parameter,
                        max_iter=self.max_iter)
            MF_data, loss_list = clf.fit_transform(M_data)
            W = MF_data.W
            Q = MF_data.Q

        else:
            
            print('Method not implemented')
            W = self.P.copy()
            Q = self.P.copy().T
        
        return W, Q
            
    
    def BIC_dimension(self, search_range=np.arange(3, 20), plot=True, update=True):
        
        BIC = []
        for k in search_range:
            self.dimension = k
            W, Q = self.decomposition()
            
            mismatch_loss = 2 * 0.5 * np.linalg.norm(self.data_mask*(self.P - W@(Q.T)), ord='fro') ** 2
            freedom_loss = np.log(self.P.shape[0])*(self.P.shape[1]*(k+1) - k*(k-1)/2)
           
            BIC.append( mismatch_loss + freedom_loss )
            
        kn = KneeLocator(search_range, BIC, curve='convex', direction='decreasing')
        
        if plot:
            fig, ax = pyplot.subplots(figsize=(16,3))
            ax.plot(search_range, BIC,  marker='o')
            ax.set_title('BIC, dimension detected: {}'.format(kn.knee))
            pyplot.show()
            
        if update:
            # self.dimension = search_range[np.argmin(BIC)]
            self.dimension = kn.knee
            
        W, Q = self.decomposition()
            
        # return search_range[np.argmin(BIC)], W, Q
        return kn.knee, W, Q
    
    
    def obj_func(self, mask, W, Q):
        return 0.5 * np.sum(mask*(self.P - np.matmul(W, Q.T)) ** 2)
    
        
    def embed_holdout(self, mask_train, mask_valid):
        start = time.time()

        mask_train *= self.data_mask
        mask_valid *= self.data_mask
        
        self.data_mask = mask_train

        W, Q = self.decomposition()

        train_error = self.obj_func(mask_train, W, Q)
        valid_error = self.obj_func(mask_valid, W, Q)

        embedding_stat = [self.dimension, self.reg_parameter, train_error, valid_error]
        end = time.time()
        return embedding_stat
    

                
                
                
        
            
            