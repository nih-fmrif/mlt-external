import os
import pickle
from tqdm import tqdm, trange

import matplotlib
from matplotlib import pyplot

# math imports
import numpy as np
import scipy
import sklearn
from sklearn import linear_model
from sklearn.utils import extmath
from sklearn.base import BaseEstimator, TransformerMixin
from cvxopt import matrix, spmatrix, solvers, sparse, spdiag
import quadprog

import warnings

from dataclasses import dataclass


@dataclass
class MF_class:
    W: np.ndarray
    Q: np.ndarray
    C: np.ndarray
    Qc: np.ndarray
    Z: np.ndarray
    aZ: np.ndarray
    mask: np.ndarray
    nan_mask: np.ndarray
    M: np.ndarray
    M_nan: np.ndarray
    row_idx: np.ndarray
    col_idx: np.ndarray
    

class BCNMF(TransformerMixin, BaseEstimator):
    
    def __init__(
        self,
        n_components=None,
        *,
        W_beta=0.1,
        Q_beta=0.1,
        regularizer=1,
        rho=3.0,
        tau=3.0,
        W_upperbd=(False, 1.0),
        Q_upperbd=(False, 1.0),
        min_iter=10,
        max_iter=200,
        tol=1e-4,
        random_state=None,
        verbose=True):
        
        self.n_components = n_components
        self.W_beta = W_beta
        self.Q_beta = Q_beta
        self.regularizer = regularizer
        self.rho = rho
        self.tau = tau
        self.W_upperbd = W_upperbd
        self.Q_upperbd = Q_upperbd
        self.min_iter = min_iter
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = None
        self.verbose = verbose
        
        self.MF_data = None
    
    def initialize_NMF(self, X, eps=1e-6):
        U, S, V = extmath.randomized_svd(X, self.n_components, random_state=None)
        W, H = np.zeros(U.shape), np.zeros(V.shape)

        W[:, 0] = np.sqrt(S[0]) * np.abs(U[:, 0])
        H[0, :] = np.sqrt(S[0]) * np.abs(V[0, :])
        for j in range(1, self.n_components):
            x, y = U[:, j], V[j, :]
            # extract positive and negative parts of column vectors
            x_p, y_p = np.maximum(x, 0), np.maximum(y, 0)
            x_n, y_n = np.abs(np.minimum(x, 0)), np.abs(np.minimum(y, 0))
            # and their norms
            x_p_nrm, y_p_nrm = np.linalg.norm(x_p), np.linalg.norm(y_p)
            x_n_nrm, y_n_nrm = np.linalg.norm(x_n), np.linalg.norm(y_n)
            m_p, m_n = x_p_nrm * y_p_nrm, x_n_nrm * y_n_nrm
            # choose update
            if m_p > m_n:
                u = x_p / x_p_nrm
                v = y_p / y_p_nrm
                sigma = m_p
            else:
                u = x_n / x_n_nrm
                v = y_n / y_n_nrm
                sigma = m_n
            lbd = np.sqrt(S[j] * sigma)
            W[:, j] = lbd * u
            H[j, :] = lbd * v

        W[W < eps] = 0
        H[H < eps] = 0
        return W, H

    def initialize(self, M_class):

        assert np.sum(np.isnan(M_class.M)) == 0

        W, QT = self.initialize_NMF(M_class.M)
        Q = QT.T

        W[W < 0] = 0
        if self.W_upperbd[0] == True:
            W[W > self.W_upperbd[1]] = self.W_upperbd[1]

        Q[Q < 0] = 0
        if self.Q_upperbd[0] == True:
            Q[Q > self.Q_upperbd[1]] = self.Q_upperbd[1]

        Z = W@(Q.T)
        Z[Z < 0] = 0
        Z[Z > np.max(M_class.M)] = np.max(M_class.M)

        aZ = np.zeros_like(M_class.M)
        
        if M_class.cofounder is not None:
            C = np.hstack((M_class.cofounder, 1.0-M_class.cofounder))
            C = np.hstack((C, np.ones((C.shape[0], 1))))

            # initialization on Qc
            Qc = np.zeros((M_class.M.shape[1], C.shape[1]))
        else:
            C = None
            Qc = None

        M = M_class.M.copy()
        M[M < 0] = 0
        M_class.M = M

        MF_data = MF_class(W, Q,
                           C, Qc,
                           Z, aZ,
                           M_class.mask, M_class.nan_mask,
                           M_class.M, M_class.M_nan,
                           M_class.row_idx, M_class.col_idx)

        return MF_data
    
    
    
    
    def soft_shrinkage(self, X, l):
        return np.sign(X) * np.maximum(np.abs(X) - l, 0.)

    
    def _multiFISTA(self, X, eta, A, B, V, operator, lips_const, t=1.0):
        
        # row-ise FISTA algorithm
        # arg min_x eta ||x||_1 + 1/2 * f(x)
        
        f = lambda X : np.sum((X@A-B)**2, axis=1) + self.tau*np.sum((X-V)**2, axis=1)
        grad_f = lambda X : X@operator - 2*(B@A.T + self.tau*V) 
        loss_fcn = lambda X : 0.5*f(X) + eta*np.linalg.norm(X, ord=1, axis=1)
        
        loss = loss_fcn(X)
        t0 = t
        _row_idx = np.arange(X.shape[0])
        
        for iteration in range(self.max_iter):
            _X = X.copy()
            _X[_row_idx] = self.soft_shrinkage(X[_row_idx] - grad_f(X)[_row_idx]/lips_const,
                                               eta/lips_const)
            t = (1 + np.sqrt(1 + 4*t0**2))/2
            X[_row_idx,:] = _X[_row_idx,:] + ((t0 - 1)/t) * (_X[_row_idx,:] - X[_row_idx,:])
            
            new_loss = loss_fcn(X)
            _ratio = np.abs(loss - new_loss)/(np.abs(new_loss)+1e-12)
            _row_idx = np.where(_ratio > self.tol)[0]
            loss = new_loss
            
            if len(_row_idx) == 0:
                return X
        return X
    
    def _projection(self, _Y, uppbd):
            _Y = np.maximum(_Y, 0)
            if uppbd[0]:
                _Y = np.minimum(_Y, uppbd[1])
            return _Y
            
            
    def _multiregularized_loss(self, X, A, B, gamma):
        # loss function for _admm_constrained_multilasso
        mismatch_loss = 0.5 * np.sum((X@A - B)**2, axis=1)
        reg_loss = gamma * np.linalg.norm(X, ord=self.regularizer, axis=1)
        return mismatch_loss + reg_loss
    
    def _admm_constrained_multilasso(self, X, A, B, gamma, uppbd,
                                     min_iter=1, max_iter=200):
        
        # solve row-wise constrained Lasso with ADMM
        # 1/2 * || B - XA ||^2_F + gamma * sum_i || X[i,:] ||_1
        # where the L1-norm is computed row-wise
        
        # Intuition: sparsity control on every single subject/question representation
        
        # gamma = beta / rho
        
        # initialize _mu
        _mu = np.zeros_like(X)
        # initialize _Y
        _Y = self._projection(X, uppbd)
            
        loss = self._multiregularized_loss(X, A, B, gamma)
        row_idx = np.arange(X.shape[0])
        
        loss_list = [loss]

        operator = 2*(A@A.T + self.tau * np.eye(A.shape[0]))
        lipschitz_const = np.linalg.norm(operator, ord=2)
        
        for iteration in range(max_iter):
            
            _V = _Y - _mu/self.tau
            X[row_idx] = self._multiFISTA(X[row_idx],
                                           gamma,
                                           A, # for defining f and grad_f
                                           B[row_idx], # for defining f and grad_f
                                           _V[row_idx], # for defining f and grad_f
                         
                                          operator, # for defining f and grad_f
                                           lipschitz_const, # for defining f and grad_f
                                           )
            
            _Y[row_idx] = self._projection(X[row_idx] + _mu[row_idx]/self.tau, uppbd)
            _mu[row_idx] += self.tau * (X[row_idx] - _Y[row_idx])
            
            new_loss = self._multiregularized_loss(X, A, B, gamma)
            _ratio = np.abs(loss_list[-1] - new_loss)/(np.abs(new_loss)+1e-12)
            row_idx = np.where(_ratio > self.tol)[0]
            loss_list.append(new_loss)
            
            if iteration > min_iter:
                converged = True
                if len(row_idx) > 0:
                    converged = False
                if converged:
                    return X, _Y
                
        return X, _Y 
    
    def _nnQPsolver_CVXOPT(self, H, f, upperbound):
        
        solvers.options['show_progress'] = False
        
        n = H.shape[0]
        P = matrix(H)
        q = matrix(f)
        if upperbound[0] == False:
            G = matrix( -np.eye(n) )
            h = matrix( np.zeros(n) )
        elif upperbound[0] == True:
            G = matrix( np.vstack((-np.eye(n), np.eye(n))) )
            h = matrix( np.hstack((np.zeros(n), upperbound[1]*np.ones(n) )) )
        sol = solvers.qp(P,q,G,h)
        return np.array(sol['x']).flatten()
    
    def _nnQPsolver_quadprog(self, H, f, upperbound):
        
        solvers.options['show_progress'] = False
        
        n = H.shape[0]
        if upperbound[0] == False:
            qp_C = np.eye(n)
            qp_b = np.zeros(n)
        elif upperbound[0] == True:
            qp_C = np.vstack((np.eye(n), -np.eye(n))).T
            qp_b = np.hstack((np.zeros(n), -upperbound[1]*np.ones(n)))
        sol = quadprog.solve_qp(H, -f, qp_C, qp_b)[0]
        return sol

    
    def _constrained_multiquadratic(self, X, A, B, gamma, uppbd, solver='quadprog'):
        
        # solve row-wise constrained quadratic programming with ADMM
        # 1/2 * || B - XA ||^2_F + gamma * sum_i || X[i,:] ||_2
        # where the L2-norm is computed row-wise
        
        H = A@A.T + 2*gamma*np.eye(A.shape[0])
        f = -B@A.T
        
        # quadprog package cannot handle semi-positive definite matrix
        if gamma < 1e-8:
            solver = 'cvxopt'
            
        for idx in range(X.shape[0]):
            if solver == 'quadprog':
                X[idx] = self._nnQPsolver_quadprog(H, f[idx], uppbd)
            elif solver == 'cvxopt':
                X[idx] = self._nnQPsolver_CVXOPT(H, f[idx], uppbd)
            
        return X
        
                    
    def _updateZ(self, M, nan_mask, R):
        
        # update Z element-wise
        # min || mask * (M - X) ||^2_F + rho * || X - R ||^2_F
        
        Z = (M*nan_mask + self.rho*R) / (self.rho + nan_mask*1.0)
        Z[Z < 0] = 0.0
        Z = np.minimum(Z, np.max(M))
        return Z
    
    def _obj_func(self, MF_data, betaW, betaQ):
    
        R = MF_data.W @ (MF_data.Q.T)
        if MF_data.C is not None:
            R += MF_data.C @ (MF_data.Qc.T)
            
        mismatch_loss = 0.5 * np.sum(MF_data.nan_mask * (MF_data.M - R)**2)
        W_norm = np.sum(np.linalg.norm(MF_data.W, ord=self.regularizer, axis=1))
        Q_norm = np.sum(np.linalg.norm(MF_data.Q, ord=self.regularizer, axis=1))
        reg_loss = W_norm+Q_norm
        return mismatch_loss, reg_loss
    
    
    def fit_transform(self, M_class):
        
        # initialization on W, Q
        MF_data = self.initialize(M_class)
        MF_data, loss_history = self._fit_transform(MF_data, update_Q=True)
        
        return MF_data, loss_history 

    def transform(self, M_class):
        
        assert self.MF_data is not None
        
        MF_data = self.MF_data.copy()
        MF_data.C = M_class.C
        
        MF_data.mask = M_class.mask
        MF_data.nan_mask = M_class.nan_mask
        MF_data.M = M_class.M
        MF_data.M_nan = M_class.M_nan
        MF_data.row_idx = M_class.row_idx
        MF_data.col_idx = M_class.col_idx
        
        MF_data, loss_history = self._fit_transform(MF_data, update_Q=False)
        return MF_data, loss_history
    
    
    def _fit_transform(self, MF_data, update_Q=True, save_MF=True):
        
        # initial distance value
        loss_history = []

        # tqdm setting
        tqdm_iterator = trange(self.max_iter, desc='Loss', leave=True, disable=not self.verbose)

        betaW = self.W_beta
        betaQ = self.Q_beta * MF_data.W.shape[0]/MF_data.Q.shape[0]

        # Main iteration
        for i in tqdm_iterator:
            
            # subproblem 1
            B = MF_data.Z + MF_data.aZ/self.rho
            if MF_data.C is not None:
                B -= MF_data.C@(MF_data.Qc.T)
                
            gamma = betaW/self.rho
            if self.regularizer == 1:
                _, MF_data.W = self._admm_constrained_multilasso(MF_data.W,
                                                                      MF_data.Q.T,
                                                                      B,
                                                                      gamma,
                                                                      uppbd=self.W_upperbd)
            elif self.regularizer == 2:
                MF_data.W = self._constrained_multiquadratic(MF_data.W,
                                                             MF_data.Q.T,
                                                             B,
                                                             gamma,
                                                             uppbd=self.W_upperbd)
            else:
                raise ValueError("Unknown regularizer.")

            # subproblem 2
            if MF_data.C is not None:
                QComp = np.column_stack((MF_data.Q, MF_data.Qc))
                WComp = np.column_stack((MF_data.W, MF_data.C))
            else:
                QComp = MF_data.Q
                WComp = MF_data.W
                
                
            gamma = betaQ/self.rho
            if self.regularizer == 1:
                _, QComp = self._admm_constrained_multilasso(QComp, WComp.T, B.T, 
                                                             gamma, uppbd=self.Q_upperbd)
            elif self.regularizer == 2:
                QComp = self._constrained_multiquadratic(QComp, WComp.T, B.T,
                                                         gamma, uppbd=self.Q_upperbd)
            else:
                raise ValueError("Unknown regularizer.")
                                      
            if update_Q:
                MF_data.Q = QComp[:, :MF_data.Q.shape[1]]
            if MF_data.C is not None:
                MF_data.Qc = QComp[:, MF_data.Q.shape[1]:]

            # subproblem 3
            B = MF_data.W@(MF_data.Q.T)
            
            if MF_data.C is not None:
                B += MF_data.C@(MF_data.Qc.T)
                
            MF_data.Z = self._updateZ(MF_data.M,
                                      MF_data.nan_mask,
                                    B-MF_data.aZ/self.rho)


            # auxiliary varibles update        
            MF_data.aZ += self.rho*(MF_data.Z - B)
            
            # Iteration info
            mismatch_loss, reg_loss = self._obj_func(MF_data, betaW, betaQ)
            loss_history.append((mismatch_loss+reg_loss, mismatch_loss, reg_loss))
            tqdm_iterator.set_description("Loss: {:.4}".format(loss_history[-1][0]))
            tqdm_iterator.refresh()

            # Check convergence
            if i > self.min_iter:
                converged = True
                if np.abs(loss_history[-1][0]-loss_history[-2][0])/np.abs(loss_history[-1][0]) < self.tol:
                    if self.verbose:
                        print('Algorithm converged with relative error < {}.'.format(self.tol))
                    return MF_data, loss_history
                else:
                    converged = False
                    
        if save_MF:
            self.MF_data = MF_data

        return MF_data, loss_history
        
