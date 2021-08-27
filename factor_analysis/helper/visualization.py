import os
import numpy as np
import pandas as pd
import glob
import pickle
import seaborn as sns
import time
from tqdm import tqdm
from tqdm import trange
import json
import re

import matplotlib
from matplotlib import pyplot
from matplotlib.pyplot import *
from IPython.display import set_matplotlib_formats
set_matplotlib_formats('png', 'pdf')

from numpy.linalg import norm
import scipy
import scipy.io as sio
from scipy.spatial.distance import cosine
from scipy.optimize import linear_sum_assignment

import sklearn
from sklearn.model_selection import KFold
from sklearn.cluster import SpectralClustering, AgglomerativeClustering


from textwrap import wrap
from scipy import stats


def bestmatch(A, B, distance='euclidean', colwise=True):
    
    if A.shape[1] > B.shape[1]:
        permute_mat = A
        ref_mat = B
    else:
        permute_mat = B
        ref_mat = A
    
    if colwise == False:
        ref_mat = ref_mat.T
        permute_mat = permute_mat.T
        
    k = np.minimum(ref_mat.shape[1], permute_mat.shape[1])
    
    cost = np.zeros((ref_mat.shape[1], permute_mat.shape[1]))
    for i in range(ref_mat.shape[1]):
        for j in range(permute_mat.shape[1]):
#             if distance == 'euclidean':
#                 dist = scipy.spatial.distance.euclidean(ref_mat[:,i], permute_mat[:,j])
#                 cost[i,j] = dist
#             if distance == 'peasonr':
#                 pcoef, _ = scipy.stats.spearmanr(ref_mat[:,i], permute_mat[:,j])
#                 cost[i,j] = 1 - pcoef
#             if distance == 'spearman':
#                 scoef, _ = scipy.stats.spearmanr(ref_mat[:,i], permute_mat[:,j])
#                 cost[i,j] = 1 - scoef
#             if distance == 'kendall':
#                 kcoef, _ = scipy.stats.kendalltau(ref_mat[:,i], permute_mat[:,j])
#                 cost[i,j] = 1 - kcoef
#             if distance == 'weightedtau':
#                 kcoef, _ = scipy.stats.weightedtau(ref_mat[:,i], permute_mat[:,j])
#                 cost[i,j] = 1 - kcoef
            if distance == 'euclidean':
                dist = scipy.spatial.distance.euclidean(ref_mat[:,i], permute_mat[:,j])
                cost[i,j] = dist
            if distance == 'peasonr':
                pcoef1, _ = scipy.stats.spearmanr(ref_mat[:,i], permute_mat[:,j])
                pcoef2, _ = scipy.stats.spearmanr(ref_mat[:,i], -permute_mat[:,j])
                if pcoef1 >= pcoef2:
                    cost[i,j] = 1 - pcoef1
                else:
                    cost[i,j] = 1 - pcoef2
            if distance == 'spearman':
                scoef1, _ = scipy.stats.spearmanr(ref_mat[:,i], permute_mat[:,j])
                scoef2, _ = scipy.stats.spearmanr(ref_mat[:,i], -permute_mat[:,j])
                if scoef1 >= scoef2:
                    cost[i,j] = 1 - scoef1
                else:
                    cost[i,j] = 1 - scoef2
            if distance == 'kendall':
                kcoef1, _ = scipy.stats.kendalltau(ref_mat[:,i], permute_mat[:,j])
                kcoef2, _ = scipy.stats.kendalltau(ref_mat[:,i], -permute_mat[:,j])
                if kcoef1 >= kcoef2:
                    cost[i,j] = 1 - kcoef1
                else:
                    cost[i,j] = 1 - kcoef2
            if distance == 'weightedtau':
                kcoef1, _ = scipy.stats.weightedtau(ref_mat[:,i], permute_mat[:,j])
                kcoef2, _ = scipy.stats.weightedtau(ref_mat[:,i], -permute_mat[:,j])
                if kcoef1 >= kcoef2:
                    cost[i,j] = 1 - kcoef1
                else:
                    cost[i,j] = 1 - kcoef2
                
                
    row_ind, col_ind = scipy.optimize.linear_sum_assignment(cost)
    
    reg_mat = np.zeros((permute_mat.shape[0], k))
    for (i, ind) in enumerate(col_ind[:k]):
        reg_mat[:,i] = permute_mat[:,ind]
    
    if colwise == False:
        reg_mat = reg_mat.T
        
#     compute_stat(ref_mat, reg_mat)
    
    return reg_mat, ref_mat, col_ind[:k]


def compute_stat(Q1, Q2):
    assert Q1.shape == Q2.shape
    k = Q1.shape[1]
    
    SC = np.zeros(k)
    SCpval = np.zeros(k)
    for i in range(k):
        SC1, pval1 = stats.spearmanr(Q1[:,i], Q2[:,i])
        SC2, pval2 = stats.spearmanr(Q1[:,i], -Q2[:,i])
        if SC1 >= SC2:
            SC[i] = SC1
            SCpval[i] = pval1
        else:
            SC[i] = SC2
            SCpval[i] = pval2

    print('Spearman correlation: ', ['%.2f' % s for s in SC])
    print('mean = {:.2f}, std = {:.2f}'.format(np.mean(SC), np.std(SC)))
    print('p-value: ', ['%.1e' % p for p in SCpval])

    KT = np.zeros(k)
    KTpval = np.zeros(k)
    for i in range(k):
        KT1, pval1 = stats.kendalltau(Q1[:,i], Q2[:,i])
        KT2, pval2 = stats.kendalltau(Q1[:,i], -Q2[:,i])
        if KT1 >= KT2:
            KT[i] = KT1
            KTpval[i] = pval1
        else:
            KT[i] = KT2
            KTpval[i] = pval2

    print('Kendall correlation: ', ['%.2f' % s for s in KT])
    print('mean = {:.2f}, std = {:.2f}'.format(np.mean(KT), np.std(KT)))
    print('p-value: ', ['%.1e' % p for p in KTpval])
    
    WT = np.zeros(k)
    WTpval = np.zeros(k)
    for i in range(k):
        WT1, _ = stats.weightedtau(Q1[:,i], Q2[:,i])
        WT2, _ = stats.weightedtau(Q1[:,i], -Q2[:,i])
        if WT1 >= WT2:
            WT[i] = WT1
        else:
            WT[i] = WT2

    print('Weightedtau: ', ['%.2f' % s for s in WT])
    print('mean = {:.2f}, std = {:.2f}'.format(np.mean(WT), np.std(WT)))
    
    return SC, SCpval, KT, KTpval, WT, WTpval


def assign_solver(ref_mat, permute_mat, distance='euclidean', colwise=True):
    if colwise == False:
        ref_mat = ref_mat.T
        permute_mat = permute_mat.T
    
    cost = np.zeros((permute_mat.shape[1], ref_mat.shape[1]))
    for i in range(permute_mat.shape[1]):
        for j in range(ref_mat.shape[1]):
            if distance == 'euclidean':
                dist = scipy.spatial.distance.euclidean(permute_mat[:,i], ref_mat[:,j])
                cost[i,j] = dist
            if distance == 'peasonr':
                pcoef1, _ = scipy.stats.spearmanr(permute_mat[:,i], ref_mat[:,j])
                pcoef2, _ = scipy.stats.spearmanr(permute_mat[:,i], -ref_mat[:,j])
                if pcoef1 >= pcoef2:
                    cost[i,j] = 1 - pcoef1
                else:
                    cost[i,j] = 1 - pcoef2
            if distance == 'spearman':
                scoef1, _ = scipy.stats.spearmanr(permute_mat[:,i], ref_mat[:,j])
                scoef2, _ = scipy.stats.spearmanr(permute_mat[:,i], -ref_mat[:,j])
                if scoef1 >= scoef2:
                    cost[i,j] = 1 - scoef1
                else:
                    cost[i,j] = 1 - scoef2
            if distance == 'kendall':
                kcoef1, _ = scipy.stats.kendalltau(permute_mat[:,i], ref_mat[:,j])
                kcoef2, _ = scipy.stats.kendalltau(permute_mat[:,i], -ref_mat[:,j])
                if kcoef1 >= kcoef2:
                    cost[i,j] = 1 - kcoef1
                else:
                    cost[i,j] = 1 - kcoef2
            if distance == 'weightedtau':
                kcoef1, _ = scipy.stats.weightedtau(permute_mat[:,i], ref_mat[:,j])
                kcoef2, _ = scipy.stats.weightedtau(permute_mat[:,i], -ref_mat[:,j])
                if kcoef1 >= kcoef2:
                    cost[i,j] = 1 - kcoef1
                else:
                    cost[i,j] = 1 - kcoef2
                
                
    row_ind, col_ind = linear_sum_assignment(cost)
    
    reg_mat = np.zeros_like(permute_mat)
    for (i, k) in enumerate(col_ind):
        reg_mat[:,k] = permute_mat[:,i]
    
    if colwise == False:
        reg_mat = reg_mat.T
    
    return reg_mat, col_ind


def plot_Q(Q, C, CBCL_question, dataname, model, Ctitle=None, refQ=None, show_all=False):
    nQ = Q.shape[0]
    k = Q.shape[1]
    if C is not None:
        nC = C.shape[1]
    else:
        nC = 0
    
    quesSort = np.argsort(CBCL_question['group'])
    quesInfo = CBCL_question.iloc[quesSort]
    
    classes = np.unique(CBCL_question['CBCL2001_6-18_scale'].values)
    
    jump = np.where(np.asarray([int(i.split('-')[0]) for i in quesInfo['group'].values[1:]])-
                    np.asarray([int(i.split('-')[0]) for i in quesInfo['group'].values[:-1]]) > 0)[0]
    center = np.append(jump,nQ)
    diff = (center[1:] - center[:-1])/2
    center = center - np.append(center[0]/2, diff)
    
    if refQ is not None:
        subQ, orderIdx = assign_solver(refQ[:,:k-nC], Q[:,:k-nC], distance='euclidean', colwise=True)
        Q[:,:k-nC] = subQ
        
    if show_all and C is None:
        fig, ax = pyplot.subplots(figsize=(18,0.3*k))
    #     sns.heatmap(Q.T, cmap='Blues', ax=ax, cbar=True, cbar_kws = dict(pad=0.01))
        sns.heatmap(Q[quesSort,:].T, cmap='Blues', ax=ax, cbar=False)
        ax.set_xticks(np.linspace(0+0.5, nQ-0.5, nQ))
        ax.set_xticklabels(quesInfo['CBCL2001_6-18_varname'].values, fontsize=7, rotation=90)
        for (i, loc) in enumerate(jump):
            ax.vlines(loc+1,0-0.5,k+0.5)
        for (i, loc) in enumerate(center):
            ax.text(loc, -0.5, classes[i], ha='center')
        bottom, top = ax.get_ylim()
        ax.set_ylim(bottom + 0.5, top - 0.5)
        pyplot.show()

    #     fig.savefig("./visualization/"+dataname+"-"+model+"-QC.pdf", bbox_inches='tight')

    if C is not None:
        fig, ax = pyplot.subplots(figsize=(18,0.3*(k-nC)))
        sns.heatmap(Q[quesSort,:k-nC].T, cmap='Blues', ax=ax, cbar=False)
        ax.set_xticks(np.linspace(0+0.5, nQ-0.5, nQ))
        ax.set_xticklabels(quesInfo['CBCL2001_6-18_varname'].values, fontsize=7, rotation=90)

    #     ax.set_yticks(np.arange(k-dim)+0.5)
    #     ax.set_yticklabels(['Dim {}'.format(i+1) for i in range(k-dim)], rotation=0)
        for (i, loc) in enumerate(jump):
            ax.vlines(loc+1,0-0.5,Q.shape[1]+0.5)
        for (i, loc) in enumerate(center):
            ax.text(loc, -0.5, classes[i], ha='center')

        bottom, top = ax.get_ylim()
        ax.set_ylim(bottom + 0.5, top - 0.5)
        pyplot.show()

#     fig.savefig("./visualization/"+dataname+"-"+model+"-Q.pdf", bbox_inches='tight')

    if show_all:
        if nC > 0:
            fig, ax = pyplot.subplots(figsize=(18,0.3*(nC)))
    #         sns.heatmap(Q[:,k-dim:].T, cmap='Blues', ax=ax, cbar=True, cbar_kws = dict(pad=0.01))
            sns.heatmap(Q[quesSort,k-nC:].T, cmap='Blues', ax=ax, cbar=False)
            ax.set_xticks(np.linspace(0+0.5, nQ-0.5, nQ))
            ax.set_xticklabels(quesInfo['CBCL2001_6-18_varname'].values, fontsize=7, rotation=90)

            ax.set_yticks(np.arange(nC)+0.5)
            if Ctitle is not None:
                ax.set_yticklabels(Ctitle, rotation=0)
            else:
                ax.set_yticklabels(['C-{}'.format(i+1) for i in range(nC)], rotation=0)
            for (i, loc) in enumerate(jump):
                ax.vlines(loc+1,0-0.5,Q.shape[1]+0.5)
            for (i, loc) in enumerate(center):
                ax.text(loc, -0.5, classes[i], ha='center')

            bottom, top = ax.get_ylim()
            ax.set_ylim(bottom + 0.5, top - 0.5)
            pyplot.show()
        
#         fig.savefig("./visualization/"+dataname+"-"+model+"-C.pdf", bbox_inches='tight')
        
    if refQ is not None:
        return orderIdx
    else:
        return np.arange(k-nC)
    
    
    
def structure_plot(diagnosis, method, CBCL_question, beta, features, save_dir,
                   nfactor=None, quantile=0.95, max_factors=6, max_questions=6, 
                   show_weight=False, blind=False):
    
    ranks = np.argsort(np.abs(beta))[::-1]
    if nfactor == None:
        beta_median = np.quantile(beta,0.5)
        nfactor = np.where(beta >= beta_median)[0].shape[0]
        nfactor = np.maximum(2, np.minimum(nfactor, max_factors))

    from graphviz import Digraph
    
    if show_weight:
        dot = Digraph(comment='Factor analysis',
                      graph_attr={"nodesep": "0.1", "dpi":"300"},
                      edge_attr={"arrowhead": "none"},
                      ) # graph_attr={"splines": "ortho"}
    else:
        dot = Digraph(comment='Factor analysis',
              graph_attr={"nodesep": "0.1", "dpi":"300", "splines": "line"},
              edge_attr={"arrowhead": "none"})
    
    if blind:
        dot.attr(rankdir='LR', labelloc="t", fontsize="22")
    else:
        dot.attr(rankdir='LR', label=diagnosis + " : " + method, labelloc="t", fontsize="22")
    pal = sns.color_palette("pastel")
    colors = pal.as_hex()
    
    if nfactor <= 10:
        pal = sns.color_palette("tab10")
        edgecolors = pal.as_hex()[::-1]
    else:
        edgecolors = ['k' for i in range(nfactor)]
        
    with dot.subgraph() as s:
        s.attr('node', shape='doublecircle', style='filled', color='lightgrey')
        s.node('DG', '\n'.join(wrap(diagnosis.replace('_',' '), 20)), fontsize='20')
        s.attr('node', shape='ellipse')
        for k in range(nfactor):
            s.node('F{}'.format(k+1), 'Factor {}'.format(ranks[k]+1), fontsize='18')
#             s.edge('F{}'.format(k+1), 'DG')
    
    question_list = []
    beta_list = []
    with dot.subgraph() as s:
        s.attr('node', shape='box', width = '8', height = '.3')
        for i in range(nfactor):
            main_factors = features[:,ranks[i]]
            s.edge('F{}'.format(i+1), 'DG', label='b = {:.2f}'.format(beta[ranks[i]]), penwidth='3', fontsize='20')
            
#             question_ranks = np.argsort(np.abs(main_factors))[::-1]
#             q = np.quantile(np.abs(main_factors),quantile)
#             nquestions = np.where(np.abs(main_factors) > q)[0].shape[0]
#             nquestions = np.minimum(nquestions, max_questions)


            question_ranks = np.argsort(main_factors)[::-1]
            q = np.quantile(main_factors,quantile)
            nquestions = np.where(main_factors > q)[0].shape[0]
            nquestions = np.minimum(nquestions, max_questions)
            
            for j in range(nquestions):
                index = question_ranks[j]
                question_label = CBCL_question['CBCL2001_6-18_varname'].iloc[index]
                question_class = CBCL_question['CBCL2001_6-18_scale'].iloc[index]
                question_classnum = int(CBCL_question['group'].iloc[index].split('-')[0])
                question = CBCL_question['CBCL2001_6-18_item_text'].iloc[index]
                if not question_label in question_list:
                    question_list.append(question_label)
                    s.node(question_label, '['+question_class+']'+': '+question+'\l', fontsize='18',
                           style='filled', color='black', fillcolor=colors[question_classnum])
                if show_weight:
                    s.edge(question_label+':e', 'F{}'.format(i+1),
                           label='w = {:.2f}'.format(main_factors[index]),
                           penwidth='2',
                           color=edgecolors[i])
                else:
                    s.edge(question_label+':e', 'F{}'.format(i+1),
                       penwidth='2',
                       color=edgecolors[i])

    
#     dot.view()
    dot.format = 'png'
    filename = os.path.join(save_dir, 'ABCD2001' + '-' + diagnosis + '-' + method)
    dot.render(filename, view=False)
    
