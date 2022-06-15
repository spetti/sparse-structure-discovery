import numpy as np
import rpy2.robjects as ro
import rpy2.robjects.numpy2ri
from scipy.stats import norm
import math
import crossgen as cg
from scipy.special import comb
import sys
import os

from rpy2.robjects.packages import importr

def initialize_r():
    rpy2.robjects.numpy2ri.activate()
    glmnet=importr('glmnet')

def to_r(mat):
    nr,nc = mat.shape
    mat_r = ro.r.matrix(mat, nrow=nr, ncol=nc)
    return mat_r

def run_glmnet(tG,tP,vG,vP):
    # Defining the R script and loading the instance in Python
    r = ro.r
    r['source']('utils/glmnet_fns.R')# Loading the function we have defined in R.
    glmnet_coeffs_r = ro.globalenv['glmnet_coeffs']

    #convert to R matrics
    tG_r = to_r(tG)
    tP_r = to_r(tP)
    vG_r = to_r(vG)
    vP_r = to_r(vP)

    #run glmnet
    out=glmnet_coeffs_r(tG_r,tP_r,vG_r,vP_r)
    the_coeffs=np.array(out[0])
    t_preds=np.array(out[1])
    
    
    return the_coeffs, t_preds

def run_glmnet_coeffs_all(tG,tP,vG,vP):
    # Defining the R script and loading the instance in Python
    r = ro.r
    r['source']('utils/glmnet_fns.R')# Loading the function we have defined in R.
    glmnet_coeffs_all_r = ro.globalenv['glmnet_coeffs_all']

    #convert to R matrics
    tG_r = to_r(tG)
    tP_r = to_r(tP)
    vG_r = to_r(vG)
    vP_r = to_r(vP)

    #run glmnet
    out=glmnet_coeffs_all_r(tG_r,tP_r,vG_r,vP_r)
    the_coeffs=np.array(out[0])
    t_preds=np.array(out[1])
    lambdas = np.array(out[2])
    
    return the_coeffs, t_preds,lambdas

def mae(a,b):
    return np.mean(np.abs(a-b))

def prune_loci(ccthresh, thresh, o_preds, coeffs, loci_raw_pos, tG, tP, cc,verbose = True):
    
    def consider_merge(loci_1,loci_2, o_preds, coeffs, tG, tP):
    
        G_at_loci=tG[:,[loci_1, loci_2]]
        coeff_at_loci=coeffs[[loci_1,loci_2],:]

        #prediction if we removed both loci
        preds_wo_loci=o_preds-G_at_loci@coeff_at_loci
        sum_coeff=np.sum(coeff_at_loci, axis=0)
        G_at_l1=G_at_loci[:,0]
        G_at_l2=G_at_loci[:,1]

        #predictions if we put all the weight on just one loci
        preds_l1=preds_wo_loci+np.einsum('i,j->ij',G_at_l1,sum_coeff)
        preds_l2=preds_wo_loci+np.einsum('i,j->ij',G_at_l2,sum_coeff)
        return mae(o_preds,tP), mae(preds_l1,tP), mae(preds_l2,tP)
    
    def decide(loci_1,loci_2, o_preds, coeffs, tG, tP, thresh):
        o_mae, l1_mae, l2_mae=consider_merge(loci_1,loci_2, o_preds, coeffs, tG, tP)
        merged=min(l1_mae,l2_mae)
        
        #merge
        if verbose:
            print(f"error ratio: {merged/o_mae}")
        if merged/o_mae< thresh:
            if verbose: print("merge")
            if merged==l1_mae: return [loci_1]
            elif merged==l2_mae: return [loci_2]
            else: return False
            
        #don't merge
        else:
            if verbose: print("don't merge")
            return [loci_1,loci_2]

    #locations of loci (wrt remaining loci) with non-zero coeffs
    non_zero_loci=np.where(np.sum(np.abs(coeffs), axis=-1)!=0)[0]
    
    #locations of loci (wrt orginal genotype array) with non-zero coeffs
    non_zero_loci_locs=np.array(loci_raw_pos)[non_zero_loci]

    cc_gaps = []
    pos = non_zero_loci_locs[0]
    for _ in range(1,non_zero_loci_locs.shape[0]):
        next_pos = non_zero_loci_locs[_]
        if next_pos-pos-1< cc.shape[1]:
            cc_gaps.append(cc[pos,next_pos-pos-1])
        else:
            cc_gaps.append(0.0)
        pos = next_pos
    
    pos=0
    to_keep=[]
    to_keep_raw_pos=[]
    while pos<len(cc_gaps):
        if cc_gaps[pos]<ccthresh:
            to_keep.append(non_zero_loci[pos])
            to_keep_raw_pos.append(loci_raw_pos[non_zero_loci[pos]])
            pos+=1
        else:
            loci_1=non_zero_loci[pos]
            loci_2=non_zero_loci[pos+1]
            if verbose:
                print(f"\nconsidering loci 1: {loci_1}, loci 2: {loci_2}, cc: {cc_gaps[pos]}")
            keepers=decide(loci_1,loci_2, o_preds, coeffs, tG, tP, thresh)
            to_keep+=keepers
            to_keep_raw_pos+=list(loci_raw_pos[keepers])
            pos+=2
            
    if pos==len(cc_gaps):
        to_keep.append(non_zero_loci[pos])
        to_keep_raw_pos.append(loci_raw_pos[non_zero_loci[pos]])
    if verbose:
        print(f"started with {len(non_zero_loci)} loci, keeping {len(to_keep)}")
    return to_keep, np.array(to_keep_raw_pos)


def single_elim(thresh, o_preds, coeffs, loci_raw_pos, tG, tP):
     
    def decide_one(loci_1, o_preds, coeffs, tG, tP, thresh):
    
        G_at_loci=tG[:,loci_1]
        coeff_at_loci=coeffs[loci_1,:]

        #prediction if we removed loci
        preds_wo_loci=o_preds-np.einsum('i,j->ij',G_at_loci,coeff_at_loci)
        without=mae(preds_wo_loci,tP)
        
        if without/o_mae < thresh:
            return []
        else:
            return[loci_1]
    to_keep_raw_pos=[]
    to_keep=[]
    non_zero_loci=np.where(np.sum(np.abs(coeffs), axis=-1)!=0)[0]
    o_mae=mae(o_preds,tP)
    for loci in non_zero_loci:
        keepers=decide_one(loci,o_preds, coeffs, tG, tP, thresh)
        to_keep+=keepers
        to_keep_raw_pos+=list(loci_raw_pos[keepers])

    return to_keep, np.array(to_keep_raw_pos)

def backward_elim(ccthreshs, threshs, first_coeffs, first_o_preds, tG,tP,vG,vP, cc, single_elim_thresh, verbose=True):
    
    initialize_r()
    
    #eliminate nearby similar positions
    for i in range(len(ccthreshs)+1):
        if verbose: print(f"\niter {i}")
        if i==0:
            if verbose:
                print("\nalready ran glmnet")
            new_coeffs, new_o_preds=first_coeffs, first_o_preds
            o_mae = mae(new_o_preds, tP)
            to_keep_raw_pos=np.array(range(tG.shape[1]))
            new_tG=tG
            new_vG=vG
            mae_now=o_mae
            if verbose:
                print(f"mae: {mae_now}, mae/o_mae: {o_mae/o_mae} num loci input: {len(to_keep_raw_pos)}")
        else:
            if verbose:
                print("\nrunning glmnet...")
            new_coeffs, new_o_preds = run_glmnet(new_tG,tP,new_vG,vP)
            mae_now=mae(new_o_preds, tP)
            if verbose:
                print(f"mae: {mae_now}, mae/o_mae: {mae_now/o_mae} num loci input: {len(to_keep)}")
        if i==len(ccthreshs): break

        if verbose: print("pruning...")
        to_keep, to_keep_raw_pos=prune_loci(ccthreshs[i],threshs[i], new_o_preds, new_coeffs, to_keep_raw_pos, new_tG, tP, cc, verbose=verbose)
        if verbose: print(to_keep_raw_pos)
        new_tG=new_tG[:,to_keep]
        new_vG=new_vG[:,to_keep]
        
    if single_elim_thresh!=False:
        if verbose:
            print("\nsingle elimination")
        to_keep, to_keep_raw_pos=single_elim(single_elim_thresh, new_o_preds, new_coeffs, to_keep_raw_pos, new_tG, tP)
        if verbose:
            print(f"keeping {len(to_keep)} loci")
            print(to_keep_raw_pos)
        new_tG=new_tG[:,to_keep]
        new_vG=new_vG[:,to_keep]
        if verbose: print("\nrunning glmnet...")
        new_coeffs, new_o_preds = run_glmnet(new_tG,tP,new_vG,vP)
        mae_now=mae(new_o_preds, tP)
        if verbose: print(f"mae: {mae_now}, mae/o_mae: {mae_now/o_mae} num loci input: {len(to_keep)}")
        
    return new_coeffs, new_o_preds, to_keep_raw_pos

#X is n_samples x n_features, Y is n_samples x n_tasks.  
#Output should coeffs with n_tasks x n_features
def get_coefs(Xtrain,Ytrain, Xval, Yval, n_alphas = 50, verbose=True): #either specify number of alphas, n_alphas, or the alphas matrix. 
    initialize_r()
    first_coeffs, first_o_preds = run_glmnet(Xtrain,Ytrain,Xval,Yval)
    return first_coeffs.T,first_o_preds

#block
def blockPrint():
    sys.stdout = open(os.devnull, 'w')

# Restore
def enablePrint():
    sys.stdout = sys.__stdout__










