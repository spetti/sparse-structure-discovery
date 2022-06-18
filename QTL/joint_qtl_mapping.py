import numpy as np
from copy import deepcopy
import time
import pickle
import os
import sys
import argparse

sys.path.insert(0, 'utils/')
import call_glmnet as mlg
import localize as lc
from sklearn.metrics import r2_score

# input data
parser = argparse.ArgumentParser(description='Run GLMNET and localization')
parser.add_argument("-gt", "--genotype_train", dest = "GT_file", default = None, help="path to training genotype npy file", type = str, required=True)
parser.add_argument("-pt", "--phenotype_train", dest = "PT_file", default = None, help="path to training phenotype npy file", type = str, required=True)
parser.add_argument("-gv", "--genotype_val", dest = "GV_file", default = None, help="path to validation genotype npy file", type = str, required=True)
parser.add_argument("-pv", "--phenotype_val", dest = "PV_file", default = None, help="path to validation phenotype npy file", type = str, required=True)

# input either list of loci or correlations
parser.add_argument("-c", "--correlations", dest = "C_file", default = None, help="path to correlation coeff npy file", type = str, required=False)
parser.add_argument("-l", "--loci_list", dest = "L_file", default = None, help="npy file with length number of loci in genotypes; true/false indicates whether to use loci", type = str, required=False)

# location for results and intermediate outputs 
parser.add_argument("-output", "--output_prefix", dest = "output_prefix", default = None, help="where to save results and intermediate steps", type = str)

# load F matrix and predictions when starting pipeline at localization steps 1 or 2
parser.add_argument("-load_F", "--load_F", dest = "F_in", default = None, help="load F; for use when starting pipeline after first or second run of GLMNET", type = str)
parser.add_argument("-load_P", "--load_preds", dest = "Preds_in", default = None, help="load predictions; for use when starting pipeline after first or second run of GLMNET (must correspond to same training set)", type = str)

# options with defaults
parser.add_argument("-ct", "--correlation_threshold", dest = "ct", default = .99, help="include loci with maximum pearson correlation c", type = float)
parser.add_argument("-lt1", "-round_1_localization_threshold", dest = "lt1", default = 0.0, help="only localize loci with magnitude at least lt1 for some phenotype", type = float)
parser.add_argument("-lt2", "--round_2_localization_threshold", dest = "lt2", default = .003, help="only localize loci with magnitude at least lt2 for some phenotype", type = float)
parser.add_argument("-norm", "--norm_to_rank_loci", dest = "norm", default = "l1", help="use norm l1 or l2 to rank loci to localize", type = str)
parser.add_argument("-v", action='store_true', help="verbose", default = False)
parser.add_argument("-po", action='store_true', help="compute number of loci that remain with cc threshold",  default = False)
parser.add_argument("-w", "--width", dest = "width", default = 50, help="max width of confidence intervals", type = int)
parser.add_argument("-std", "--std", dest = "std", default = 2, help="std of tolerance for confidence interval computations", type = int)

# options for running only part of the pipeline
parser.add_argument("-fo", action='store_true', help="only run first round of glmnet",  default = False)
parser.add_argument("-sl1", action='store_true', help="start pipeline with first localization step; must use -load_F to feed in F matrix and -l to feed in list of loci",  default = False)
parser.add_argument("-sl1o", action='store_true', help="run first localization step only; must use -load_F to feed in F matrix ",  default = False)
parser.add_argument("-sl2o", action='store_true', help="only perform second localization step; must use -load_F to feed in F matrix ",  default = False)

args = parser.parse_args()

# check that either input a list of loci or correlations between all loci
if args.C_file is None and args.L_file is None:
    raise ValueError("need to input correlations or list of loci; must use one of -c or -l")
if args.C_file is not None and args.L_file is not None:
    raise ValueError("input either correlations or list of loci; either -c or -l")    
do_corr = False
if args.C_file is not None: do_corr = True
    
    
# check that input F is given when pipeline starts at a localization set
first_run_glmnet_needed = True
if args.sl1 + args.sl2o + args.sl1o>1:
    raise ValueError("only can use one of sl1, sl1o, sl2o")
if args.sl1 or args.sl2o or args.sl1o:
    first_run_glmnet_needed = False
    if args.F_in is None:
        raise ValueError("need to specify an input F matrix using -load_F")
    if args.L_file is None:
        raise ValueError("need to specify a list of loci with -l")

# load in data
GT = np.load(args.GT_file) 
if args.v: print("loaded training genotypes")
PT = np.load(args.PT_file) 
if args.v: print("loaded training phenotypes")
GV = np.load(args.GV_file) 
if args.v: print("loaded validation genotypes")
PV = np.load(args.PV_file) 
if args.v: print("loaded validation phenotypes")
    
if do_corr:
    C = np.load(args.C_file) 
    if args.v: print("loaded correlations")
else:
    loci_to_keep = np.load(args.L_file) 
    if args.v: print("loaded list of loci to include")

if PT.shape[0]!=GT.shape[0]:
    raise ValueError(f"training phenotype and genotype dimensions don't match: {PT.shape} {GT.shape}")

if PV.shape[0]!=GV.shape[0]:
    raise ValueError(f"validation phenotype and genotype dimensions don't match: {PT.shape} {GT.shape}")
    
if GT.shape[1]!=GV.shape[1]:
    raise ValueError(f"training and validation don't have same number of loci: {GT.shape[1]} {GV.shape[1]}")
    
#remove correlated loci
if do_corr:
    if args.v: print(f"filtering out loci correlated > {args.ct}")

    loci_to_keep = [False for _ in range(GT.shape[1])]
    loci_to_keep[0] = True
    last = 0
    while last < GT.shape[1]:
        for i in range(0,C.shape[1]):
            found_next = False
            if C[last,i]< args.ct:
                if last+i+1 < len(loci_to_keep):
                    loci_to_keep[last+i+1] = True
                    last = last+i+1
                    found_next = True
                break
        if found_next == False:
            if last+i+2> len(loci_to_keep):
                break
            else:
                loci_to_keep[last+i+2] = True
                last = last+i+2


    if args.v or args.po: print(f"for correlation threshold {args.ct}, after greedy algorithm {sum(loci_to_keep)} loci remain")
    np.save(open(args.output_prefix+f"/loci_kept_cc_{args.ct}.npy", 'wb'), np.array(loci_to_keep))
    if args.po: exit()
    
L = np.arange(0,GT.shape[1])
Xtrain = np.copy(GT)[:,loci_to_keep]
Ytrain = np.copy(PT)
Xval = np.copy(GV)[:,loci_to_keep]
Yval = np.copy(PV)
for A in [Xtrain,Ytrain, Xval, Yval]:
    if np.isnan(A).any():
        raise ValueError("inputs cannot have NaN")
    
if args.v:
    print(f"training genotype shape: {Xtrain.shape}")
    print(f"training phenotype shape: {Ytrain.shape}")
    print(f"validation genotype shape: {Xval.shape}")
    print(f"validation phenotype shape: {Yval.shape}")
    
# run glmnet or load results

if first_run_glmnet_needed:
    if args.v: print("running glmnet round 1")
    F,preds = mlg.get_coefs(Xtrain,Ytrain,Xval,Yval)
    np.save(open(args.output_prefix+f"/first_F_cc_{args.ct}.npy", 'wb'), F)
    np.save(open(args.output_prefix+f"/first_preds_cc_{args.ct}.npy", 'wb'), preds)
    if args.v: print(f"first glmnet run finished and results saved at {args.output_prefix}/first_F_cc_{args.ct}.npy")
    if args.fo: exit()
else:
    F = np.load(args.F_in) 
    preds = np.load(args.Preds_in)
    if args.v: print(f"loaded F")

        
# run first round of localization
if args.sl2o==False:
    computed_preds = Xtrain @ F.T
    affine_term = (preds-computed_preds)[0,...]
    if args.v: print("running localization round 1")
    idx_filt, loci_filt, F_ori, X, Y, Ysub, loci = lc.prep_arrays(F, np.copy(GT), np.copy(PT), loci_to_keep, args.lt1, args.v, affine_term, norm = args.norm)
    expanded_list = lc.localize_and_split(idx_filt, loci_filt, F_ori, X, Y, Ysub, loci, args.width, args.std, args.v)
    loci_to_keep = [False for _ in range(GT.shape[1])]
    for _ in expanded_list:
        try:
            loci_to_keep[_] = True
        except:
            print(_, len(loci_to_keep), GT.shape[1])
            raise ValueError("index issue found")
            
    if args.v: 
        print(f"num of loci identified: {sum(loci_to_keep)}")
    print("test")
    np.save(open(args.output_prefix+f"/loci_kept_after_localization_round_1_cc_{args.ct}_lt1_{args.lt1}_width_{args.width}_std_{args.std}_norm_{args.norm}.npy", 'wb'), loci_to_keep)

        
# run glmnet with loci selected in first round of localization
if args.sl2o == False and args.sl1o== False:
    Xtrain = np.copy(GT)[:,loci_to_keep]
    Ytrain = np.copy(PT)
    Xval = np.copy(GV)[:,loci_to_keep]
    Yval = np.copy(PV)
    if args.v: print("running glmnet round 2")
    F, preds = mlg.get_coefs(Xtrain,Ytrain,Xval,Yval)
    print("R2 time!")
    for _ in range(Ytrain.shape[1]):
        print(r2_score(Ytrain[:,_], preds[:,_]))
    np.save(open(args.output_prefix+f"/second_F_cc_{args.ct}_lt1_{args.lt1}_width_{args.width}_std_{args.std}_norm_{args.norm}.npy", 'wb'), F)
    np.save(open(args.output_prefix+f"/second_preds_cc_{args.ct}_lt1_{args.lt1}_width_{args.width}_std_{args.std}_norm_{args.norm}.npy", 'wb'), preds)
    if args.v: print(f"second glmnet run finished and results saved at {args.output_prefix}/second_F_cc_{args.ct}_lt1_{args.lt1}_width_{args.width}_std_{args.std}.npy")
    np.save(open(args.output_prefix+"/ytrain.npy","wb"), Ytrain)
    np.save(open(args.output_prefix+"/F.npy","wb"),F)
    np.save(open(args.output_prefix+"/preds","wb"),preds)
    exit()

# run second round of localization
if args.v: print("running localization round 2")
print(F.shape)
print(args.lt2)
computed_preds = Xtrain @ F.T
affine_term = (preds-computed_preds)[0,...]
idx_filt, loci_filt, F_ori, X, Y, Ysub, loci = lc.prep_arrays(F,np.copy(GT), np.copy(PT), loci_to_keep, args.lt2, args.v, affine_term, norm = args.norm)
loci_localized, all_confidence_intervals = lc.compute_interval_intersections(idx_filt, loci_filt, F_ori, X, Y, Ysub, loci, args.width, args.std, args.lt2, args.v)
pickle.dump((loci_localized, all_confidence_intervals), open(args.output_prefix+f"/CI_after_localization_round_2_cc_{args.ct}_lt1_{args.lt1}_lt2_{args.lt2}_width_{args.width}_std_{args.std}_norm_{args.norm}.pickle", 'wb'))

    
    


