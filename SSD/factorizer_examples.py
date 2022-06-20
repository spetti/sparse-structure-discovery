#!/usr/bin/env python
# coding: utf-8

import numpy as np
import matplotlib.pyplot as plt
import sys
import pickle
sys.path.insert(0, 'utils/')
from factorizer import *
import ssd


mode = sys.argv[1] 

out_location = sys.argv[2]

# random seeds for rotation
of_seeds = [0,1,2] # env rotation
fo_seeds = [3,4,5] # loci rotation


# lambda values
lamb2_range = 10**(np.linspace(np.log10(1e-3),np.log10(1.5),25))
lamb1_range = 10**(np.linspace(np.log10(1e-4),np.log10(1e-2),25))
lamb1_fixed = [1e-4]
lamb2_fixed = [1e-3]


## Common functions

# pick which k to run the methods with
def pick_k(F, thresh = .95, printout = True):
    u,s1,vh = np.linalg.svd(F)
    Ks = []
    err_SVDs =[]
    for K in range(1,F.shape[0],1):
        u_K = u[:,:K]
        s_K = s1[:K]
        vh_K = vh[:K]

        err_SVD =  np.mean((F - np.matmul(u_K,s_K[:,None]*vh_K))**2)
        if printout: print("%02d   %.3f" %(K,err_SVD))

        Ks += [K]
        err_SVDs += [err_SVD]
        
    for n, err in enumerate(err_SVDs):
        if err< 1-thresh:
            return Ks[n]
        
    return False
    
# rotate  Fs and initial factorizer objects
def init_fct_for_rotated_Fs(fcts, label, of_seeds, fo_seeds):
    E, L = fcts[(label, None, None)].FF.shape
    for ofs in of_seeds:
        fos = None
        fcts[(label, ofs, fos)] = factorizer()
        fcts[(label, ofs, fos)].subtract_means = False
        fcts[(label, ofs, fos)].rescale = False
        FF = deepcopy(fcts[(label, None, None)].FF) # this is centered, normalized WM + C after noise added
        O = ortho_group.rvs(dim=E, random_state = ofs)
        FF_rot = O@FF 
        fcts[(label, ofs, fos)].init_with_F(FF_rot)

    for fos in fo_seeds:
        ofs = None
        fcts[(label, ofs, fos)] = factorizer()
        fcts[(label, ofs, fos)].subtract_means = False
        fcts[(label, ofs, fos)].rescale = False
        FF = deepcopy(fcts[(label, None, None)].FF) # this is centered, normalized WM + C after noise added
        O = ortho_group.rvs(dim=L, random_state = fos)
        FF_rot = FF @ O
        fcts[(label, ofs, fos)].init_with_F(FF_rot)
    return fcts

# decompose Fs for the appropriate ranges 
def run_factorizer(fcts, lamb1_range, lamb2_range, lamb1_fixed, lamb2_fixed, K, svd_ks = None, printout = True, grid = True):
    for key in fcts: 
        if printout: print(f"for {key}")
        
        # svd decomposition at each k 
        if svd_ks is None: svd_ks = [K]
        for k in svd_ks:
            fcts[key].svd(K) 
        
        # run for grid of values
        if grid == True and key[-1] is None and key[-2] is None:
            fcts[key].factorize_regularized_range(K, lamb1_range, lamb2_range, verbose = printout) 
            if lamb2_fixed[0] in lamb2_range and lamb2_fixed[0] in lamb2_range: continue
                
        # run for F v. OF, env rotation test
        if key[-1] is None:
            fcts[key].factorize_regularized_range(K, lamb1_range, lamb2_fixed, verbose = printout) 
            
        # run for F v. FO, loci rotation test
        if key[-2] is None:
            fcts[key].factorize_regularized_range(K, lamb1_fixed, lamb2_range, verbose = printout) 



# # RUN FOR DATASETS

# ## Synthetic data: independent

if mode == 'syn_ind':
    E,L,K = 96,200, 6

    mw_pairs = []
    #for m in [0.2, 0.4, 0.6, 0.8]:
    #    mw_pairs.append((m, 1.0))
    #for w in [0.2, 0.4, 0.6, 0.8]:
    #    mw_pairs.append((1.0, w))
    mw_pairs.append((0.2, 1.0))
    mw_pairs.append((1.0, 0.2))
    mw_pairs.append((1.0, 1.0))
    mw_pairs.append((0.2, 0.2))

    mws = 0
    sig = .3
    noise_seed = 0

    triples = {}
    fcts = {}

    label = 'syn_ind'


    for pair in mw_pairs:
        m,w = pair
        print(f"M ~ B({m}), W ~ B({w})")
        triples[(label, m,w, mws)] = ssd.get_W_and_M_simple(E,L,K, m, w, seed = mws)

        if m in [.2,1.0] and w in [.2,1.0]:
            run_grid = True
        else:
            run_grid = False
        print(f"factorize whole grid: {run_grid}")
        
        # make factorizer object with this F
        fct = factorizer()
        fct.subtract_means = False    
        fct.init_with_MW( triples[(label,m,w,mws)][1], triples[(label,m,w,mws)][0], triples[(label,m,w,mws)][2], sig=sig, seed=noise_seed)
        print(f"ave mod / loci : {fct.true_ampl:.2f}, ave mod / env : {fct.true_ampe:.2f}")
        fcts[((label, m,w, mws), None, None)]=fct

        # intial factorizer with rotated Fs
        init_fct_for_rotated_Fs(fcts, (label, m,w, mws), of_seeds, fo_seeds)

        # pick k 
        k = pick_k(fcts[((label, m,w, mws), None, None)].FF, printout = False)
        print(f"will run method with k = {k}")

        # decompose Fs and rotated Fs
        run_factorizer(fcts, lamb1_range, lamb2_range, lamb1_fixed, lamb2_fixed, k, svd_ks = None, printout = True, grid = run_grid)

    # save results as a pickle
    pickle.dump(fcts, open(f"{out_location}/{label}", "wb"))
    pickle.dump(triples, open(f"{out_location}/triples_{label}", "wb"))


# ## Synthetic data: hub structure


if mode == "syn_hub":
    L = 200
    h = 20 # hub envs
    H = 8  # hub modules
    p = 4  # num perturbations
    rho = 1.0 #magnitude of perturbation
    E = int(h*(1+p))
    K = H + p

    mws = 0
    sig = .3
    noise_seed = 0

    triples = {}
    fcts = {}

    mw_pairs = []
    for m in [.2]:
        for w in [2]:
   # for m in [0.2, 1.0]:
    #    for w in [2,8]:
        #for w in [2, H]:
            mw_pairs.append((m, w))

    label = 'syn_hub'


    for pair in mw_pairs:
        m,w = pair
        print(f"M ~ B({m}), hub modules per hub = {w}")
        triples[(label, m,w, mws)] = ssd.get_W_and_M_structured(h, H, p, L, w, m, rho, seed = mws)

        # make factorizer object with this F
        fct = factorizer()
        fct.subtract_means = False    
        fct.init_with_MW( triples[(label,m,w,mws)][1], triples[(label,m,w,mws)][0], triples[(label,m,w,mws)][2], sig=sig, seed=noise_seed)
        print(f"ave mod / loci : {fct.true_ampl:.2f}, ave mod / env : {fct.true_ampe:.2f}")
        fcts[((label, m,w, mws), None, None)]=fct

        # intial factorizer with rotated Fs
        init_fct_for_rotated_Fs(fcts, (label, m,w, mws), of_seeds, fo_seeds)

        # pick k 
        K = pick_k(fcts[((label, m,w, mws), None, None)].FF, printout = False)
        print(f"will run method with k = {K}")

        # decompose Fs and rotated Fs
        run_factorizer(fcts, lamb1_range, lamb2_range, lamb1_fixed, lamb2_fixed, K, svd_ks = None, printout = True)

    # save results as a pickle
    pickle.dump(fcts, open(f"{out_location}/{label}", "wb"))
    pickle.dump(triples, open(f"{out_location}/triples_{label}", "wb"))


    
    
    
# ## BBQ

# In[5]:
if mode == 'bbq':

    results_dir = "../QTL/BBQ_results_6_18"
    inloc = "../QTL/BBQ_data_processed"
    cc= "0.94"
    lt1 ="0.0"
    lt2 = "0.003"
    width = "100"
    std = "2"
    norm = "l1"
    F = np.load(f"{results_dir}/second_F_cc_{cc}_lt1_{lt1}_width_{width}_std_{std}_norm_{norm}.npy")
    L = np.load(f"{results_dir}/loci_kept_after_localization_round_1_cc_{cc}_lt1_{lt1}_width_{width}_std_{std}_norm_{norm}.npy")
    causal_loci = np.where(L==True)[0]
    Xtrain = np.load(f"{inloc}/geno_train.npy") [:,causal_loci] 
    Ytrain = np.load(f"{inloc}/pheno_train.npy") 
    Xtest = np.load(f"{inloc}/geno_test.npy")[:,causal_loci] 
    Ytest = np.load(f"{inloc}/pheno_test.npy")
    
    non_zero_loci = np.where(np.sum(np.abs(F), axis=0))[0]
    Xtrain = Xtrain[:, non_zero_loci]
    Xtest = Xtest[:, non_zero_loci]
    F = F[:, non_zero_loci]
    causal_loci = causal_loci[non_zero_loci]


    
    envs=["ynb","suloc","raff","mol","27C","eth","30C","25C","sds","cu","33C","li","gu","23C","35C","mann","37C", "4NQO"]
    envs= sorted(envs)
    
    # intialize factorizer for data and rotated objects
    label = "BBQ"
    fcts = {}

    # intial factorizer with F
    fct = factorizer()
    fct.subtract_means = False
    fct.init_with_F_XY(F, Xtrain, Ytrain, Xtest, Ytest)
    fct.loci_names = causal_loci
    fct.env_names = envs
    fcts[(label, None, None)] = fct

    # intial factorizer with rotated Fs
    init_fct_for_rotated_Fs(fcts, label, of_seeds, fo_seeds)

    # pick k 
    K = pick_k(fcts[(label, None, None)].FF, printout = False)
    print(f"will run method with k = {K}")

    # decompose Fs and rotated Fs
    run_factorizer(fcts, lamb1_range, lamb2_range, lamb1_fixed, lamb2_fixed, K, svd_ks = None, printout = True)

    # save results as a pickle
    pickle.dump(fcts, open(f"{out_location}/{mode}", "wb"))
    
if mode == 'bbq8':

    results_dir = "../QTL/BBQ_results_6_18"
    inloc = "../QTL/BBQ_data_processed"
    cc= "0.94"
    lt1 ="0.0"
    lt2 = "0.003"
    width = "100"
    std = "2"
    norm = "l1"
    F = np.load(f"{results_dir}/second_F_cc_{cc}_lt1_{lt1}_width_{width}_std_{std}_norm_{norm}.npy")
    L = np.load(f"{results_dir}/loci_kept_after_localization_round_1_cc_{cc}_lt1_{lt1}_width_{width}_std_{std}_norm_{norm}.npy")
    causal_loci = np.where(L==True)[0]
    Xtrain = np.load(f"{inloc}/geno_train.npy") [:,causal_loci] 
    Ytrain = np.load(f"{inloc}/pheno_train.npy") 
    Xtest = np.load(f"{inloc}/geno_test.npy")[:,causal_loci] 
    Ytest = np.load(f"{inloc}/pheno_test.npy")
    
    non_zero_loci = np.where(np.sum(np.abs(F), axis=0))[0]
    Xtrain = Xtrain[:, non_zero_loci]
    Xtest = Xtest[:, non_zero_loci]
    F = F[:, non_zero_loci]
    causal_loci = causal_loci[non_zero_loci]


    
    envs=["ynb","suloc","raff","mol","27C","eth","30C","25C","sds","cu","33C","li","gu","23C","35C","mann","37C", "4NQO"]
    envs= sorted(envs)
    
    # intialize factorizer for data and rotated objects
    label = "BBQ"
    fcts = {}

    # intial factorizer with F
    fct = factorizer()
    fct.subtract_means = False
    fct.init_with_F_XY(F, Xtrain, Ytrain, Xtest, Ytest)
    fct.loci_names = causal_loci
    fct.env_names = envs
    fcts[(label, None, None)] = fct

    # intial factorizer with rotated Fs
    init_fct_for_rotated_Fs(fcts, label, of_seeds, fo_seeds)

    # pick k 
    K = 8
    print(f"will run method with k = {K}")

    # decompose Fs and rotated Fs
    run_factorizer(fcts, lamb1_range, lamb2_range, lamb1_fixed, lamb2_fixed, K, svd_ks = None, printout = True)

    # save results as a pickle
    pickle.dump(fcts, open(f"{out_location}/{mode}", "wb"))

# ## Kinsler

if mode == 'kinsler':
    od = pd.read_csv("data/elife-61271-fig2-data1-v2.csv", sep = ',') 

    fitness_keys = od.keys()[7::2]
    error_keys = od.keys()[8::2]

    env_names = list(fitness_keys)
    env_names = [e[:-8] for e in env_names]

    E = len(env_names)

    #filter unsequenced mutations
    ECmean_thr = 0.05
    error_thr = 0.5
    filtered = []
    for l,t in enumerate(od['type']):
        F_ECs = np.zeros(8)
        for e in range(8):
            F_ECs[e] = od[fitness_keys[e]][l]
        #print("%.3f  %.3f" %(np.mean(F_ECs)))

        F_errors = np.zeros(E)
        for e in range(E):
            F_errors[e] = od[error_keys[e]][l]
        #print("%.3f   %.3f" %(np.mean(F_errors), np.max(F_errors)))

        if t == "NotSequenced" or abs(np.mean(F_ECs)) < ECmean_thr or np.max(F_errors) > error_thr:
            filtered += [False]
        else:
            filtered += [True]

    L = np.sum(filtered)

    sorted_ = np.array(np.argsort(od['mutation_type'][filtered]),dtype=int)

    print(E,L,env_names)

    F = np.zeros((E,L))
    Ferrs = np.zeros((E,L))

    for e in range(len(fitness_keys)):
        F[e] = np.array(od[fitness_keys[e]][filtered])[sorted_]
        Ferrs[e] = np.array(od[error_keys[e]][filtered])[sorted_]

    types_filt = [o for i,o in enumerate(list(od['mutation_type'])) if filtered[i]]
    loci_types = [types_filt[i] for i in sorted_]



    # intialize factorizer for data and rotated objects
    label = "kinsler"
    fcts = {}

    # intial factorizer with F
    fct = factorizer()
    fct.subtract_means = False
    fct.init_with_F(F)
    fct.env_names = env_names
    fct.loci_names = loci_types
    fcts[(label, None, None)] = fct

    # intial factorizer with rotated Fs
    init_fct_for_rotated_Fs(fcts, label, of_seeds, fo_seeds)

    # pick k 
    K = pick_k(fcts[(label, None, None)].FF, printout = False)
    print(f"will run method with k = {K}")

    # decompose Fs and rotated Fs
    run_factorizer(fcts, lamb1_range, lamb2_range, lamb1_fixed, lamb2_fixed, K, svd_ks = None, printout = True)

    # save results as a pickle
    pickle.dump(fcts, open(f"{out_location}/{label}", "wb"))


# ## Genotoxin

if mode == "genotoxin":


    od = pd.read_csv("data/genotoxic_input.tsv", sep = '\t') # downloaded from https://figshare.com/articles/dataset/Webster_Supplemental_Output/14963561
    od = od.set_index("Treatment")
    F = od.values

    # intialize factorizer for data and rotated objects
    label = "genotoxin"
    fcts = {}

    # intial factorizer with F
    fct = factorizer()
    fct.subtract_means = False
    fct.init_with_F(F)
    fct.env_names = list(od.index)
    fct.loci_names = list(od.columns)
    fcts[(label, None, None)] = fct

    # intial factorizer with rotated Fs
    init_fct_for_rotated_Fs(fcts, label, of_seeds, fo_seeds)

    # pick k 
    K = pick_k(fcts[(label, None, None)].FF, printout = False)
    print(f"will run method with k = {K}")

    # decompose Fs and rotated Fs
    run_factorizer(fcts, lamb1_range, lamb2_range, lamb1_fixed, lamb2_fixed, K, svd_ks = None, printout = True)

    # save results as a pickle
    pickle.dump(fcts, open(f"{out_location}/{label}", "wb"))


# ## HIPHOP

if mode == "hiphop":
# In[20]:


    od = pd.read_csv("data/2019-09-10_PMID24723613_log2ratios-het.tsv", sep = '\t') # downloaded from https://www.dropbox.com/s/y8t32jt9ud1wq1n/2019-09-10_PMID24723613_log2ratios-het.tsv?dl=0
    #od = od.set_index("Treatment")
    F = od.values[:,1:]
    F = np.array(F,dtype = float).T

    #Filter rows which have NaNs
    filtered = []
    envs = []
    for e in range(F.shape[0]):
        filtered += [np.sum(F[e] != F[e]) == 0]
        if filtered[-1]:
            envs += [list(od.keys())[1+e]]


    # intialize factorizer for data and rotated objects
    label = "hiphop"
    fcts = {}

    # intial factorizer with F
    fct = factorizer()
    fct.subtract_means = False
    fct.init_with_F(F)
    fct.env_names = envs
    fct.loci_names = list(od.values[:,0])
    fcts[(label, None, None)] = fct

    # intial factorizer with rotated Fs
    init_fct_for_rotated_Fs(fcts, label, of_seeds, fo_seeds)

    # pick k 
    K = pick_k(fcts[(label, None, None)].FF, printout = False)
    print(f"will run method with k = {K}")

    # decompose Fs and rotated Fs
    run_factorizer(fcts, lamb1_range, lamb2_range, lamb1_fixed, lamb2_fixed, K, svd_ks = None, printout = True)

    # save results as a pickle
    pickle.dump(fcts, open(f"{out_location}/{label}", "wb"))

