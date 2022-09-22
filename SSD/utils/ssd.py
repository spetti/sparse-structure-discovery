import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.stats import bernoulli
from scipy.stats import ortho_group  # Requires version 0.18 of scipy
from copy import deepcopy
from matplotlib import rc
from sklearn.metrics import r2_score
from sklearn.linear_model import Lasso,Ridge, LinearRegression,orthogonal_mp,Lars
from scipy.linalg import orth
from scipy.interpolate import interp1d
from scipy import interpolate
from sklearn.decomposition import SparsePCA

from sklearn.cluster import KMeans

def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

#################### Factorization Functions ######################

#norm = 'envs', 'loci', 'mods_W', 'mods_M'
#method_M = 'Lasso', 'Lars', 'Ridge', 'OMP'
#method_W = 'Lasso', 'Ridge'
#mask = None or a Boolean matrix with the shape of F specifying which values are missing (True = present, False = missing)



def run_factorizer_regularized_range(F, K, lamb1s, lamb2s , method_M = 'Lasso', method_W = 'Lasso', norm = 'envs', verbose=True, mask = None, \
M_lb=1e-3, W_lb = 1e-3,  fit_intercept = False): #add "all" option by default
    #SVD
    u,s,vh = np.linalg.svd(F - fit_intercept*np.mean(F,axis=0))
    u_K = u[:,:K]
    s_K = s[:K]
    vh_K = vh[:K]

    to_return = None
    if verbose:
        print("K    lamb1    lamb2   error_ours  error_svd   Keff   av.proc./loci   av.proc./env   loci dropped    grassmann to svd")
    our_errs = []
    mods = []
    ndls = []
    Ws = np.zeros((len(lamb1s),len(lamb2s),F.shape[0],K))
    Ms = np.zeros((len(lamb1s),len(lamb2s),K,F.shape[1]))
    bs = np.zeros((len(lamb1s),len(lamb2s),F.shape[1]))
    for il1, lamb1 in enumerate(lamb1s):
        for il2,lamb2 in enumerate(lamb2s):
            if fit_intercept:
                b = np.mean(F,axis=0)
            else:
                b = np.zeros(F.shape[1])

            if norm == 'loci' or norm == 'mods_M':
                M = np.zeros((Ms.shape[2],Ms.shape[3]))
                M = deepcopy(vh_K)
                W = np.matmul(F - np.mean(F,axis=0),M.T)
            else:
                W = np.zeros((Ws.shape[2],Ws.shape[3]))
                W = deepcopy(u_K)
                M  = np.matmul(W.T,F - np.mean(F,axis=0))

            
            for i in range(5):
                if i%1==0:
                    err_EM = np.mean((F - np.matmul(W,M) - b)**2)
                    err_SVD =  np.mean((F - np.matmul(u_K,s_K[:,None]*vh_K)  - fit_intercept*np.mean(F,axis=0))**2)

                W,M,b = optimize(F,W,M,lamb1,lamb2, method_M = method_M, method_W = method_W, norm = norm, mask = mask, fit_intercept = fit_intercept, b=b)
            # truncate
            W = (np.abs(W)> W_lb)*W
            M = (np.abs(M)> M_lb)*M

            if fit_intercept == False:
                b = np.zeros(F.shape[1])

            Ws[il1,il2] = W
            Ms[il1,il2] = M
            bs[il1,il2] = b
            
            #compute L2 error
            err_EM = np.mean((F - np.matmul(W,M) - b)**2)
            err_SVD =  np.mean((F - np.matmul(u_K,s_K[:,None]*vh_K) - fit_intercept*np.mean(F,axis=0))**2)
            grass = grassmann(M,vh_K)

            #compute modularity
            ndl = dropped_loci(M)
            nde = dropped_envs(W)
            mod = ave_mod_per_loci(M)
            mode = ave_mod_per_env(W)
            
            mods.append(mod)
            ndls.append(ndl)
            our_errs.append(err_EM)
           
            if verbose:
                if norm == 'loci' or norm == 'mods_M': keff = int(np.ceil(np.sum(M**2)))
                else: keff = int(np.ceil(np.sum(W**2)))
                print(f"{K:2d}   {lamb1:.4f}    {lamb2:.4f}     {err_EM:.3f}       {err_SVD:.3f}     {keff:2d}      {mod:.3f}          {mode:.3f}         {ndl:3d}             {grass:.3f}")   
            
    return Ws, Ms, bs, mods, ndls



def optimize(F,W,M,lamb1,lamb2, method_M = 'Lasso', method_W = 'Lasso', norm = 'envs', mask = None, fit_intercept = False, b = None, niters = 5): 
    if mask is None:
        mask = np.ones(F.shape,dtype = bool)

    if fit_intercept == True and b is None:
        raise ValueError("need b")

    for i in range(niters):
        M,b = optimize_M(F,W,lamb2,method = method_M, norm = norm, mask = mask, fit_intercept = fit_intercept)
        W = optimize_W(F,M,lamb1,method = method_W, norm = norm, mask = mask, fit_intercept = fit_intercept, b = b)
        
    return W,M,b


def optimize_M(F,W,lamb2, method = 'Lasso', norm = None, mask = None, fit_intercept = False): #norm = 'loci' or norm = 'mods_M' or norm = None
    if mask is None:
        mask = np.ones(F.shape,dtype = bool)
    M  = np.zeros((W.shape[1],F.shape[1]))

    M2_1= np.sum(np.sum(M**2,axis=1) > 1e-6)

    b = np.mean(F,axis=0)
    for i in range(F.shape[1]):
        if method == 'Ridge':
            clf = Ridge(alpha = lamb2, fit_intercept  = fit_intercept)
            clf.fit(W[mask[:,i]],F[mask[:,i],i])
            M[:,i] = clf.coef_
            b[i] = clf.intercept_
        elif method == 'Lasso':
            clf = Lasso(alpha = lamb2, fit_intercept  = fit_intercept)
            clf.fit(W[mask[:,i]],F[mask[:,i],i])
            M[:,i] = clf.coef_
            b[i] = clf.intercept_
        elif method == 'Lars':
            clf = Lars(n_nonzero_coefs=lamb2, fit_intercept  = fit_intercept)
            clf.fit(W[mask[:,i]],F[mask[:,i],i])
            M[:,i] = clf.coef_
            b[i] = clf.intercept_
        elif method == 'OMP':
            M[:,i] = orthogonal_mp(W[mask[:,i]],F[mask[:,i],i],n_nonzero_coefs = lamb2)
        else:
            clf = Ridge(alpha = 1e-5, fit_intercept  = fit_intercept)
            clf.fit(W[mask[:,i]],F[mask[:,i],i])
            M[:,i] = clf.coef_
            b[i] = clf.intercept_

    M2_2 = np.sum(np.sum(M**2,axis=1) > 1e-6)

    if norm == 'loci':
        M = M/(1e-9+np.sqrt(np.sum(M**2,axis=1))[:,None])
    elif norm == 'mods_M':
        M = M/(1e-9+np.sqrt(np.sum(M**2,axis=0))[None,:])

    M2_3 = np.sum(np.sum(M**2,axis=1) > 1e-6)

    return M,b

def optimize_W(F,M,lamb1, method = 'Lasso', norm = 'envs', mask = None, fit_intercept = False, b = None): #norm = 'envs' or 'mods_W' or 'None'
    if mask is None:
        mask = np.ones(F.shape,dtype = bool)

    if fit_intercept == True and b is None:
        raise ValueError("need b")

    if fit_intercept == False:
        b = np.zeros(F.shape[1])

    K = M.shape[0]
    W = np.zeros((F.shape[0],M.shape[0]))
    for e in range(F.shape[0]):
        if method == 'Lasso':
            clf = Lasso(alpha = lamb1, fit_intercept  = False)
            clf.fit(M[:,mask[e,:]].T,F[e,mask[e,:]] - fit_intercept*b[mask[e,:]])
            W[e,:] = clf.coef_
        else:
            clf = Ridge(alpha = lamb1, fit_intercept  = False)
            clf.fit(M[:,mask[e,:]].T,F[e,mask[e,:]] - fit_intercept*b[mask[e,:]])
            W[e,:] = clf.coef_
            
    if norm == 'envs':
        W = W/(1e-9+np.sqrt(np.sum(W**2,axis=0))[None,:]) #normalize across environments
    elif norm == 'mods_W':
        W = W/(1e-9+np.sqrt(np.sum(W**2,axis=1))[:,None]) #normalize across modules
        
    return W



def ptk(l,r = 4): #parameter to key
    return round(l,r)


def cluster_kmeans(M,Kp, rs = 1 ):
    norms = np.sqrt(np.sum(M**2,axis=0))
    filt = norms > 1e-6
    Mfilt = deepcopy(M[:,filt])
    M_norm = Mfilt#/norms[filt]
    
    kmeans = KMeans(n_clusters=Kp, random_state=rs).fit(M_norm.T)
    D = kmeans.cluster_centers_.T
    P = np.zeros((Kp,M_norm.shape[1]))
    for i in range(M_norm.shape[1]):
        P[kmeans.labels_[i],i] = 1#norms[i]
        
    P_final = np.zeros((Kp,M.shape[1]))
    labels_final = np.zeros(M.shape[1],dtype= int) - 1
    
    inds= np.arange(M.shape[1])[filt]
    P_final[:,inds] = deepcopy(P)
    labels_final[inds] = kmeans.labels_
        
    return D,P_final, labels_final

def optimize_TDP(F,M,W,b,Kp, lamb1, lamb2, niter = 8, verbose = False):
    Mhat = deepcopy(M)
    bhat = deepcopy(b)
    That = deepcopy(W)
    K = W.shape[1]
    Ferr_M = np.mean((F - W@M - b)**2)
    Ts = np.zeros((niter,W.shape[0],K))
    Ds = np.zeros((niter,K,Kp))
    Ps = np.zeros((niter,Kp,M.shape[1]))
    errs =  np.zeros(niter)
    for i in range(niter):
        #get D,P
        T = deepcopy(That)
        D,P,l = cluster_kmeans(Mhat,Kp, rs = i)
        Mhat = D@P
        Merr = np.mean((M - Mhat)**2)/(np.std(M)**2)
        Ferr_Mhat = np.mean((F - That@Mhat - bhat)**2)
        
        #get TD
        #TDhat = ((F-bhat)@np.linalg.pinv(P))
        TDhat = That@D
        
        Ferr_TDhat = np.mean((F - TDhat@P - bhat)**2)
        
        #decompose TD
        # u,s,vh = np.linalg.svd(TDhat)
        # u_K = u[:,:K]
        # s_K = s[:K]
        # vh_K = vh[:K]
        # D = deepcopy(vh_K)
        # T = np.matmul(TDhat,D.T)
        # T,D,_ = optimize(TDhat,T,D,lamb1,lamb2, norm = "mods_M", niters = 10)
        
        # TD = T@D
        TDerr = 0
        #TDerr = np.mean((TD - TDhat)**2)/(np.std(TDhat)**2)
        #P,_ = em.optimize_M(F - bhat,TD,1,method = "OMP")
        #P = np.linalg.pinv(TD)@(F-bhat)

        T = ((F-bhat)@np.linalg.pinv(D@P))

        #Ferr_TDPhat = np.mean((F - TDhat@P - bhat)**2)
        
        #Mhat = np.linalg.pinv(T)@(F-bhat)
        Mhat,_ = optimize_M(F-bhat,T,lamb2,norm = "loci", fit_intercept=False)
        That = deepcopy(T)
        T = deepcopy(That)

        Ferr_TDPhat = np.mean((F - (T@D)@P - bhat)**2)


        Ts[i] = T
        Ds[i] = D
        Ps[i] = P
        errs[i] = Ferr_TDPhat
        
        if verbose:
            print("%02d  %.4f   %.4f   %.4f   %.4f   %.4f   %.4f" %(Kp, Merr, Ferr_M, Ferr_Mhat, Ferr_TDhat, TDerr, Ferr_TDPhat))

    minner = np.argmin(errs)
    T = Ts[minner]
    D = Ds[minner]
    P = Ps[minner]
    if verbose:
        print("TDP:     %02d  %.3f" %(Kp, errs[minner]))

    return T,D,P,errs[minner]

def run_svd(F, K,  M_lb=0, W_lb=0):
    u,s,vh=np.linalg.svd(F)
    W, M = u[:,:K]*s[None,:K], vh[:K]
    W = (np.abs(W)> W_lb)*W
    M = (np.abs(M)> M_lb)*M
    return M, W

def dropped_loci(M):
    K = M.shape[0]
    nzl = np.abs(np.sum(M, axis = 0)) > 0
    ndl = len(nzl)-sum(nzl)
    return ndl

def dropped_envs(W):
    K = W.shape[1]
    nze = np.abs(np.sum(W, axis = 1)) > 0
    nde = len(nze)-sum(nze)
    return nde


def grassmann(A,B, mode = "squares"): #A, B are K x L
    if np.sum(np.abs(A)) == 0 or np.sum(np.abs(B)) == 0:
        return np.sqrt(A.shape[0]*np.pi**2)
    Ma = orth(A.T)
    Mb = orth(B.T)
    Q = Ma.T @ Mb

    u,s,vh = np.linalg.svd(Q)
    if mode == "squares":
        s = (s>=1)*np.ones(len(s)) + (s<1)*s
        return np.sqrt(np.sum(np.arccos(s)**2))
    elif mode == "cosines":
        return np.sum((1-s)/2)
    
def ave_mod_per_loci(M):
    K = M.shape[0]
    nzl = np.abs(np.sum(M, axis = 0)) > 0
    return np.mean(np.sum(np.abs(M[:, nzl])>0,axis=0)) # doesn't include dropped loci

def ave_mod_per_env(W):
    K = W.shape[1]
    nze = np.abs(np.sum(W, axis = 1)) > 0
    return np.mean(np.sum(np.abs(W[nze, :])>0,axis=1)) # doesn't include dropped envs

def get_attributes_of_solns(fct_F, mode = 'reg'):
    mods =  []
    keffs = []
    errors = []
    ave_part_in = []
    l1s = []
    l2s = []
    if fct_F.init_mode == "FXY":
        T_errors = []
        V_errors = []
    ks = []
    ave_mod_per_env  = []
    for p in fct_F.computed_params(printout=False):
        if p[0]!=mode: continue
        l1s.append(p[2][0])
        l2s.append(p[2][1])
        ks.append(p[1])
        keffs.append(int(np.ceil(np.sum(fct_F.M_preds[p]**2))))
        mods.append(fct_F.modularities[p])
        errors.append(np.mean(fct_F.L2FFs[p]))
        ave_part_in.append(ave_mod_per_loci(fct_F.M_preds[p])) # doesn't include dropped loci
        ave_mod_per_env.append(ave_mod_per_loci(fct_F.W_preds[p].T)) # doesn't include dropped loci

        if fct_F.init_mode == "FXY":
            T_errors.append(np.mean(fct_F.R2Ts[p]))
            V_errors.append(np.mean(fct_F.R2Vs[p]))
        print(p, ks[-1], keffs[-1], errors[-1], ave_part_in[-1])
    atts= {}
    atts["modularity"] = mods
    atts["ave modules / loci"]= ave_part_in
    atts["ave modules / env"]= ave_mod_per_env
    atts["keff"] = keffs
    atts["error"] = errors
    atts["k"] = ks
    atts["lamb1"] = l1s
    atts["lamb2"] = l2s
    if fct_F.init_mode == "FXY":
        atts["training variance explained"] = T_errors
        atts["validation variance explained"] = V_errors

    return atts

def plot_attributes(atts, x_key, y_key, color_key, filter_type=None, filter_range =None, cmap = "Dark2", title = None, xrange = None, yrange = None, crange = None):
    if filter_type is not None:
        xs, ys, cs = [], [], []
        for _, x in enumerate(atts[filter_type]):
            if x>filter_range[0] and x< filter_range[1]:
                xs.append(atts[x_key][_])
                ys.append(atts[y_key][_])
                cs.append(atts[color_key][_])
    else:
        xs, ys, cs = atts[x_key], atts[y_key], atts[color_key]
    if crange is None:
        plt.scatter(xs,ys, c = cs, cmap = cmap)
    else:
        plt.scatter(xs,ys, c = cs, cmap = cmap, vmin = crange[0], vmax = crange[1])

    if xrange is not None: plt.xlim(xrange)
    if yrange is not None: plt.ylim(yrange)
    plt.xlabel(x_key)
    plt.ylabel(y_key)
    if title is not None: plt.title(title)
    cbar = plt.colorbar()
    cbar.set_label(color_key)
    plt.grid()
    plt.show()
    

################### Functions for generating synthetic F = WM + c ###########################

def get_W_and_M_structured(h, H, p, L, hub_sparsity, M_sparsity, rho, seed = 0):
     #there are h hub envs with p possible perturbations to each. rho sets magnitude of the perturbation, Total of h*(p+1) possible environments. 
    #1 core trait which affects all environments, H hub modules, p perturbation modules. K = 1 + H + p
    
    if M_sparsity is None: M_sparsity = 1
    if hub_sparsity is None: hub_sparsity = 1
    np.random.seed(seed)

    E = int(h*(1+p))
    K = int(H + p)

    W = np.zeros((E,K))

    for e in range(h): #for each hub env
        filt_hub = np.random.uniform(size = (H)) < hub_sparsity # hub-module membership 
        modules_hub = np.zeros(K,dtype=int)
        modules_hub[:H][filt_hub] = 1

        Whub = np.random.randn(K)
        #Whub = np.ones(K)

        W[e*(1+p),:] = Whub*modules_hub #hub environment

        for n in range(p):
            modules_pert = deepcopy(modules_hub)
            modules_pert[H+n] = rho
            W[e*(1+p) + n + 1, :] = Whub*modules_pert
                
   
    filt = np.random.uniform(size = (K,L)) < M_sparsity # loci-module membership
    M = filt*np.random.randn(K,L) # weights are iid normal
    M = M/(1e-9+np.sqrt(np.sum(M**2,axis=1))[:,None]) # normalize so each module has norm 1
    
    F = W @ M
    
    # chose magnitude of C (is one std of means across loci)
    locs = np.mean(F ,axis=0)
    mag_locs = np.std(locs)
    
    C = mag_locs*np.random.randn(L)

    return W,M,C


def get_W_and_M_simple(E,L,K, M_sparsity, W_sparsity, seed = 0):
    # E = envs
    # L = loci
    # K = modules
    # M_sparsity = bernoulli prob of each loci in each module; 1 if None
    # W_sparsity = bernoulli prob of each env using each module; 1 if None
  
    np.random.seed(seed)

    if M_sparsity is None: M_sparsity = 1
    if W_sparsity is None: W_sparsity = 1
    
    filt = np.random.uniform(size = (E,K)) < W_sparsity
    W = filt*np.random.randn(E,K) # weights are iid normal
  
    filt = np.random.uniform(size = (K,L)) < M_sparsity # loci-module membership
    M = filt*np.random.randn(K,L) # weights are iid normal
    M = M/(1e-9+np.sqrt(np.sum(M**2,axis=1))[:,None]) # normalize so each module has norm 1
    
    F = W @ M
    
    # chose magnitude of C (is one std of means across loci)
    locs = np.mean(F ,axis=0)
    mag_locs = np.std(locs)
    C = mag_locs*np.random.randn(L)

    return W,M,C


############### OLD (PROBABLY DELETE) ####################


#plots the dot product similarity between W vectors amongst themselves and with singular vectors
def plot_dots(F,W, savefile= None, compare = "w"): #compare = "w" or "u" for Ws and SVD resp.
    K = W.shape[1]

    u,s,vh=np.linalg.svd(F)

    u_K = u[:,:K]
    s_K = s[:K]
    vh_K = vh[:K]

    if compare == "w":
        dots = np.zeros((K,K))
        for i in range(K):
            for j in range(K):
                dots[i,j] = np.dot(W[:,i],W[:,j])

        plt.close("all")
        fig,axis = plt.subplots(1,1,figsize=  (5,4.5))
        im = axis.imshow(np.abs(dots),vmin = 0,vmax = 1)
        fig.colorbar(mappable=im)
        axis.tick_params(labelsize = 18)
        axis.set_title("$w_i.w_j$",fontsize = 20)
        if savefile is not None:
            fig.savefig(savefile + "_%d"%W.shape[1] + "_ww.png",dpi = 150)
        plt.show()

    else:
        dots_svd = np.zeros((K,K))
        for i in range(K):
            for j in range(K):
                dots_svd[i,j] = np.dot(W[:,i],u_K[:,j])

        filt = np.argsort(-np.abs(np.diag(dots_svd)))  
        dots_svd_sorted = (dots_svd[:,filt])[filt,:]

        plt.close("all")
        fig,axis = plt.subplots(1,1,figsize=  (5,4.5))
        im = axis.imshow(np.abs(dots_svd_sorted),vmin = 0,vmax = 1)
        fig.colorbar(mappable=im)
        axis.tick_params(labelsize = 18)
        axis.set_title("$w_i.u_j$",fontsize = 20)
        if savefile is not None:
            fig.savefig(savefile + "_%d"%W.shape[1] + "_wu.png",dpi = 150)
        plt.show()
        
        


def get_errors_heldout(F,envs_ho,K,lamb2s,lamb1,verbose = True):
    E_ho = len(envs_ho)
    E = F.shape[0]
    L = F.shape[1]
    
    envs_ho_slice = np.zeros(E,dtype=bool)
    envs_ho_slice[envs_ho] = 1

    
    Ws, Ms = run_factorizer_regularized_range(F[~envs_ho_slice], K, lamb2s, lamb1, norm = "loci")

    u,s,vh = np.linalg.svd(F[~envs_ho_slice])
    u_K = u[:,:K]
    s_K = s[:K]
    vh_K = vh[:K]
    Msvd = vh_K

    Ws = np.zeros((len(Ws),E,K))
    Wsvds = np.zeros((len(Ws),E,K))

    F_ems = np.zeros((len(Ws),E,L))
    F_svds = np.zeros((len(Ws),E,L))

    for im,M in enumerate(Ms):
        A = np.eye(K)*lamb1 + np.matmul(M,M.T)
        B = np.linalg.solve(A,M)
        Ws[im] = np.matmul(F,B.T)
        F_ems[im] = Ws[im]@M

        A = np.eye(K)*lamb1 + np.matmul(Msvd,Msvd.T)
        B = np.linalg.solve(A,Msvd)
        Wsvds[im] = np.matmul(F,B.T)
        F_svds[im] = Wsvds[im]@Msvd

    if verbose:
        print("lamb2    Err EM nHO     Err EM HO    Err SVD nHO     Err SVD HO    ")
    errs_EM_nho = np.zeros(len(lamb2s))
    errs_EM_ho = np.zeros(len(lamb2s))
    errs_SVD_nho = np.zeros(len(lamb2s))
    errs_SVD_ho = np.zeros(len(lamb2s))

    for i in range(len(lamb2s)):
        err_EM_nho = np.mean((F[~envs_ho_slice] - F_ems[i][~envs_ho_slice])**2)
        err_EM_ho = np.mean((F[envs_ho_slice] - F_ems[i][envs_ho_slice])**2)

        err_SVD_nho = np.mean((F[~envs_ho_slice] - F_svds[i][~envs_ho_slice])**2)
        err_SVD_ho = np.mean((F[envs_ho_slice] - F_svds[i][envs_ho_slice])**2)

        errs_EM_nho[i] = err_EM_nho
        errs_EM_ho[i] = err_EM_ho
        errs_SVD_nho[i] = err_SVD_nho
        errs_SVD_ho[i] = err_SVD_ho
        
        if verbose:
            print("%.3f      %.3f          %.3f           %.3f          %.3f"%(lamb2s[i],err_EM_nho,err_EM_ho, err_SVD_nho, err_SVD_ho))

    return errs_EM_nho, errs_EM_ho, errs_SVD_nho, errs_SVD_ho

def get_atts_with_keff_and_k_fixed(atts, k ):
    restricted_atts = dict.fromkeys(atts.keys(),[])
    for key in atts:
        new_list = []
        for _ in range(len(atts["keff"])):
            if atts["keff"][_] == k and atts["k"][_]==k:
                new_list.append(atts[key][_])
        restricted_atts[key] = new_list

    return restricted_atts

def plot_tradeoffs(atts, atts_rot, kmin, kmax, ks=None, title = None):
    fontsize = 14
    restricted_atts = {}
    restricted_atts_rot = {}
    if kmin is not None:
        ks = range(kmin,kmax+1)
    else:
        kmin = min(ks)
        kmax = max(ks)
    for k in ks:
        restricted_atts[k] = get_atts_with_keff_and_k_fixed(atts, k)
        restricted_atts_rot[k] = get_atts_with_keff_and_k_fixed(atts_rot, k)
        #plt.scatter(restricted_atts[k]["ave modules / loci"], restricted_atts[k]["error"])
        #plt.scatter(restricted_atts_rot[k]["ave modules / loci"], restricted_atts_rot[k]["error"])
        #plt.title(f"k={k}")
        #plt.show()
        if len(restricted_atts_rot[k]["ave processes / loci"])>0:
            mn = min(restricted_atts_rot[k]["ave processes / loci"])
            mx = max(restricted_atts_rot[k]["ave processes / loci"])
        else:
            mn = 0
            mx = 0
        f = interpolate.interp1d(restricted_atts_rot[k]["ave processes / loci"], restricted_atts_rot[k]["error"])
        restricted_atts[k]["FO err / F err"] = []
        for _,aml in enumerate(restricted_atts[k]["ave processes / loci"]):
            if aml < mn or aml > mx:
                restricted_atts[k]["FO err / F err"].append(-1)
            else:
                restricted_atts[k]["FO err / F err"].append(f(aml)/restricted_atts[k]["error"][_])
        #plt.scatter(restricted_atts[k]["ave modules / loci"],restricted_atts[k]["FO err / F err"] )
        #plt.show()

    plt.figure(figsize=(12,8))
    mn = 100
    mx = -1
    for k in ks:
        if len(restricted_atts[k]["FO err / F err"])>0:
            mx = max(max(restricted_atts[k]["FO err / F err"]), mx)
        if len([_ for _ in restricted_atts[k]["FO err / F err"] if _>0])>0:
            mn = min(min([_ for _ in restricted_atts[k]["FO err / F err"] if _>0]), mn)

    for k in ks:
        x1s, x2s, y1s, y2s, c1s = [], [], [], [], []
        for n, val in enumerate(restricted_atts[k]["FO err / F err"]):
            if val > 0:
                x1s.append(restricted_atts[k]["error"][n])
                y1s.append(restricted_atts[k]["ave processes / loci"][n])
                c1s.append(restricted_atts[k]["FO err / F err"][n])
            else:
                x2s.append(restricted_atts[k]["error"][n])
                y2s.append(restricted_atts[k]["ave processes / loci"][n])  
        plt.plot(restricted_atts[k]["error"],restricted_atts[k]["ave processes / loci"], label = f"k = {k}", c = 'black', alpha = .8-.8*((k-kmin) / (kmax-kmin + 1)))
        plt.scatter(x2s, y2s, marker = 'd', c= 'black')
        plt.scatter(x1s,y1s, c = c1s, vmin = mn, vmax =mx, cmap = 'viridis_r')


    plt.xlabel("error", fontsize = fontsize)
    plt.ylabel("ave processes / loci", fontsize = fontsize)
    cbar = plt.colorbar()
    cbar.set_label("FO error / F error")
    plt.legend()
    plt.grid()
    if title is not None: plt.title(title, fontsize = fontsize)
    plt.show()
    
