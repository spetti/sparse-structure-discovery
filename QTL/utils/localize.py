import numpy as np
import matplotlib.pyplot as plt
import scipy
import sys

#subtract residuals of top loci (those with effect size at least 0.001 in some env)
def prep_arrays(F, X, Y, loci, thresh, verbose):
    F_ori = F
    loci = np.array([e for e, i in enumerate(loci) if i])
    print(np.max(np.abs(F_ori),axis=0))
    loci_sorted = np.argsort(-np.max(np.abs(F_ori),axis=0))
    numloci = np.sum(np.max(np.abs(F_ori),axis=0) > thresh)
    if verbose: print(f"subtracting out effects of {numloci} loci with effect size greater than {thresh} in some env")
    idx_filt = loci_sorted[:numloci]
    loci_filt = loci[idx_filt]
    Y -= np.nanmean(Y,axis=0)
    Ysub = X[:,loci_filt]@F_ori[:,idx_filt].T
    
    return idx_filt, loci_filt, F_ori, X, Y, Ysub, loci

# computes additive effect approximation and confidence interval for a locus l / ls
# z is number of std of tolerance
# width is max width of confidence interval

def compute_fbest_and_intervals(l, ls, F_ori, X, Y, Ysub, width, z, verbose):
    S = X.shape[1]
    if verbose: print(f"processing locus {l}, {ls}")
    varX = np.std(X[:,ls])**2
    varYs = (np.nanstd(Y,axis=0)**2)

    ran = np.arange(max(0,ls - width), min(S,ls + width + 1),dtype=int)
    mus = np.nanmean(X[:,ran],axis=0)
    cov = ((X[:,ran] - mus).T)@(X[:,ran] - mus)/X.shape[0]
    corrs = cov/np.sqrt(np.outer(np.diag(cov),np.diag(cov)))
    Ypred = Ysub - X[:,None,ls]*F_ori[None,:,l]
    Ypred -= np.nanmean(Ypred,axis=0)

    R = Y - Ypred
    R -= np.nanmean(R,axis=0)

    Xfilt = 1.0*X[:,None,ran] 
    Xfilt -= np.mean(Xfilt,axis=0)[None,:]

    num = np.nanmean(R[:,:,None]*Xfilt,axis=0)
    den = np.nanmean(Xfilt**2,axis=0)

    fbest = num/den
    intervals = np.zeros(fbest.shape)

    for e in range(Y.shape[1]):
        r2minner = np.argmax(np.abs(fbest[e]))
        fe = fbest[e][r2minner]

        sigma2 = np.nanstd(R[:,e] - fe*(X[:,r2minner + ran[0]]))**2

        rhs = 2*sigma2*(z**2)/(X.shape[0]*fe*fe*varX)
        lhs = 1-corrs[:,r2minner]

    for e in range(Y.shape[1]):
        r2minner = np.argmax(np.abs(fbest[e]))
        fe = fbest[e][r2minner]

        sigma2 = np.nanstd(R[:,e] - fe*(X[:,r2minner + ran[0]]))**2

        rhs = 2*sigma2*(z**2)/(X.shape[0]*fe*fe*varX)
        lhs = 1-corrs[:,r2minner]

        interval = lhs < rhs
        intervals[e] = interval
        
    return fbest, intervals


# round 1 localization
def localize_and_split(idx_filt, loci_filt, F_ori, X, Y, Ysub, loci, width, z, verbose):
    
    to_keep_all = []
    for ind in range(len(idx_filt)):
        l = idx_filt[ind]
        ls = loci_filt[ind]
        fbest, intervals = compute_fbest_and_intervals(l, ls, F_ori, X, Y, Ysub, width,z, verbose)
        r_fbest = np.copy(fbest)
        r_intervals = np.copy(intervals)
        tops = []
        while r_intervals.shape[0]>0:
            sums = np.sum(r_intervals*np.abs(r_fbest), axis = 0)
            top = np.argmax(sums)
            keep = r_intervals[:,top] == 0
            r_intervals= r_intervals[keep,:]
            r_fbest= r_fbest[keep, :]
            tops.append(top) 
        if verbose: print([ls +50-i for i in tops])
        to_keep_all.append([ls +50-i for i in tops])
    all_loci = [i for j in to_keep_all for i in j]
    all_loci = np.sort(list(set(all_loci)))
    return all_loci

# round 2 localization
def compute_interval_intersections(idx_filt, loci_filt, F_ori, X, Y, Ysub, loci, width, z, thresh, verbose):
    num_to_localize = np.sum(np.max(np.abs(F_ori),axis=0) > thresh)
    loci_localized = []
    if verbose: print(num_to_localize)
    loci_lists = []
    for ind in range(num_to_localize):
        l = idx_filt[ind]
        ls = loci_filt[ind]
        fbest, intervals = compute_fbest_and_intervals(l, ls, F_ori, X, Y, Ysub, width,z, verbose)
       
        sums = np.sum(intervals*np.abs(fbest), axis = 0) 
        top = np.argmax(sums)
        sets = []
        bool_intervals = intervals.astype(dtype=bool)
        for e in range(intervals.shape[0]):
            if bool_intervals[e, top]:
                sets.append(set(np.where(bool_intervals[e, :]==True)[0]))
        intersection = sets[0]
        for i in range(1, len(sets)):
            intersection = intersection.intersection(sets[i])
        if verbose: print([i-50+ls for i in intersection])
        loci_lists.append([i-50+ls for i in intersection])
        loci_localized.append((l,ls))
    return loci_localized, loci_lists