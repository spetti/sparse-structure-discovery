import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.stats import bernoulli
from scipy.stats import ortho_group  # Requires version 0.18 of scipy
from copy import deepcopy
from matplotlib import rc
from sklearn.metrics import r2_score
from sklearn.linear_model import Lasso, Ridge, LinearRegression, orthogonal_mp, Lars
from scipy.linalg import orth
from scipy.interpolate import interp1d
import sys
import pandas as pd
import ssd
from IPython.display import display, Latex
import matplotlib
import palettable
from scipy.interpolate import griddata
from palettable.cartocolors.qualitative import Prism_8
import matplotlib as mpl

matplotlib.rcParams["font.sans-serif"] = "Arial"
matplotlib.rcParams["font.family"] = "sans-serif"


class factorizer:

    ############### Initialization functions ###############

    def __init__(self):

        self.init_mode = None  # F, FXY, MW

        # inputs and processing
        self.FTrue = None  # input F
        self.loc_means = None  # mean effect per loci
        self.subtract_means = True  # whether to subtract means (True or False)
        self.rescale = True
        self.scale = None  # std deviation
        self.FF = None  # [noise added if MW] centered and normalized F,
        self.E = None
        self.L = None

        self.Xtrain = None
        self.Ytrain = None
        self.Xval = None
        self.Yval = None

        self.sig = None
        self.seed = None
        self.K = None
        self.c = None

        # predictions and errors
        # all will be dictionaries with key = parameter settings
        self.M_preds = {}  # M predictions
        self.W_preds = {}  # W predictions
        self.b_preds = {}  # W predictions
        self.ave_mod_per_loci = (
            {}
        )  # ave number of modules each loci participates in (excludes loci with no modules)
        self.ave_mod_per_env = (
            {}
        )  # ave number of modules that each envs uses (excludes envs that use no no modules)
        self.ndl = {}  # number of dropped loci in prediction
        self.nde = {}  # number of dropped envs in prediction

        self.G2SVDs = {}  # Grassman distance with SVD with same K
        self.L2Ms = {}  # L2 error in M of prediction
        self.L2FFs = (
            {}
        )  # L2 error between prediction and the F that gets fed to the factorization algorithm ([noise added if MW], rescaled, centered)
        self.L2FTs = {}  # L2 error between prediction and the true F
        self.R2Ts = None  # R2 error on training set
        self.R2Vs = None  # R2 error on validation set

        self.error_names = {
            "G": "Grassmann to SVD",
            "M": r"L2 error in modules $M$",
            "FF": r"L2 error in reconstructing $F_{processed}$",
            "FT": r"L2 error in reconstructing $F_{true}$",
            "T": "% variance explained on training set",
            "V": "% variance explained on validation set",
        }

        # names
        self.env_names = None
        self.loci_names = None

    def init_with_F(self, F):

        self.init_mode = "F"

        # normalize F
        self.FTrue = deepcopy(F)
        F = deepcopy(F)
        self.loc_means = np.mean(F, axis=0) * self.subtract_means
        F -= self.loc_means
        if self.rescale:
            self.scale = np.std(F)
        else:
            self.scale = 1.0
        F /= self.scale
        self.FF = F
        self.E, self.L = F.shape[0], F.shape[1]

    def init_with_F_XY(self, F, Xtrain, Ytrain, Xval, Yval):

        self.init_with_F(F)
        self.init_mode = "FXY"

        self.Xtrain = Xtrain
        self.Ytrain = Ytrain
        self.Xval = Xval
        self.Yval = Yval

        self.R2Ts = {}
        self.R2Vs = {}

    def init_with_MW(self, M, W, c, sig, seed=0):

        self.init_mode = "MW"
        self.M = M
        self.W = W
        self.c = c
        self.K = self.M.shape[0]
        self.L = self.M.shape[1]
        self.E = self.W.shape[0]
        self.true_ampl = ssd.ave_mod_per_loci(self.M)
        self.true_ampe = ssd.ave_mod_per_env(self.W)

        self.sig = sig
        self.seed = seed
        np.random.seed(seed)

        F = self.W @ self.M + c
        self.FTrue = deepcopy(F)  # true F
        F += np.std(F) * np.random.randn(self.E, self.L) * sig
        self.loc_means = np.mean(F, axis=0) * self.subtract_means
        F -= self.loc_means
        if self.rescale:
            self.scale = np.std(F)
        else:
            self.scale = 1.0
        F /= self.scale
        self.FF = F  # noise added and then normalized and centered (if applicable)

    ############### Factorization functions ###############

    # regularized factorization
    def factorize_regularized_range(
        self,
        K,
        lamb1s,
        lamb2s,
        W_lb=0,
        M_lb=0,
        method_M="Lasso",
        method_W="Lasso",
        verbose=True,
    ):
        Ws, Ms, bs, mods, nnzls = ssd.run_factorizer_regularized_range(
            self.FF,
            K,
            lamb1s,
            lamb2s,
            method_M=method_M,
            method_W=method_W,
            norm="loci",
            verbose=verbose,
            M_lb=M_lb,
            W_lb=W_lb,
            fit_intercept=not self.subtract_means,
        )
        if self.init_mode == "MW" and verbose:
            print(
                f"True ave mod / loci: {self.true_ampl:.2f}, True ave mod / env: {self.true_ampe:.2f}"
            )
        for i, lamb1 in enumerate(lamb1s):
            for j, lamb2 in enumerate(lamb2s):
                self.store_results_and_losses(
                    ("reg", K, (ssd.ptk(lamb1), ssd.ptk(lamb2)), (M_lb, W_lb)),
                    Ms[i, j, ...],
                    Ws[i, j, ...],
                    bs[i, j, ...],
                )

    # factorize with SVD
    def svd(self, K, M_lb=0, W_lb=0):
        M, W = ssd.run_svd(self.FF - np.mean(self.FF, axis=0), K, M_lb, W_lb)
        self.store_results_and_losses(
            ("svd", K, None, (M_lb, W_lb)), M, W, np.mean(self.FF, axis=0)
        )

    # cluster loci into pathways
    def cluster_loci(self, Kp, params, lamb1, lamb2, niter=1, verbose=True):
        M = self.M_preds[params]
        W = self.W_preds[params]
        b = self.b_preds[params]
        F = deepcopy(self.FF)
        T, D, P, err = ssd.optimize_TDP(
            F, M, W, b, Kp, lamb1, lamb2, niter=niter, verbose=verbose
        )
        return T, D, P, err

    def store_results_and_losses(self, key, M, W, b):
        # if we know true M's reorder so that predicted Ms are in similar order
        if self.init_mode == "MW" and M.shape[0] == self.M.shape[0]:
            M, W, b = self.permute_M_W(M, W, b)
        self.M_preds[key], self.W_preds[key], self.b_preds[key] = M, W, b
        self.ndl[key] = ssd.dropped_loci(M)
        self.ave_mod_per_loci[key] = ssd.ave_mod_per_loci(M)
        self.ave_mod_per_env[key] = ssd.ave_mod_per_env(W)
        self.nde[key] = ssd.dropped_envs(W)
        self.compute_all_errors(key)

    # tools for reordering M and W to match ground truth (only used on syn data with W, M known)
    def permute_M_W(self, M_pred, W_pred, b_pred):
        perm = self.get_perm(M_pred, self.M)
        return M_pred[perm, :], W_pred[:, perm], b_pred

    def get_perm(self, M_pred, M_true):
        def _remove(p1, p2, the_list):
            return [_ for _ in the_list if _[0] != p1 and _[1] != p2]

        K = M_pred.shape[0]
        if K != M_true.shape[0]:
            # could modify code so we don't need this; but if we know the true K, will we run it with different values
            raise ValueError("need same shape K")
        dots_svd = np.abs(np.einsum("ij,kj->ik", M_pred, M_true))
        pairs = [(a, b) for a in range(K) for b in range(K)]
        plist = [pairs[_] for _ in np.argsort(dots_svd, axis=None)[::-1]]
        perm = []
        while len(plist) > 0:
            perm.append(plist[0])
            plist = _remove(perm[-1][0], perm[-1][1], plist)
        new_pos = [0 for _ in range(K)]
        for pair in perm:
            new_pos[pair[1]] = pair[0]
        return new_pos

    def computed_params(self, printout=True):
        if printout:
            print("Factorization complete for the following parameter choices:")
            for params in self.M_preds.keys():
                print(params)
        return list(self.M_preds.keys())

    ############### Functions for computing errors ###############

    def get_r2s(self, F, dset="val"):  # dset = val or train

        if dset == "train":
            X = deepcopy(self.Xtrain)
            Y = deepcopy(self.Ytrain)
        elif dset == "val":
            X = deepcopy(self.Xval)
            Y = deepcopy(self.Yval)

        Ypred = np.matmul(X, F.T)
        Ypred -= np.nanmean(Ypred, axis=0)

        Y -= np.nanmean(Y, axis=0)

        r2s = []
        for e in range(F.shape[0]):
            r2 = 100 * r2_score(Y[:, e], Ypred[:, e])
            r2s += [r2]
        return r2s

    def error_mode_to_dictionary(self, mode):

        # L2 error in reconstruction of F that is fed into factorization
        if mode == "FF":
            loc = self.L2FFs

        elif mode == "FT":
            loc = self.L2FTs

        elif mode == "G":
            loc = self.G2SVDs

        # L2 error in reconstruction of true M (synthetic data only)
        elif mode == "M":
            assert self.init_mode == "MW", "error in M requires MW init mode"
            loc = self.L2Ms

        # R2 error on training set (must have X and Y input)
        elif mode == "T":
            assert (
                self.init_mode == "FXY"
            ), "error on training set requires FXY init mode"
            loc = self.R2Ts

        # R2 error on validation set (must have X and Y input)
        elif mode == "V":
            assert (
                self.init_mode == "FXY"
            ), "error on validation set requires FXY init mode"
            loc = self.R2Vs

        else:
            raise ValueError("error mode must be FF, FT, M, T, or V")

        return loc

    def compute_error_from_M_W_b(self, mode, M, W, b, special_FF=None):

        # assumes error with respect to the F being factorized in class
        # for use with rotating F, use special_FF  to feed in the F
        # special_FF only relevant for mode FF and G

        err = None
        if special_FF is not None:
            if mode not in ["FF", "G"]:
                raise ValueError(
                    "Doesn't make sense to use special FF in modes other than FF and G"
                )
            FF = special_FF
        else:
            FF = self.FF
        if mode == "M":
            if self.M.shape[0] != M.shape[0]:
                err = -1
            else:
                err = 1 - np.abs(np.sum(self.M * M, axis=-1))
        elif mode == "FF":
            err = np.mean(np.square(np.matmul(W, M) + b - FF), axis=-1)
        elif mode == "G":
            # compute SVD if needed
            k = M.shape[0]
            if special_FF is None:
                if ("svd", k, None, (0, 0)) not in self.M_preds:
                    self.svd(k, 0, 0)
                err = ssd.grassmann(M, self.M_preds[("svd", k, None, (0, 0))])
            else:
                svd_M = ssd.run_svd(FF - np.mean(FF, axis=0), k)[0]
                err = ssd.grassmann(M, svd_M)

        else:
            Fpred = self.scale * (np.matmul(W, M) + b) + self.loc_means
            if mode == "FT":
                err = np.mean(np.square(Fpred - self.FTrue), axis=-1)
            elif mode == "V":
                err = self.get_r2s(Fpred, dset="val")
            elif mode == "T":
                err = self.get_r2s(Fpred, dset="train")
        return err

    def compute_error(self, params, mode, printout=False):

        loc = self.error_mode_to_dictionary(mode)

        if params not in self.M_preds.keys():
            raise ValueError(
                f"Cannot find factorization with parameters {params}. Run factorization function first."
            )

        if params in loc.keys():
            if printout:
                print(f"previously computed error {mode} for {params}; doing nothing")

        else:
            # compute error
            loc[params] = self.compute_error_from_M_W_b(
                mode, self.M_preds[params], self.W_preds[params], self.b_preds[params]
            )
            if printout:
                print(f"computed error {mode} for {params}")

    def compute_all_errors(self, params, printout=False):
        # errors relevant for each mode
        modes = ["G", "FF", "FT"]
        if self.init_mode == "MW":
            modes += ["M"]
        elif self.init_mode == "FXY":
            modes += ["V", "T"]
        for mode in modes:
            self.compute_error(params, mode, printout=printout)

    def find_key(self, avl_t, ave_t):
        pminner = 0
        dist = 1e4

        for p in self.W_preds.keys():
            W = self.W_preds[p]
            M = self.M_preds[p]

            avgmodloc = ssd.ave_mod_per_loci(M)
            avgmodenv = ssd.ave_mod_per_env(W)

            if np.sqrt((ave_t - avgmodenv) ** 2 + (avl_t - avgmodloc) ** 2) < dist:
                dist = np.sqrt((ave_t - avgmodenv) ** 2 + (avl_t - avgmodloc) ** 2)
                pminner = p

        W = self.W_preds[pminner]
        M = self.M_preds[pminner]

        avgmodloc = ssd.ave_mod_per_loci(M)
        avgmodenv = ssd.ave_mod_per_env(W)

        return pminner

    def get_keff(self, avl_t, ave_t):
        p = self.find_key(avl_t, ave_t)
        M = self.M_preds[p]
        keff = int(np.ceil(np.sum(M**2)))
        return keff


def print_factorizer_help():
    print("Ways to initialize: ")
    print("init_mode = F, just an F matrix; use init_with_F")
    print(
        "init_mode = MW the matrices M,W, and C generated synthetically; use init_with_MW"
    )
    print(
        "init_mode = F_XY, an F matrix and training and validation data; use init_with_F_XY"
    )

    print("\n")
    print("Factorization functions: ")
    print("factorize_disjoint: algorithm that assigns only one module per loci")
    print(
        "factorize_regularized_range: algorithm with regularize to encourage sparse modules"
    )
    print("svd: takes the svd\n")
    print(
        "When these functions are called, the predicted M and W, relevant losses, and modularity are stored in dictionaries."
    )
    print(
        "The keys have the form (method, number of modules k, method-specific parameters, truncation thresholds for M and W.)"
    )
    print("To view all parameters computed, call computed_params().")

    print("\n")
    print("Error modes: ")
    print(
        "FF: error between reconstructed F and the F that was fed to the factorization algorithm "
    )
    print(
        "FT: error between reconstructed F after uncentering and scaling and the input F (or in init_mode MW, F = WM + C)"
    )
    print(
        "G: Grassmann distance between the predicted M and the M from SVD computed with same number of modules k"
    )
    print(
        "M: L2 error between predicted M and the true M; only works in init_mode = MW"
    )
    print("T: R2 error on training set; only works in init_mode = F_XY")
    print("V: R2 error on validation set; only works in init_mode = F_XY")


################ Factorizer plotting functions ####################


############ Functions to display M, W, and modularity ###############


def print_Ms(fct, params, display="all"):
    print("M: print \n")
    M = fct.M_preds[params]
    if display == "all":
        display = M.shape[1]
    sorted_ = np.argsort(-np.std(fct.FTrue, axis=0))  # sort loci by most variance
    K = M.shape[0]
    print(f"Idx \t Loci       \t  Modules         \t Contributions to each module")
    if fct.loci_names is None:
        names = [f"L{_}" for _ in range(len(sorted_))]
    else:
        names = fct.loci_names
    for i in range(display):
        np.set_printoptions(precision=2, suppress=True)
        filt = np.abs(M[:, sorted_[i]]) > 0
        mods = [str(j) for j in np.arange(K)[filt]]
        print(
            f"{i:3d} \t {names[sorted_[i]]:<10} \t {str(np.arange(K)[filt]):<15} \t {str(M[filt,sorted_[i]]):<30}"
        )


def print_Ws(fct, params, max_print=100):
    print("W: print \n")
    W = fct.W_preds[params]
    np.set_printoptions(precision=2, suppress=True)
    Wfilt = deepcopy(W[:, np.sum(W**2, axis=0) > 0.5])
    sort_c = np.argsort(-np.sum(Wfilt**4, axis=0))
    if fct.env_names is None:
        names = [f"E{_}" for _ in range(fct.FF.shape[0])]
    else:
        names = fct.env_names
    print("Envs             Module weights")
    for e in range(min(fct.FF.shape[0], max_print)):
        print(f"{names[e]:<10} \t {str(Wfilt[e,sort_c]):<30}")
        # print("%5s      " %envs[e],Wfilt[e,sort_c])


def plot_Ms(
    fct,
    params,
    two_line=True,
    restricted = None,
    labelsize=15,
    pp=95,
    max_cols=90,
    sort="names",
    cluster_labels = None,
    save_name = None
):
    print("M: plot\n")

    if fct.loci_names is None:
        l_names = [f"L{_}" for _ in range(fct.FTrue.shape[1])]
    else:
        l_names = fct.loci_names

    M = (fct.M_preds[params])
    M, _ = cut_zero_mods(fct.M_preds[params],fct.W_preds[params])

    if restricted is not None and sort == "names":
        M = M[:,restricted]
        l_names = [l for i,l in enumerate(l_names) if restricted[i]]


    xlabel = "Loci"


    if sort == "names":
        sorted_ = np.argsort(l_names)
        M = M[:, sorted_]
        l_names = [l_names[l] for l in sorted_]
    elif sort == "magnitudes":
        sorted_ = np.argsort(-np.std(fct.FTrue, axis=0))  # sort loci by most variance
        M = M[:, sorted_]
        l_names = [l_names[l] for l in sorted_]
    elif sort == "clusters":
        if cluster_labels is None or len(cluster_labels) != M.shape[1]:
            raise ValueError("Number of cluster labels does not match shape of M")

        Mnorm = M/np.sqrt(np.sum(M**2 + 1e-5,axis=0))

        Mnew = np.zeros((M.shape[0],int(np.max(cluster_labels)+1)))
        for i in range(Mnew.shape[1]):
            Mnew[:,i] = np.mean(Mnorm[:,cluster_labels == i],axis=1)

        sorted_ = np.ones(Mnew.shape[1],dtype = bool)
        M = deepcopy(Mnew)
        l_names = np.arange(Mnew.shape[1]) + 1
        xlabel = "Groups"

    colorlabels = ["k" for i in range(M.shape[1])]

    if two_line:
        numl = min(int(M.shape[1] / 2), max_cols)
    else:
        numl = min(M.shape[1], max_cols)

    fig, axis = plt.subplots(1, 1, figsize=(24, 6))
    mm = np.percentile(np.abs(M), pp)
    # print(mm, np.max(np.abs(M)))
    im = axis.imshow(M[:, :numl], cmap="RdBu", vmin=-mm, vmax=mm)
    axis.tick_params(labelsize=labelsize)
    axis.set_xticks(np.arange(numl))
    axis.set_yticks(np.arange(M.shape[0]))
    axis.set_xticklabels(l_names[:numl], rotation=90)
    [t.set_color(colorlabels[i]) for i, t in enumerate(axis.xaxis.get_ticklabels())]
    axis.set_xlabel(xlabel,fontsize = labelsize*1.2)
    axis.set_ylabel("Modules",fontsize = labelsize*1.2)
    for sp in ["top", "bottom", "left", "right"]:
        axis.spines[sp].set_linewidth(1.25)
    if save_name is not None:
        if two_line:
            fig.savefig(save_name[:-4] + "_1" + save_name[-4:], bbox_inches='tight')
        else:
            fig.savefig(save_name, bbox_inches='tight')
    plt.show()

    if two_line:
        fig, axis = plt.subplots(1, 1, figsize=(24, 6))
        im = axis.imshow(
            np.abs(M[:, numl : 2 * numl - 1]), cmap="Greys", vmin=0, vmax=mm
        )
        axis.tick_params(labelsize=labelsize)
        axis.set_xticks(np.arange(len(l_names[numl : 2 * numl - 1])))
        axis.set_yticks(np.arange(M.shape[0]))
        axis.set_xticklabels(l_names[numl : 2 * numl - 1], rotation=90)
        [
            t.set_color(colorlabels[i + numl])
            for i, t in enumerate(axis.xaxis.get_ticklabels())
        ]
        axis.set_xlabel("Loci",fontsize = labelsize*1.2)
        axis.set_ylabel("Modules",fontsize = labelsize*1.2)
        for sp in ["top", "bottom", "left", "right"]:
            axis.spines[sp].set_linewidth(1.25)
        if save_name is not None:
            if two_line:
                fig.savefig(save_name[:-4] + "_2" + save_name[-4:], bbox_inches='tight')
        plt.show()


def plot_Ws(fct, params, labelsize = 15, pp = 95, max_rows = 100, colorbar_aspect = [1.02,0.0,0.015,1], T = None, save_name = None):
    print("W: plot\n")
    if fct.env_names is None:
        envs = [f"E{_}" for _ in range(fct.FF.shape[0])]
    else:
        envs = fct.env_names

    _, W = cut_zero_mods(fct.M_preds[params],fct.W_preds[params])

    if T is not None and T.shape[0] == W.shape[0]:
        W = deepcopy(T)
    nume = min(W.shape[0],max_rows)
    sort_c = np.argsort(-np.sum(W**4,axis=0))
    fig,axis = plt.subplots(1,1,figsize = (14,5))
    mm = np.percentile(W, pp)
    im = axis.imshow(W[:nume,:].T,cmap = "RdBu", vmin=-mm , vmax=mm)
    
    cax = axis.inset_axes(colorbar_aspect, transform=axis.transAxes)
    cb = fig.colorbar(mappable = im, cax = cax)
    cb.ax.tick_params(labelsize = labelsize)
    axis.set_xticks(np.arange(nume))
    axis.tick_params(labelsize = labelsize)
    axis.set_xticklabels(envs[:nume],rotation=90)
    axis.set_xlabel("Environments",fontsize = labelsize*1.25)
    axis.set_ylabel("Modules",fontsize = labelsize*1.25)
    for sp in ['top','bottom','left','right']:
        axis.spines[sp].set_linewidth(1.25)
    fig.tight_layout()
    if save_name is not None:
        fig.savefig(save_name, bbox_inches='tight')
    plt.show()


def display_modularity(fct, params, save_name = None, nbins= 10, density = False, labels=True, figsize = None):
    M = fct.M_preds[params]
    M,_ = cut_zero_mods(fct.M_preds[params],fct.W_preds[params])
    dist_sparsity = []
    K = M.shape[0]

    print("Module \t Number of loci included")
    for k in range(K):
        filt = np.abs(M[k]) > 0
        print(f"{k}\t {np.sum(filt)}")
    for l in range(M.shape[1]):
        filt = np.abs(M[:,l]) > 0
        dist_sparsity += [np.sum(filt)]
    if nbins is None:
        np.arange(M.shape[0]+1)
    if figsize ==None:
        figsize = (4,4)
    fig,axis = plt.subplots(1,1,figsize = figsize)
    nbins = K+1
    n,bins = np.histogram(dist_sparsity,bins = np.arange(nbins+1), density = density)
    axis.bar(bins[:-1], height = n,width = 0.8)
    axis.tick_params(labelsize = 18)
    axis.set_xticks(np.arange(nbins))


    if density:
        if labels: axis.set_ylabel("Fraction of loci",fontsize = 20)
        axis.set_yticks([.2,.4,.6,.8,1.0])
        axis.set_ylim(0,1)

    else:
        if labels:   axis.set_ylabel("Number of loci",fontsize = 20)
    if labels:    
        axis.set_xlabel("Number of modules involved",fontsize = 20)
    else:
        axis.set_xticklabels([])
        axis.set_yticklabels([])
    for sp in ['top','bottom','left','right']:
        axis.spines[sp].set_linewidth(1.0)
    fig.tight_layout()
    if save_name is not None:
        fig.savefig(save_name)
    plt.show()


def plot_and_print_W_M(
    fct, params, loci_to_print=15, two_line=True, labelsize=15, pp=95, max_rows=100, to_plot = True, to_print = True
):
    if to_print:
        print_Ms(fct, params, display=loci_to_print)
        print("")
    if to_plot:
        plot_Ms(fct, params, two_line=two_line, labelsize=labelsize, pp=pp)
        print("")
    if to_print:
        print_Ws(fct, params)
        print("")
    if to_plot:
        plot_Ws(fct, params, labelsize=labelsize, pp=pp, max_rows=max_rows)
        print("")
        display_modularity(fct, params)
    print(f"Grassmann distance to SVD with K={params[1]}: {fct.G2SVDs[params]:3f}")

def cut_zero_mods(M, W):
    return M[np.abs(M.sum(axis = -1))>0,:], W[:,np.abs(M.sum(axis = -1))>0]

############### Functions for plotting errors ###############


def plot_svd_error(fct, kmax, mode):
    xs = []
    ys = []
    loc = fct.error_mode_to_dictionary(mode)
    for k in range(1, kmax + 1):
        if ("svd", k, None, (0, 0)) not in fct.computed_params(printout=False):
            fct.svd(k)
        xs.append(k)
        ys.append(np.mean(loc[("svd", k, None, (0, 0))]))
    plt.scatter(xs, ys)
    plt.xlabel("Number of SVD components")
    plt.ylabel(f"{fct.error_names[mode]}")
    plt.grid()
    plt.show()


def format_params_nicely(fct, params):
    if params[0] == "reg":
        return rf"Regularized, $K={params[1]}$, $\lambda_1 = {params[2][0]}$, $\lambda_2 = {params[2][1]}$"
    elif params[0] == "disjoint":
        return rf"Disjoint, $K={params[1]}$, $\sigma = {params[2]}$"
    elif params[0] == "svd":
        return rf"SVD, $K={params[1]}$"
    else:
        return str(params)


def plot_errors_pairwise(fct, paramsX, paramsY, mode, printout=False):
    # gather appropriate error
    loc = fct.error_mode_to_dictionary(mode)
    if paramsX not in loc:
        fct.compute_error(paramsX, mode, printout=printout)
    if paramsY not in loc:
        fct.compute_error(paramsY, mode, printout=printout)
    EX = loc[paramsX]
    EY = loc[paramsY]

    if mode == "M":
        print("M", EX)
        legend_labels = [f"M{_}" for _ in range(len(EX))]
    elif fct.env_names is None:
        legend_labels = [f"E{_}" for _ in range(fct.FF.shape[0])]
    else:
        legend_labels = fct.env_names

    plt.close("all")
    fig, axis = plt.subplots(1, 1, figsize=(5, 5))
    axis.tick_params(labelsize=15)
    axis.ticklabel_format(useOffset=False, style="sci", scilimits=(-2, 2))
    mn = min(np.min(EX), np.min(EY))
    mx = max(np.max(EX), np.max(EY))
    buf = (mx - mn) * 0.1
    mn -= buf
    mx += buf
    for _ in range(len(EX)):
        axis.plot(EX[_], EY[_], "o", label=legend_labels[_], ms=7)
    axis.plot([mn, mx], [mn, mx], "C3--")
    axis.set_xlim(mn, mx)
    axis.set_ylim(mn, mx)
    axis.set_xlabel(fct.format_params_nicely(paramsX), fontsize=15)
    axis.set_ylabel(fct.format_params_nicely(paramsY), fontsize=15)
    if fct.E < 100:
        axis.legend(
            fontsize=14,
            bbox_to_anchor=(1.05, 1.0),
            ncol=int(np.ceil(fct.E / 10)),
            frameon=False,
        )
    for sp in ["top", "bottom", "left", "right"]:
        axis.spines[sp].set_linewidth(1.25)
    axis.set_title(f"{fct.error_names[mode]}", fontsize=15)
    plt.show()

    return EX, EY


def get_params_to_include(fct, K, lamb1s, lamb2s, M_lb, W_lb, verbose=False):
    params_to_include = []
    lambs_not_stored = []
    # check if parameter combinations are available, otherwise factorize F for those values
    for lamb1 in lamb1s:
        for lamb2 in lamb2s:
            params_to_include += [
                ("reg", K, (ssd.ptk(lamb1), ssd.ptk(lamb2)), (M_lb, W_lb))
            ]
            params_stored = False
            for params in fct.computed_params(printout=False):
                if (
                    params[0] == "reg"
                    and params[1] == K
                    and params[2][0] == lamb1
                    and params[2][1] == lamb2
                    and params[3] == (M_lb, W_lb)
                ):
                    params_stored = True
                    break
            if not params_stored:
                lambs_not_stored += [[lamb1, lamb2]]

    if len(lambs_not_stored) > 0:
        print("Factorizing F for missing parameter values")
        for lamb1, lamb2 in lambs_not_stored:
            fct.factorize_regularized_range(
                K, [lamb1], [lamb2], M_lb=M_lb, W_lb=W_lb, verbose=verbose
            )
    return params_to_include


def plot_error_v_mod_lamb2_range(
    fct, left_mode, right_mode, K, lamb1s, lamb2s, M_lb, W_lb, printout=False, ave=True
):

    if len(lamb1s) != 1:
        raise ValueError("Number of Lambda_1s should be 1")

    lamb1 = lamb1s[0]

    loc_left = fct.error_mode_to_dictionary(left_mode)
    if right_mode is not None:
        loc_right = fct.error_mode_to_dictionary(right_mode)
    else:
        loc_right = None

    params_to_include = fct.get_params_to_include(K, lamb1s, lamb2s, M_lb, W_lb)

    x_mods = []
    y_error_left = []
    y_error_right = []

    if printout:
        print("Including the following parameters:")
    for params in params_to_include:
        if printout:
            display(Latex(fct.format_params_nicely(params)))
        if ave:
            x_mods.append(fct.ave_mod_per_loci[params])
        else:
            x_mods.append(fct.modularities[params])
        y_error_left.append(np.mean(loc_left[params]))
        if right_mode is not None:
            y_error_right.append(np.mean(loc_right[params]))

    plt.close("all")
    fig, axis = plt.subplots(1, 1, figsize=(4.5, 4.5))
    plt.scatter(x_mods, y_error_left, c="C0")
    if ave:
        axis.set_xlabel(f"ave mod/loci", fontsize=18)
    else:
        axis.set_xlabel(f"Modularity", fontsize=18)
        axis.set_xlim(0, 1)
    axis.set_ylabel(fct.error_names[left_mode], fontsize=18, color="C0")
    axis.tick_params(labelsize=15)
    for sp in ["top", "bottom", "left", "right"]:
        axis.spines[sp].set_linewidth(1.25)
    if fct.init_mode == "MW":
        if ave:
            plt.axvline(x=fct.true_ampl, c="tab:purple")
        else:
            plt.axvline(x=fct.true_modularity, c="tab:purple")

    plt.axhline(y=np.mean(loc_left[("svd", K, None, (0, 0))]), c="C0")

    if right_mode is not None:
        ax = axis.twinx()
        ax.plot(x_mods, y_error_right, "C3s", alpha=0.8)
        ax.set_ylabel(fct.error_names[right_mode], fontsize=18, color="C3")
        ax.tick_params(labelsize=15)
        plt.axhline(y=np.mean(loc_right[("svd", K, None, (0, 0))]), c="C3")
        # plt.legend()
    plt.show()

    return x_mods, y_error_left, y_error_right


def plot_k_err_mod_lamb1_lamb2(
    fct, K, lamb1s, lamb2s, interp=True, contours=True, levels=6, printout=True
):
    keffs = np.zeros((len(lamb1s), len(lamb2s)))
    errs = np.zeros((len(lamb1s), len(lamb2s)))
    amls = np.zeros((len(lamb1s), len(lamb2s)))

    for i, l1 in enumerate(lamb1s):
        for j, l2 in enumerate(lamb2s):
            p = ("reg", K, (ssd.ptk(l1), ssd.ptk(l2)), (0, 0))

            loc = fct.error_mode_to_dictionary("FF")
            if p not in loc:
                fct.compute_error(p, "FF", printout=printout)

            W = fct.W_preds[p]
            M = fct.M_preds[p]
            keff = int(np.ceil(np.sum(M**2)))

            err_FF = np.mean(loc[p])

            avgmodloc = ssd.ave_mod_per_loci(M)

            keffs[i, j] = keff
            errs[i, j] = np.log(err_FF)
            amls[i, j] = avgmodloc

    l1s, l2s = np.meshgrid(lamb1s, lamb2s)

    if interp:
        shading = "gouraud"
    else:
        shading = "auto"

    plt.close("all")

    fig, axis = plt.subplots(1, 3, figsize=(15, 4))

    im = axis[0].pcolormesh(l1s, l2s, keffs.T, shading=shading, vmin=4, vmax=15)
    axis[0].set_xscale("log")
    axis[0].set_yscale("log")
    if contours:
        axis[0].contour(l1s, l2s, errs.T, levels, colors="k", alpha=0.5)

    axis[0].tick_params(labelsize=15)
    axis[0].set_xlabel(r"$\lambda_1$", fontsize=18)
    axis[0].set_ylabel(r"$\lambda_2$", fontsize=18)
    axis[0].set_title(r"$K_{eff}$", fontsize=18)
    fig.colorbar(im, ax=axis[0])

    im = axis[1].pcolormesh(l1s, l2s, errs.T, shading=shading)
    axis[1].set_xscale("log")
    axis[1].set_yscale("log")
    if contours:
        axis[1].contour(l1s, l2s, errs.T, levels, colors="k", alpha=0.5)

    axis[1].tick_params(labelsize=15)
    axis[1].set_xlabel(r"$\lambda_1$", fontsize=18)
    axis[1].set_ylabel(r"$\lambda_2$", fontsize=18)
    axis[1].set_title(r"Log l2 error in $F$", fontsize=18)
    fig.colorbar(im, ax=axis[1])

    im = axis[2].pcolormesh(l1s, l2s, amls.T, shading=shading)
    axis[2].set_xscale("log")
    axis[2].set_yscale("log")
    if contours:
        axis[2].contour(l1s, l2s, errs.T, levels, colors="k", alpha=0.5)

    axis[2].tick_params(labelsize=15)
    axis[2].set_xlabel(r"$\lambda_1$", fontsize=18)
    axis[2].set_ylabel(r"$\lambda_2$", fontsize=18)
    axis[2].set_title(r"Avg. mod. per loci", fontsize=18)
    fig.colorbar(im, ax=axis[2])

    fig.tight_layout()
    plt.show()


def find_key(fct, avl_t, ave_t):
    pminner = 0
    dist = 1e4

    for p in fct.W_preds.keys():
        W = fct.W_preds[p]
        M = fct.M_preds[p]

        avgmodloc = ssd.ave_mod_per_loci(M)
        avgmodenv = ssd.ave_mod_per_env(W)

        if np.sqrt((ave_t - avgmodenv) ** 2 + (avl_t - avgmodloc) ** 2) < dist:
            dist = np.sqrt((ave_t - avgmodenv) ** 2 + (avl_t - avgmodloc) ** 2)
            pminner = p

    return pminner


def get_keff_avl_ave(fct, avl_t, ave_t):
    p = fct.find_key(avl_t, ave_t)
    W = fct.W_preds[p]
    M = fct.M_preds[p]
    avl = ssd.ave_mod_per_loci(M)
    ave = ssd.ave_mod_per_env(W)
    keff = int(np.ceil(np.sum(M**2)))
    return keff, avl, ave


# def plot_error_landscape(fct,K,lamb1s, lamb2s, avls_t = None, aves_t = None, minx = 1.25, maxx = 10, miny = 1.25, maxy = 6.5, vmax = 0.25):
#     keffs = np.zeros((len(lamb1s),len(lamb2s)))
#     errs = np.zeros((len(lamb1s),len(lamb2s)))
#     amls = np.zeros((len(lamb1s),len(lamb2s)))
#     ames = np.zeros((len(lamb1s),len(lamb2s)))
#     nzs = np.zeros((len(lamb1s),len(lamb2s)))

#     for i,l1 in enumerate(lamb1s):
#         for j,l2 in enumerate(lamb2s):
#             p = ("reg", K, (em.ptk(l1), em.ptk(l2)), (0, 0))

#             loc = fct.error_mode_to_dictionary("FF")
#             if p not in loc: fct.compute_error(p, "FF", printout = True)

#             W = fct.W_preds[p]
#             M = fct.M_preds[p]
#             keff = int(np.ceil(np.sum(M**2)))

#             err_FF = np.mean(loc[p])

#             avgmodloc = em.ave_mod_per_loci(M)
#             avgmodenv = em.ave_mod_per_env(W)

#             nzl = np.sum(np.abs(M) > 0)
#             nze = np.sum(np.abs(W) > 0)

#             keffs[i,j] = keff
#             errs[i,j] = err_FF
#             amls[i,j] = avgmodloc
#             ames[i,j] = avgmodenv
#             nzs[i,j] = nzl + nze

#     ames = np.ravel(ames)
#     amls = np.ravel(amls)
#     errs = np.ravel(errs)
#     keffs = np.ravel(keffs)
#     nzs = np.ravel(nzs)

#     points = (amls,ames)
#     values = errs

#     grid_x, grid_y = np.mgrid[minx:maxx:500j, miny:maxy:500j]

#     grid_z0 = griddata(points, values, (grid_x, grid_y), method='linear', fill_value= vmax + 1e-4)

#     plt.close("all")
#     fig,axis = plt.subplots(1,1,figsize = (9,5))

#     lower = Prism_8.mpl_colormap(np.arange(256))
#     upper = np.ones((int(256/4),4))
#     for i in range(3):
#         upper[:,i] = np.linspace(lower[-1,i],1, upper.shape[0])
#     cmap = np.vstack(( lower, upper ))
#     cmap = mpl.colors.ListedColormap(cmap, name='myColorMap', N=cmap.shape[0])

#     im = axis.contourf(grid_z0.T, extent=(minx,maxx,miny,maxy), origin='lower', levels = np.linspace(0,vmax,10),\
#                        vmin = 0,vmax = vmax,cmap = cmap )
#     #axis.scatter(points[0], points[1], c = values, s = 100)

#     axis.tick_params(labelsize = 18)
#     axis.set_xlabel(r"Avg. mod. / locus",fontsize = 20)
#     axis.set_ylabel(r"Avg. mod. / env",fontsize = 20)
#     #axis.set_title(r"$K_{eff}$",fontsize = 18)
#     cb = fig.colorbar(im,ax = axis, ticks =  np.linspace(0,vmax,6))
#     cb.ax.tick_params(labelsize=15)

#     cb.ax.set_ylabel("Error in F", fontsize = 18)

#     keffs = []
#     for i in range(len(avls_t)):
#         keff,avl,ave = get_keff_avl_ave(fct,avls_t[i], aves_t[i])
#         keffs += [keff]
#         axis.text(avl,ave,'%d'%(keff), fontsize = 14)
#         #axis.plot(avls_t[i],aves_t[i],'C%d'%(keff),marker = (keff,2), ms=12, mec = 'C%d'%keff, mew = 1.5)

#     axis.set_xticks(np.arange(np.ceil(minx),np.ceil(maxx)))

#     #plt.grid()
#     fig.tight_layout()
#     plt.show()


def plot_rotation_test(
    fct,
    fct_rot,
    K,
    lamb1s,
    lamb2s,
    xrange,
    yrange,
    rotate,
    fs=16,
    svd_k=None,
    true_line=False,
    save_name=None,
):

    if rotate == "loci":
        rotate = "FO"
        xlabel = "Ave. modules per locus"
        title = "Loci rotation test"
        legend_label = "F after loci rotated"
        if len(lamb1s) != 1:
            raise ValueError(" can only have one value of lamb1 for OF test")

    if rotate == "env":
        rotate = "OF"
        xlabel = "Ave. modules per trait"
        title = "Trait rotation test"
        legend_label = "F after traits rotated"
        if len(lamb2s) != 1:
            raise ValueError(" can only have one value of lamb2 for FO test")

    #  regularization parameters
    params_to_include = []
    for lamb1 in lamb1s:
        for lamb2 in lamb2s:
            params_to_include += [("reg", K, (ssd.ptk(lamb1), ssd.ptk(lamb2)), (0, 0))]

    # plot
    fig, axis = plt.subplots(1, 1, figsize=(5, 5))

    # get values to plot
    F_spars = []
    F_err = []
    F_keffs = []

    rot_spars = []
    rot_err = []
    rot_keffs = []

    for params in params_to_include:
        F_keffs.append(int(np.ceil(np.sum(fct.M_preds[params] ** 2))))
        rot_keffs.append(int(np.ceil(np.sum(fct_rot.M_preds[params] ** 2))))

        if rotate == "FO":
            if true_line:
                true_x = fct.true_ampl
            F_spars.append(fct.ave_mod_per_loci[params])
            rot_spars.append(fct_rot.ave_mod_per_loci[params])
        elif rotate == "OF":
            if true_line:
                true_x = fct.true_ampe
            F_spars.append(fct.ave_mod_per_env[params])
            rot_spars.append(fct_rot.ave_mod_per_env[params])
        F_err.append(np.mean(fct.L2FFs[params]))
        rot_err.append(np.mean(fct_rot.L2FFs[params]))

    # plot
    axis.scatter(F_spars, F_err, label="F")
    axis.scatter(rot_spars, rot_err, c="tab:red", marker="D", label=legend_label)

    if true_line:
        axis.axvline(true_x, color="tab:blue")
    if svd_k is not None:
        if rotate == "FO":
            # print(np.mean(fct.L2FFs[('svd', svd_k, None, (0,0))]))
            axis.axhline(
                np.mean(fct.L2FFs[("svd", svd_k, None, (0, 0))]),
                color="gray",
                linestyle="dashed",
            )
        else:
            axis.axhline(
                np.mean(fct.L2FFs[("svd", svd_k, None, (0, 0))]),
                color="tab:blue",
                linestyle="dashed",
            )
            axis.axhline(
                np.mean(fct_rot.L2FFs[("svd", svd_k, None, (0, 0))]),
                color="tab:red",
                linestyle="dashed",
            )

    axis.grid()
    axis.set_xlim(xrange)
    axis.set_ylim(yrange)
    axis.tick_params(axis="y", labelsize=fs)
    axis.tick_params(axis="x", labelsize=fs)
    axis.set_ylabel("Reconstruction error", fontsize=fs)
    axis.set_xlabel(xlabel, fontsize=fs)
    axis.legend(fontsize=fs*0.8)
    axis.set_title(title, fontsize=fs)

    if save_name is not None:
        plt.savefig(save_name, bbox_inches="tight")

    plt.show()


def plot_rotation_test_w_error(
    fct,
    fct_rots,
    K,
    lamb1s,
    lamb2s,
    xrange,
    yrange,
    rotate,
    fs=16,
    svd_k=None,
    true_line=False,
    save_name=None,
    figsize=(5, 5),
    xticks=None,
    oc="tab:blue",
    rotc="tab:red",
    legend=True,
    labels=True,
):

    if rotate == "loci":
        rotate = "FO"
        xlabel = "Ave. modules / locus"
        title = "Loci rotation test"
        legend_label = "F after loci rotated"
        if len(lamb1s) != 1:
            raise ValueError(" can only have one value of lamb1 for OF test")

    if rotate == "env":
        rotate = "OF"
        xlabel = "Ave. modules / trait"
        title = "Trait rotation test"
        legend_label = "F after traits rotated"
        if len(lamb2s) != 1:
            raise ValueError(" can only have one value of lamb2 for FO test")

    #  regularization parameters
    params_to_include = []
    for lamb1 in lamb1s:
        for lamb2 in lamb2s:
            params_to_include += [("reg", K, (ssd.ptk(lamb1), ssd.ptk(lamb2)), (0, 0))]

    # plot
    fig, axis = plt.subplots(1, 1, figsize=figsize)

    # get values to plot
    F_spars = []
    F_err = []

    rot_spars = []
    rot_err = []

    for params in params_to_include:
        if rotate == "FO":
            if true_line:
                true_x = fct.true_ampl
            F_spars.append(fct.ave_mod_per_loci[params])
            rot_spars.append([fct_rot.ave_mod_per_loci[params] for fct_rot in fct_rots])
        elif rotate == "OF":
            if true_line:
                true_x = fct.true_ampe
            F_spars.append(fct.ave_mod_per_env[params])
            rot_spars.append([fct_rot.ave_mod_per_env[params] for fct_rot in fct_rots])
        F_err.append(np.mean(fct.L2FFs[params]))
        rot_err.append([np.mean(fct_rot.L2FFs[params]) for fct_rot in fct_rots])

    # plot
    # nr = np.sqrt(len(fct_rots))
    nr = 1
    rot_err_mean = [np.mean(_) for _ in rot_err]
    rot_err_se = [np.std(_) / nr for _ in rot_err]

    rot_spars_mean = [np.mean(_) for _ in rot_spars]
    rot_spars_se = [np.std(_) / nr for _ in rot_spars]

    axis.scatter(F_spars, F_err, label="F", c=oc, s=25)
    axis.scatter(
        rot_spars_mean, rot_err_mean, c=rotc, marker="D", s=25, label=legend_label
    )
    axis.errorbar(
        rot_spars_mean,
        rot_err_mean,
        xerr=rot_spars_se,
        yerr=rot_err_se,
        c=rotc,
        marker="D",
        fmt="none",
    )

    if true_line:
        axis.axvline(true_x, color=oc)
    if svd_k is not None:
        if rotate == "FO":
            # print(np.mean(fct.L2FFs[('svd', svd_k, None, (0,0))]))
            axis.axhline(
                np.mean(fct.L2FFs[("svd", svd_k, None, (0, 0))]),
                color=oc,
                linestyle="dashed",
            )
        else:
            axis.axhline(
                np.mean(fct.L2FFs[("svd", svd_k, None, (0, 0))]),
                color=oc,
                linestyle="dashed",
            )
            axis.axhline(
                np.mean(
                    [
                        np.mean(fct_rot.L2FFs[("svd", svd_k, None, (0, 0))])
                        for fct_rot in fct_rots
                    ]
                ),
                color=rotc,
                linestyle="dashed",
            )

    axis.grid()
    axis.set_xlim(xrange)
    axis.set_ylim(yrange)
    if xticks is not None:
        axis.set_xticks(xticks)
    axis.tick_params(axis="y", labelsize=fs)
    axis.tick_params(axis="x", labelsize=fs)
    if labels:
        axis.set_ylabel("Reconstruction error", fontsize=fs)
        axis.set_xlabel(xlabel, fontsize=fs)
        axis.set_title(title, fontsize=fs)
    else:
        axis.set_xticklabels([])
        axis.set_yticklabels([])
    if legend:
        axis.legend(fontsize=fs)

    if save_name is not None:
        plt.savefig(save_name, bbox_inches="tight")

    plt.show()


def plot_solution_space(
    fct,
    K,
    lamb1s,
    lamb2s,
    minx,
    maxx,
    miny,
    maxy,
    vmax,
    vmin,
    scatter=False,
    scatter_restricted=False,
    restrict_in_range=True,
    k_labeled_points=None,
    circled_points=None,
    save_name=None,
    fs=18,
    cp_colors=None,
    labels=True,
    legend=True,
):

    xlabel = r"Ave. Modules / Locus"
    ylabel = r"Ave. Modules / Trait"
    plt.close("all")
    ylen = 8
    xlen = (maxx - minx) / (maxy - miny) * ylen
    fig, axis = plt.subplots(1, 1, figsize=(xlen, ylen))

    keffs = []
    errs = []
    amls = []
    ames = []
    nzs = []

    for i, l1 in enumerate(lamb1s):
        for j, l2 in enumerate(lamb2s):
            p = ("reg", K, (ssd.ptk(l1), ssd.ptk(l2)), (0, 0))

            loc = fct.error_mode_to_dictionary("FF")
            if p not in loc:
                fct.compute_error(p, "FF", printout=True)

            W = fct.W_preds[p]
            M = fct.M_preds[p]
            keff = int(np.ceil(np.sum(M**2)))

            err_FF = np.mean(loc[p])

            avgmodloc = ssd.ave_mod_per_loci(M)
            avgmodenv = ssd.ave_mod_per_env(W)

            nzl = np.sum(np.abs(M) > 0)
            nze = np.sum(np.abs(W) > 0)

            if restrict_in_range == False or (
                err_FF > vmin
                and err_FF < vmax
                and avgmodloc > minx
                and avgmodloc < maxx
                and avgmodenv > miny
                and avgmodenv < maxy
            ):

                if np.isnan(avgmodloc) == False and np.isnan(avgmodenv) == False:
                    amls.append(avgmodloc)
                    ames.append(avgmodenv)
                    nzs.append(nzl + nze)
                    keffs.append(keff)
                    errs.append(err_FF)

    points = (amls, ames)
    values = errs

    grid_x, grid_y = np.mgrid[minx:maxx:500j, miny:maxy:500j]

    grid_z0 = griddata(
        points, values, (grid_x, grid_y), method="linear", fill_value=vmax + 1e-4
    )

    lower = Prism_8.mpl_colormap(np.arange(256))
    upper = np.ones((int(256 / 4), 4))
    for i in range(3):
        upper[:, i] = np.linspace(lower[-1, i], 1, upper.shape[0])
    cmap = np.vstack((lower, upper))
    cmap = mpl.colors.ListedColormap(cmap, name="myColorMap", N=cmap.shape[0])
    # im=axis[l,k].tricontourf(amls, ames, errs, extent=(minx,maxx,miny,maxy), origin='lower', levels = np.linspace(vmin,vmax,10),\
    #               vmin =vmin, vmax = vmax,cmap = cmap )

    im = axis.contourf(
        grid_z0.T,
        extent=(minx, maxx, miny, maxy),
        origin="lower",
        levels=np.linspace(vmin, vmax, 10),
        vmin=vmin,
        vmax=vmax,
        cmap=cmap,
    )
    if scatter:
        amls_ts, ames_ts = [], []
        for _ in range(len(amls)):
            if amls[_] < maxx and ames[_] < maxy:
                if scatter_restricted:
                    if errs[_] > vmax or errs[_] < vmin:
                        continue
                amls_ts.append(amls[_])
                ames_ts.append(ames[_])

        axis.scatter(amls_ts, ames_ts, color="black", s=10, alpha = 0.4)

    axis.tick_params(labelsize=fs)
    if legend:
        colorbar_aspect = [1.04,0,0.05,1]
        cax = axis.inset_axes(colorbar_aspect, transform=axis.transAxes)
        cb = fig.colorbar(im, ax=axis, cax = cax)  # , ticks =  np.linspace(0,vmax,6))
        cb.ax.tick_params(labelsize=fs*0.9)
        arr = np.linspace(vmin, vmax, 10)
        yticklabels = ["%.2f"%a for a in arr]
        cb.ax.set_yticklabels(yticklabels)
        cb.ax.set_ylabel("Reconstruction error", fontsize=fs)

    keffs = []
    if circled_points is not None:
        for _, pt in enumerate(circled_points):
            if cp_colors is not None:
                ec = cp_colors[_]
            else:
                ec = "white"
            p = fct.find_key(pt[0], pt[1])
            keff = int(np.ceil(np.sum(fct.M_preds[p] ** 2)))
            print(
                p,
                fct.ave_mod_per_loci[p],
                fct.ave_mod_per_env[p],
                keff,
                np.mean(loc[p]),
            )
            axis.scatter(
                fct.ave_mod_per_loci[p],
                fct.ave_mod_per_env[p],
                marker="o",
                s=80,
                c="None",
                edgecolors=ec,
                linewidth=2.5,
            )
    if k_labeled_points is not None:
        for pt in k_labeled_points:
            p = fct.find_key(pt[0], pt[1])
            keff = int(np.ceil(np.sum(fct.M_preds[p] ** 2)))
            if fct.ave_mod_per_loci[p] < maxx and fct.ave_mod_per_env[p] < maxy:
                axis.text(
                    fct.ave_mod_per_loci[p],
                    fct.ave_mod_per_env[p],
                    "%d" % (keff),
                    fontsize=20,
                )

    axis.set_yticks(np.arange(np.ceil(miny), np.ceil(maxy)))
    axis.set_xticks(np.arange(np.ceil(minx), np.ceil(maxx)))
    if labels:
        #axis.set_title("Solution space", fontsize=fs)
        axis.set_ylabel(ylabel, fontsize=fs)
        axis.set_xlabel(xlabel, fontsize=fs)
    else:
        axis.set_xticklabels([])
        axis.set_yticklabels([])
    for sp in ['top','bottom','left','right']:
        axis.spines[sp].set_linewidth(1.5)
    axis.grid(linewidth = 2)
    #axis.set_title("SSD Solution Space", fontsize = 22)
    axis.set_aspect("equal")

    if save_name is not None:
        plt.savefig(save_name, bbox_inches="tight")

    plt.show()


def bar_plot_M_W_error(
    fcts,
    name,
    pt,
    k,
    save_name=None,
    fs=16,
    colors=["lightblue", "tab:blue", "navajowhite", "tab:orange"],
    labels=True,
):
    fig, axis = plt.subplots(1, 1, figsize=(3, 6))
    m, w = name[1], name[2]

    fct = fcts[(name, None, None)]
    fct.svd(k)
    K = fct.computed_params(printout=False)[0][1]
    p = fct.find_key(pt[0], pt[1])

    our_M = fct.M_preds[p][np.where(np.abs(fct.M_preds[p].sum(axis=-1)) > 0)]
    our_W = fct.W_preds[p][:, np.where(np.abs(fct.M_preds[p].sum(axis=-1)) > 0)[0]]
    our_M, our_W, _ = fct.permute_M_W(our_M, our_W, [0])

    svd_M = fct.M_preds[("svd", k, None, (0, 0))]
    svd_W = fct.W_preds[("svd", k, None, (0, 0))]
    print(f"Our Reconstruction error: {np.mean(fct.L2FFs[p]):.3f}")
    print(f"SVD Reconstruction error: {np.mean(fct.L2FFs[('svd',k,None,(0,0))]):.3f}")
    our_M_err = np.mean(1 - np.abs(np.sum(fct.M * our_M, axis=-1)))
    svd_M_err = np.mean(1 - np.abs(np.sum(fct.M * svd_M, axis=-1)))
    print(f"Our M err: {np.mean(1-np.abs(np.sum(fct.M * our_M, axis = -1))):.3f}")
    print(f"SVD M err: {np.mean(1-np.abs(np.sum(fct.M * svd_M, axis = -1))):.3f}")
    our_to_flip = 2 * (np.sum(fct.M * our_M, axis=-1) > 0) - 1
    our_W = our_W * our_to_flip
    svd_to_flip = 2 * (np.sum(fct.M * svd_M, axis=-1) > 0) - 1
    svd_W = svd_W * svd_to_flip

    W_cos_our = np.sum(fct.W * our_W, axis=-1) / (
        1e-8
        + np.sqrt(np.sum(np.square(fct.W), axis=-1))
        * np.sqrt(np.sum(np.square(our_W), axis=-1))
    )
    W_cos_svd = np.sum(fct.W * svd_W, axis=-1) / (
        1e-8
        + np.sqrt(np.sum(np.square(fct.W), axis=-1))
        * np.sqrt(np.sum(np.square(svd_W), axis=-1))
    )

    # when both vectors are zero manually make cosine = 1
    W_zero = (np.where(np.sum(np.square(fct.W), axis=-1) == 0))[0]
    our_W_zero = (np.where(np.sum(np.square(our_W), axis=-1) == 0))[0]
    svd_W_zero = (np.where(np.sum(np.square(svd_W), axis=-1) == 0))[0]
    for t in W_zero:
        if t in our_W_zero:
            W_cos_our[t] = 1
        if t in svd_W_zero:
            W_cos_svd[t] = 1

    our_W_err = np.mean(1 - W_cos_our)
    svd_W_err = np.mean(1 - W_cos_svd)
    print(f"Our W cos err: {np.mean(1-W_cos_our):.3f}")
    print(f"SVD W cos err: {np.mean(1-W_cos_svd):.3f}")

    axis.bar(
        [0.6, 1.1, 1.8, 2.3],
        [our_M_err, svd_M_err, our_W_err, svd_W_err],
        width=[0.4, 0.4, 0.4, 0.4],
        color=colors,
    )
    axis.set_xticks([0.6, 1.1, 1.8, 2.3])
    axis.set_yticks(np.linspace(0, 0.4, 9))
    axis.set_ylim(ymin = 0, ymax = 0.4)

    if labels:
        axis.set_title(f"M ~ Bern({m}), W ~ Bern({w})", fontsize=fs)
        axis.set_xticklabels(
            ["SSD M", "SVD M", "SSD W", "SVD W"], rotation=45, fontsize=fs
        )
        axis.set_yticklabels(
            [np.round(x, 2) for x in np.linspace(0, 0.4, 9)], fontsize=fs
        )
        axis.set_ylabel("Mean cos error", fontsize=fs)
    else:
        axis.set_xticklabels([])
        axis.set_yticklabels([])

    if save_name is not None:
        plt.savefig(save_name, bbox_inches="tight")

    plt.show()


def compare_Ws(fcts, name, point, vs=None, save_name=None, hubticks=False):

    print(name)
    fct = fcts[(name, None, None)]
    k = fct.computed_params(printout=False)[0][1]
    pt = point
    p = fct.find_key(pt[0], pt[1])

    keep = list(np.where(np.abs(fct.M_preds[p].sum(axis=-1)) > 0)[0])
    tk = fct.M.shape[0]
    if tk > len(keep):
        mxi = max(keep)
        for _ in range(fct.M.shape[0] - len(keep)):
            keep.append(mxi + _ + 1)

    our_M = fct.M_preds[p][keep]
    our_W = fct.W_preds[p][:, keep]
    our_M, our_W, _ = fct.permute_M_W(our_M, our_W, [0])

    svd_M = fct.M_preds[("svd", k, None, (0, 0))][:tk]
    svd_W = fct.W_preds[("svd", k, None, (0, 0))][:, 0:tk]
    print(f"Our Reconstruction error: {np.mean(fct.L2FFs[p]):.3f}")
    print(f"SVD Reconstruction error: {np.mean(fct.L2FFs[('svd',k,None,(0,0))]):.3f}")
    print(f"Our M err: {np.mean(1-np.abs(np.sum(fct.M * our_M, axis = -1))):.3f}")
    print(f"SVD M err: {np.mean(1-np.abs(np.sum(fct.M * svd_M, axis = -1))):.3f}")
    our_to_flip = 2 * (np.sum(fct.M * our_M, axis=-1) > 0) - 1
    our_W = our_W * our_to_flip
    svd_to_flip = 2 * (np.sum(fct.M * svd_M, axis=-1) > 0) - 1
    svd_W = svd_W * svd_to_flip

    print(
        f"num dropped envs: {len(np.where(np.sum(np.square(our_W), axis = -1)==0)[0])} "
    )
    print(
        f"num true 0 envs: {len(np.where(np.sum(np.square(fct.W), axis = -1)==0)[0])} "
    )

    W_cos_our = np.sum(fct.W * our_W, axis=-1) / (
        1e-8
        + np.sqrt(np.sum(np.square(fct.W), axis=-1))
        * np.sqrt(np.sum(np.square(our_W), axis=-1))
    )
    W_cos_svd = np.sum(fct.W * svd_W, axis=-1) / (
        1e-8
        + np.sqrt(np.sum(np.square(fct.W), axis=-1))
        * np.sqrt(np.sum(np.square(svd_W), axis=-1))
    )
    # when both vectors are zero manually make cosine = 1
    W_zero = (np.where(np.sum(np.square(fct.W), axis=-1) == 0))[0]
    our_W_zero = (np.where(np.sum(np.square(our_W), axis=-1) == 0))[0]
    svd_W_zero = (np.where(np.sum(np.square(svd_W), axis=-1) == 0))[0]
    for t in W_zero:
        if t in our_W_zero:
            W_cos_our[t] = 1
        if t in svd_W_zero:
            W_cos_svd[t] = 1

    print(f"Our W cos err: {np.mean(1-W_cos_our):.3f}")
    print(f"SVD W cos err: {np.mean(1-W_cos_svd):.3f}")

    if vs is None:
        vs = [10 * np.mean(np.abs(_)) for _ in [fct.W.T, our_W, svd_W]]

    fig, axis = plt.subplots(1, 1, figsize=(10, 4))
    axis.imshow(fct.W.T, cmap="PRGn", vmin=-vs[0], vmax=vs[0])
    # plt.colorbar()
    axis.set_title("True W")
    if hubticks:
        axis.set_xticks(np.arange(-0.5, fct.W.T.shape[1] + 1, 5))
        axis.set_yticks([-0.5, 7.5, fct.W.T.shape[0] - 0.5])
        axis.set_xticklabels([])
        axis.set_yticklabels([])
    if save_name is not None:
        fig.savefig(save_name[:-4] + "_true" + save_name[-4:], bbox_inches="tight")
    plt.show()

    fig, axis = plt.subplots(1, 1, figsize=(10, 4))
    axis.imshow(our_W.T, cmap="PRGn", vmin=-vs[1], vmax=vs[1])
    # plt.colorbar()
    axis.set_title("SSD W")
    if hubticks:
        axis.set_xticks(np.arange(-0.5, fct.W.T.shape[1] - 1, 5))
        axis.set_yticks([-0.5, 7.5, fct.W.T.shape[0] - 0.5])
        axis.set_xticklabels([])
        axis.set_yticklabels([])
    if save_name is not None:
        fig.savefig(save_name[:-4] + "_ssd" + save_name[-4:], bbox_inches="tight")
    plt.show()

    fig, axis = plt.subplots(1, 1, figsize=(10, 4))
    axis.imshow(svd_W.T, cmap="PRGn", vmin=-vs[2], vmax=vs[2])
    axis.set_title("SVD W")
    if hubticks:
        axis.set_xticks(np.arange(-0.5, fct.W.T.shape[1] + 1, 5))
        axis.set_yticks([-0.5, 7.5, fct.W.T.shape[0] - 0.5])
        axis.set_xticklabels([])
        axis.set_yticklabels([])
    if save_name is not None:
        fig.savefig(save_name[:-4] + "_svd" + save_name[-4:], bbox_inches="tight")
    plt.show()

    E = len(1 - W_cos_our)
    plt.scatter(range(E), 1 - W_cos_our, alpha=0.5)
    plt.scatter(range(E), 1 - W_cos_svd, alpha=0.5)
    plt.xlabel("env")
    plt.ylabel("cos sim")
    plt.show()

    return fct.W.T, our_W, svd_W

def display_pathways_of_modules_frac(M, pathways, pathway_names, weighted = True, include_NA = False, title = None, figsize = (5,5), colorbar = True, vmax = None):
    if not weighted:
        M = np.abs(M)>0
    else: M = np.abs(M)
    if vmax is None: vmax =1
    pw_weights = ((M@pathways).T/(np.sum(M, axis= -1))).T
    #pw_weights = M@pathways
    print(np.sum(pw_weights, axis= -1))
    labelsize=16
    if not include_NA:
        pw_weights = pw_weights[...,:-1]
        pathway_names = pathway_names[:-1]
    fig,axis = plt.subplots(1,1,figsize = figsize)
    #mm = np.max(pw_weights)
    im = axis.imshow(pw_weights.T,cmap = "Greys", vmin=0 , vmax=vmax)
    if colorbar:
        cb = fig.colorbar(mappable = im)
        cb.ax.tick_params(labelsize = labelsize)
    axis.set_yticks(np.arange(pw_weights.shape[1]))
    axis.set_xticks(np.arange(pw_weights.shape[0]))
    axis.tick_params(labelsize = labelsize)
    axis.set_yticklabels(pathway_names)
    axis.set_ylabel("Pathways",fontsize = labelsize)
    axis.set_xlabel("Modules",fontsize = labelsize)
    for sp in ['top','bottom','left','right']:
        axis.spines[sp].set_linewidth(1.25)
    if title is not None:
        axis.set_title(title, fontsize = labelsize)
    fig.tight_layout()
    plt.show()