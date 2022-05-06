import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import plotly.graph_objects as go
import time
import shutil
import imp
import scipy.stats
import numpy as np
from pyodesys.symbolic import SymbolicSys
from pyodesys.native import native_sys
from scipy.stats import gaussian_kde
import pathos.multiprocessing as mp
from smt.sampling_methods import LHS
import scipy.stats

#import utils.blackbox_git as bb
jmax=150 #changing this requires full recompilation and symbolic derivatives.
tmax=2000

kn_prefix = np.zeros((jmax, jmax, jmax), dtype=np.int16) #
for nc_ in range(jmax):
    for i_ in range(jmax):
        for j_ in range(jmax):
            i = i_ + 1
            j = j_ + 1
            nc = nc_ + 1
            if i <= nc and j <= nc and i + j <= nc:
                kn_prefix[nc_, i_, j_] = 1
            elif i <= nc and j <= nc and nc < (i + j):
                kn_prefix[nc_, i_, j_] = -(i + j - nc - 1)
            elif j <= nc and nc < i and nc < i + j:
                kn_prefix[nc_, i_, j_] = -(j - 1)
            elif i <= nc and nc < j and nc < i + j:
                kn_prefix[nc_, i_, j_] = -(i - 1)
            elif nc < i and nc < j and nc < i + j:
                kn_prefix[nc_, i_, j_] = -(nc - 1)


def install():
    tmax = 10
    kp = 1
    kn = 1
    ka = 0.5
    kd1 = 1
    kd2 = 1
    kd3 = 1
    kd = 10
    m0 = 1000
    nc=4
    e=0
    ec=1.5



    print("Computing analytical derivatives ...")
    neq_coop_tr = SymbolicSys.from_callback(neq_coop_rhs_transient, ny=jmax+2, nparams=8, backend="sympysymengine")
    neq_coop = SymbolicSys.from_callback(neq_coop_rhs, ny=jmax+1, nparams=8, backend="sympysymengine")
    #neq2_isodesic = SymbolicSys.from_callback(neq2_isodesic_rhs, ny=jmax + 1, nparams=6, backend="sympysymengine")
    #neq_isodesic = SymbolicSys.from_callback(neq_isodesic_rhs, ny=jmax + 1, nparams=4, backend="sympysymengine")
    #eq_isodesic = SymbolicSys.from_callback(eq_isodesic_rhs, ny=jmax, nparams=2, backend="sympysymengine")
    #eq_coop = SymbolicSys.from_callback(eq_coop_rhs, ny=jmax, nparams=4, backend="sympysymengine")

    print("Trying to compile ...")

    neq_coop_tr_compiled= native_sys['cvode'].from_other(neq_coop_tr)
    ic = np.zeros(jmax+2)
    ic[0] = m0
    neq_coop_tr_compiled.integrate(np.linspace(0, tmax, 1000), ic, params=(nc, ec, ka, kd1,kd2,kd3,e,kp), integrator="cvode")


    neq_coop_compiled= native_sys['cvode'].from_other(neq_coop)
    ic = np.zeros(jmax+1)
    ic[0] = m0
    neq_coop_compiled.integrate(np.linspace(0, tmax, 1000), ic, params=(nc, ec, ka, kd1,kd2,kd3,e,kp), integrator="cvode")

    #eq_coop_compiled= native_sys['cvode'].from_other(eq_coop)
    ic = np.zeros(jmax)
    ic[0] = m0
    #eq_coop_compiled.integrate(np.linspace(0, tmax, 1000), ic, params=(nc,e,ec,kp), integrator="cvode")

    #neq2_isodesic_compiled = native_sys['cvode'].from_other(neq2_isodesic)
    ic = np.zeros(jmax + 1)
    ic[0] = m0
    #neq2_isodesic_compiled.integrate(np.linspace(0, tmax, 1000), ic, params=(kp, kn, ka, kd1,kd2,kd3), integrator="cvode")

    #neq_isodesic_compiled = native_sys['cvode'].from_other(neq_isodesic)
    ic = np.zeros(jmax + 1)
    ic[0] = m0
    #neq_isodesic_compiled.integrate(np.linspace(0, tmax, 1000), ic, params=(kp, kn, ka, kd), integrator="cvode")

    #eq_isodesic_compiled = native_sys['cvode'].from_other(eq_isodesic)
    ic = np.zeros(jmax)
    ic[0] = m0
    #eq_isodesic_compiled.integrate(np.linspace(0, tmax, 1000), ic, params=(kp, kn), integrator="cvode")
    print("Compilation successful, saving systems!")
    shutil.copytree(neq_coop_tr_compiled._native.mod._binary_path.rpartition("/")[0],
                    "compiled/{}/neq_coop_tr".format(jmax) + "/mod")
    shutil.copytree(neq_coop_compiled._native.mod._binary_path.rpartition("/")[0],
                    "compiled/{}/neq_coop".format(jmax) + "/mod")
    """
    shutil.copytree(eq_coop_compiled._native.mod._binary_path.rpartition("/")[0],
                    "compiled/{}/eq_coop".format(jmax) + "/mod")
    shutil.copytree(neq2_isodesic_compiled._native.mod._binary_path.rpartition("/")[0],
                    "compiled/{}/neq2_isodesic".format(jmax) + "/mod")
    shutil.copytree(neq_isodesic_compiled._native.mod._binary_path.rpartition("/")[0],
                    "compiled/{}/neq_isodesic".format(jmax) + "/mod")
    shutil.copytree(eq_isodesic_compiled._native.mod._binary_path.rpartition("/")[0],
                    "compiled/{}/eq_isodesic".format(jmax) + "/mod")"""


class results_wrapper():
    def __init__(self,xout,yout,stats=None):
        self.xout=xout
        self.yout=yout
        self.stats=stats
    pass


###
class base_integration_wrapper():
    def __init__(self, name):
        import imp

        fobj, filename, data = imp.find_module("_cvode_wrapper",
                                           ["compiled/{}/".format(jmax)+name+"/mod"])
        self.__mod = imp.load_module('_cvode_wrapper', fobj, filename, data)

    def integrate(self, teval, C0, params, integrator='cvode',
                  nsteps=5000, atol=1e-08, rtol=1e-08):

        C0 = np.atleast_2d(C0)
        intern_x = teval  ##I think this needs to be a linspace
        intern_x = np.atleast_2d(np.ascontiguousarray(intern_x, dtype=np.float64))
        P = np.array(np.atleast_2d(params), dtype=np.float64)
        atol = np.array([atol] * len(C0))
        yout, stats = self.__mod.integrate_predefined(y0=np.atleast_2d(C0), xout=np.atleast_2d(
            np.ascontiguousarray(intern_x, dtype=np.float64)),
                                                      params=P, atol=atol,
                                                      rtol=rtol,
                                                     mxsteps=nsteps, dx0=None)

        return results_wrapper(np.squeeze(intern_x), yout[0], stats)

    def integrate_adaptive(self, teval, C0, params, integrator='cvode',
                  nsteps=5000, atol=1e-08, rtol=1e-08):

        C0 = np.atleast_2d(C0)
        P = np.array(np.atleast_2d(params), dtype=np.float64)
        atol = np.array([atol] * len(C0))
        intern_x = teval  ##I think this needs to be a linspace
        intern_x = np.atleast_2d(np.ascontiguousarray(intern_x, dtype=np.float64))
        tout, yout, stats = self.__mod.integrate_adaptive(y0=np.atleast_2d(C0), x0=np.atleast_1d(np.ascontiguousarray(teval[0])),
                                                          xend=np.atleast_1d(np.ascontiguousarray(teval[-1])),
                                                      params=P, atol=atol,
                                                      rtol=rtol,
                                                     mxsteps=nsteps, dx0=None)

        return results_wrapper(np.squeeze(tout), yout[0], stats)


def eq_isodesic_rhs(t, C, p, backend=np):
    kp = p[0]
    kn = p[1]

    dC = np.zeros_like(C) * C  ## 
    for n in range(0, jmax):
        dC[n] = -2 * kp * C[n] * sum(C) + kp * sum(C[:n] * (C[:n][::-1])) + 2 * kn * sum(C[n+1:]) - kn * (n) * C[n]

    return dC
def neq_isodesic_rhs(t, C, p, backend=np):
    kp = p[0]
    kn = p[1]
    ka = p[2]
    kd = p[3]

    d = C[0]

    dC = np.zeros_like(C) * C  ## 
    idx = np.arange(jmax + 1)  # 

    dC[0] = -ka * d + kd * sum(idx * C)  # one extra flop for the first element which is zero
    dC[1] = ka * d - 2 * kp * C[1] * sum(C[1:]) + 2 * kn * sum(C[2:]) - kd * C[1] + 2 * kd * sum(C[2:])
    for n in range(2, jmax + 1):
        dC[n] = -2 * kp * C[n] * sum(C[1:]) + kp * sum(C[1:n] * (C[1:n][::-1])) + 2 * kn * sum(C[n + 1:])\
                - kn * ( n - 1) * C[n] - n * kd * C[n] + 2 * kd * sum(C[n + 1:])
    return dC
def neq2_isodesic_rhs(t, C, p, backend=np):
    kp = p[0]
    kn = p[1]
    ka = p[2]
    kd1 = p[3]
    kd2 = p[4]
    kd3 = p[5]

    d = C[0]

    dC = np.zeros_like(C) * C  ## HACK

    idx = np.arange(jmax + 1)

    dC[0] = - ka*d + kd1*C[1] + 2*kd2*sum(C[2:]) + kd3*sum(idx[1:-2]*C[3:])
    dC[1] = -2*kp*C[1]*sum(C[1:]) + 2*kn*sum(C[2:]) - kd1*C[1] + 2*kd2*C[2] + 2*kd3*sum(C[3:]) + ka*d
    for n in range(2, jmax):
        dC[n] =  -2*kp*C[n]*sum (C[1:]) + kp*sum(C[1:n]*(C[1:n][::-1]))  - kn * (n-1) * C[n]  + 2*kn*sum(C[n+1:]) \
                 - 2*kd2*(C[n]-C[n+1])  - kd3*(n-2)*C[n] + 2*kd3*sum(C[n+2:]) #using that sum of empty slices is 0
    n = jmax
    dC[n] =  -2*kp*C[n]*sum (C[1:]) + kp*sum(C[1:n]*(C[1:n][::-1]))  - kn * (n-1) * C[n]  + 2*kn*sum(C[n+1:]) \
             - 2*kd2*(C[n]-0)  - kd3*(n-2)*C[n] + 2*kd3*sum(C[n+2:])

    return dC
def eq_coop_rhs(t, C, p, backend=np):

    nc = 4# p[0]
    ec = p[1]
    e = p[2]
    kp = p[3]



    dC = np.zeros_like(C) * C  
    u0 = np.zeros(jmax**2) *e

    #u0[:nc] = np.arange(nc) * ec  # <= nc, e.g. if nc = 2, this will get points 0 and 1, aka 1 and 2. arnage for nc-1
    #u0[nc:] = np.arange(1, jmax ** 2 + 1 - nc) * e + (
    #            nc - 1) * ec  # for n = 3 at nc=2, this will give a value of 1, with a maximum value of jmax-nc

    for n in range(jmax**2):
        if n < nc:#<= nc, e.g. if nc = 2, this will get points 0 and 1, aka 1 and 2. arnage for nc-1
            u0[n] = n*ec
        else:
            u0[n] = n * e + (
                        nc - 1) * ec  # for n = 3 at nc=2, this will give a value of 1, with a maximum value of jmax-nc


    kn = np.zeros((jmax, jmax))*e
    idx = np.arange(jmax) #j, but stritctly for indexing! since it's offset by 1.

    for i in range(jmax):
        for j in range(jmax):
            kn[i, j] = kp * backend.exp(u0[i + j] - u0[i] - u0[j])
    for n in range(jmax):
        dC[n] = -2*kp*C[n]*sum(C) + kp*sum(C[:n] * (C[:n][::-1])) - C[n]*sum(kn[idx[:n][::-1],idx[:n]]) + 2*sum(kn[n,0:-1-n]*C[n+1:])
    return dC
def neq_coop_rhs(t,C,p,backend=np):
    """
    :param t:
    :param C:
    :param p: nc, ec,ka,kd1,kd2,kd3,[e,kp] #scaled
    :param backend:
    :return:
    """
    # Scaled by k+ -> 1 and epsilon -> 0
    kp = p[7]
    e = p[6]
    ec = p[1]
    nc = 2 # p[0]
    ka = p[2]
    kd1 = p[3]
    kd2 = p[4]
    kd3 = p[5]

    dC = np.zeros_like(C) * C  ## HACK
    idx = np.arange(jmax + 1) #len of monomer
    kn = np.zeros((jmax,jmax)) * ec #HACK to make symbolic
    for i in range(jmax):
        for j in range(jmax):
           kn[i,j] = kp*backend.exp(e + ( ec - e) * kn_prefix[nc//1 - 1,i,j])

    d = C[0]


    dC[0] = - ka*d + kd1*C[1] + 2*kd2*sum(C[2:]) + kd3*sum(idx[1:-2]*C[3:])
    dC[1] = -2*kp*C[1]*sum(C[1:]) + 2*sum(kn[0,:-1] * C[2:]) - kd1*C[1] + 2*kd2*C[2] + 2*kd3*sum(C[3:]) + ka*d
    for n in range(2, jmax):
        dC[n] =  -2*kp*C[n]*sum (C[1:]) + kp*sum(C[1:n]*(C[1:n][::-1]))  - sum(kn[np.arange(n-1-1,-1,-1),np.arange(1-1,n-1)] ) * C[n]  + 2*sum(kn[:jmax-n,n-1]*C[n+1:]) \
                 - 2*kd2*(C[n]-C[n+1])  - kd3*(n-2)*C[n] + 2*kd3*sum(C[n+2:]) #using that sum of empty slices is 0
    n = jmax
    dC[n] =      -2*kp*C[n]*sum (C[1:]) + kp*sum(C[1:n]*(C[1:n][::-1]))      - sum(kn[np.arange(n-1-1,-1,-1),np.arange(1-1,n-1)] ) * C[n]  + 2*sum(kn[:jmax-n,n-1]*C[n+1:])  \
             - 2*kd2*(C[n]-0)  - kd3*(n-2)*C[n] + 2*kd3*sum(C[n+2:])

    return dC

def neq_coop_rhs_noleak(t,C,p,backend=np):
    """
    :param t:
    :param C:
    :param p: nc, ec,ka,kd1,kd2,kd3,[e,kp] #scaled
    :param backend:
    :return:
    """
    # Scaled by k+ -> 1 and epsilon -> 0
    kp = p[7]
    e = p[6]
    ec = p[1]
    nc = 2 # p[0]
    ka = p[2]
    kd1 = p[3]
    kd2 = p[4]
    kd3 = p[5]

    dC = np.zeros_like(C) * C  ## HACK
    idx = np.arange(jmax + 1) #len of monomer
    kn = np.zeros((jmax,jmax)) * ec #HACK to make symbolic
    for i in range(jmax):
        for j in range(jmax):
           kn[i,j] = kp*backend.exp(e + ( ec - e) * kn_prefix[nc//1 - 1,i,j])

    d = C[0]


    dC[0] = - ka*d + kd1*C[1] + 2*kd2*sum(C[2:]) + kd3*sum(idx[1:-2]*C[3:])
    dC[1] = -2*kp*C[1]*sum(C[1:]) + 2*sum(kn[0,:-1] * C[2:]) - kd1*C[1] + 2*kd2*C[2] + 2*kd3*sum(C[3:]) + ka*d
    for n in range(2, jmax):
        dC[n] =  -2*kp*C[n]*sum (C[1:]) + kp*sum(C[1:n]*(C[1:n][::-1]))  - sum(kn[np.arange(n-1-1,-1,-1),np.arange(1-1,n-1)] ) * C[n]  + 2*sum(kn[:jmax-n,n-1]*C[n+1:]) \
                 - 2*kd2*(C[n]-C[n+1])  - kd3*(n-2)*C[n] + 2*kd3*sum(C[n+2:]) #using that sum of empty slices is 0
    n = jmax
    dC[n] =      -2*kp*C[n]*sum (C[1:]) + kp*sum(C[1:n]*(C[1:n][::-1]))      - sum(kn[np.arange(n-1-1,-1,-1),np.arange(1-1,n-1)] ) * C[n]  + 2*sum(kn[:jmax-n,n-1]*C[n+1:])  \
             - 2*kd2*(C[n]-0)  - kd3*(n-2)*C[n] + 2*kd3*sum(C[n+2:])

    return dC


def hermans_rhs(t,C,p,backend=np):
    k_frag, k_nuc, k_red, k_ox = p
    JMAX = jmax  # int(jmax)
    nc = 2

    dC = np.zeros_like(C)*C ## HACK


    ## This will allow the expression below to work as intended

    m = C[-2]
    r = C[-1]

    ## Build some prefix sums
    ## https://stackoverflow.com/questions/16541618/perform-a-reverse-cumulative-sum-on-a-numpy-array
    sC = np.cumsum(C[:-2][::-1])[::-1]  ##we do a lot of sums for N to inf of the array

    S1 = 0.
    S2 = 0.
    S3 = 0.

    for j in range(nc, JMAX + 1):
        dC[j] = 2 * m * (C[j - 1] - C[j]) + 2 * k_red * (C[j + 1] - C[j]) - k_frag * ((j - 1) * C[j] - 2 * sC[j + 1])

        S1 += j * (C[j - 1] - C[j])
        S2 += j * ((j - 1) * C[j] - 2 * sC[j + 1])
        S3 += j * (C[j + 1] - C[j])

    dC[nc] += k_nuc * m ** nc  ## Nucleation

    dC[-2] = k_ox * r - nc * k_nuc * (m ** nc) - 2 * m * S1 + k_frag * S2  # m
    dC[-1] = -k_ox * r - 2 * k_red * S3  # r

    return dC







def load_systems():
    eq_isodesic = None #base_integration_wrapper("eq_isodesic")
    neq_isodesic = None #base_integration_wrapper("neq_isodesic")
    neq2_isodesic = None #base_integration_wrapper("neq2_isodesic")
    eq_coop = None #base_integration_wrapper("eq_coop")
    neq_coop = None #base_integration_wrapper("neq_coop")
    neq_coop_tr = base_integration_wrapper("neq_coop_tr")

    return eq_isodesic,neq_isodesic,neq2_isodesic,eq_coop,neq_coop,neq_coop_tr




def neq_coop_rhs_transient(t,C,p,backend=np):
    """
    :param t:
    :param C:
    :param p: nc, ec,ka,kd1,kd2,kd3,[e,kp] #scaled
    :param backend:
    :return:
    """
    # Scaled by k+ -> 1 and epsilon -> 0
    kp = p[7]
    e = p[6]
    ec = p[1]
    nc = 2# p[0]
    ka = p[2]
    kd1 = p[3]
    kd2 = p[4]
    kd3 = p[5]

    dC = np.zeros_like(C) * C  ## HACK
    idx = np.arange(jmax + 1) #len of monomer
    kn = np.zeros((jmax,jmax)) * ec #HACK to make symbolic
    for i in range(jmax):
        for j in range(jmax):
           kn[i,j] = kp*backend.exp(e + ( ec - e) * kn_prefix[nc//1 - 1,i,j])


    fuel = C[0] #done
    d = C[1] #done
    dC[0] = - ka*d*fuel ##fuel

    dC[1] = - ka*d*fuel + kd1*C[2] + 2*kd2*sum(C[3:]) + kd3*sum(idx[1:-2]*C[4:]) #done

    dC[2] = -2*kp*C[2]*sum(C[2:]) + 2*sum(kn[:-1,0] * C[3:]) - kd1*C[2] + 2*kd2*C[3] + 2*kd3*sum(C[4:]) + ka*d*fuel #done
    for n in range(2, jmax):
        dC[n+1] =  -2*kp*C[n+1]*sum(C[2:]) + kp*sum(C[2:n+1]*(C[2:n+1][::-1]))  - sum(kn[np.arange(n-1-1,-1,-1),np.arange(1-1,n-1)] ) * C[n+1]  + 2*sum(kn[:jmax-n,n-1]*C[n+2:]) \
                 - 2*kd2*(C[n+1]-C[n+2])  - kd3*(n-2)*C[n+1] + 2*kd3*sum(C[n+3:]) #using that sum of empty slices is 0
    n = jmax
    dC[n+1] =      -2*kp*C[n+1]*sum (C[2:]) + kp*sum(C[2:n+1]*(C[2:n+1][::-1])) - sum(kn[np.arange(n-1-1,-1,-1),np.arange(1-1,n-1)] ) * C[n+1]  + 2*sum(kn[:jmax-n,n-1]*C[n+2:])  \
             - 2*kd2*(C[n+1]-0)  - kd3*(n-2)*C[n+1] + 2*kd3*sum(C[n+3:])

    return dC



def load_optim(name):
    with open(name) as cf:
        cf.readline()
        res = cf.readline()
        res = -float(res.split(",")[-1])
    return res


def query_neighbors(tree, x, k):
    return tree.query(x, k=k + 1, p=float('inf'), n_jobs=-1)[0][:, k]


def smooth_grid(x):
    X, Y = np.meshgrid(np.linspace(-6, 0, 101), np.linspace(-6, 0, 101))
    return np.reshape(gaussian_kde(np.log10(x.T))(np.array([X.flatten(), Y.flatten()])), (101, 101))


def plot_pairwise_ccf(*args, **kwargs):
    """
    args should be either 1 2d array or 2 1d arrays. (or lists)
    kwargs can contain:
        color (must be tuple or list, no names!)
        sample1_name
        sample2_name
        alpha=base alpha to use, 0-1 default 0.5
        fig_data - list of 4 items including the figure, and 3 plotting axis. return of this function (to allow chaining)

    """

    if len(args) == 1:
        # print "detected 2d input"
        dens_1 = np.nansum(args[0], axis=1)
        dens_2 = np.nansum(args[0], axis=0)
        dens_1 = dens_1 / float(np.nansum(dens_1))
        dens_2 = dens_2 / float(np.nansum(dens_2))
        array_2d = args[0]
        array_2d_max = np.nanmax(array_2d)
        grid_size = np.shape(array_2d)[0]

    elif len(args) != 2:
        print("too many or too few arguments, expected 1 or 2")
        return -1
    else:
        dens_1 = np.array(args[0]) / float(np.sum(args[0]))
        dens_2 = np.array(args[1]) / float(np.sum(args[1]))

        array_2d = np.atleast_2d(np.atleast_2d(dens_1)) * np.atleast_2d(dens_2).T
        array_2d_max = np.amax(array_2d)
        grid_size = np.shape(dens_1)[0]

    [[xmin, xmax], [ymin, ymax]] = kwargs.get("lims", [[0, 1], [0, 1]])
    lims = kwargs.get("lims", [[0, 1], [0, 1]])

    """This needs to be uncommented adjusted if you want to use colors """
    # tableau40 = self._get_fixed_colors(slice(None,None,None)) #Any tuple of colors will do here...
    # Scale the RGB values to the [0, 1] range,which is the format matplotlib accepts.
    # for i in range(len(tableau40)):
    #    r, g, b = tableau40[i]
    #    tableau40[i] = (r / 255., g / 255., b / 255.)
    """"""

    """ Dictionary of cluster to n muatations, you porbably don't need this."""
    # num_assigned=collections.Counter(results["assign"])
    """"""
    if "fig_data" in kwargs:
        fig, ax, axt, axr = kwargs["fig_data"]

    else:
        fig = plt.figure()

        fig.subplots_adjust(wspace=0.015, hspace=0.025)
        gs = gridspec.GridSpec(2, 2, width_ratios=[3, 1], height_ratios=[1, 4])

        ax = plt.subplot(gs[1, 0])

        axr = plt.subplot(gs[1, 1], sharey=ax, frameon=False, xlim=(0, 1), ylim=(ymin, ymax))

        axt = plt.subplot(gs[0, 0], sharex=ax, frameon=False, xlim=(xmin, xmax), ylim=(0, 1))

    # percent=num_assigned[1+i]/np.median(num_assigned.values())*2

    # if percent > 1:
    #   percent = 1 #fades small clusters

    """ This might crash - feel free to set a color or use the tab40 dictionary"""
    clust_color = kwargs["color"] if "color" in kwargs else (
    0, 0, 0)  # ax._get_lines.color_cycle.next() #tableau40[i+1]
    """" """

    alpha = kwargs["alpha"] if "alpha" in kwargs else 0.75
    n_levels = 10
    cmap = [list(clust_color) + [alpha * (1. / n_levels * x)] for x in
            range(n_levels)]  # This sets the color levels, so whatever is used for color must be a 3 length tuple
    cmap2 = []
    for ixc, c in enumerate(cmap):
        if ixc % 2 == 1:
            cmap2.append(clust_color)
        else:
            cmap2.append([1., 1., 1., 0.])

    # Create scatter plot

    grid1 = np.linspace(lims[0][0], lims[0][1], grid_size)  # [::5]
    grid2 = np.linspace(lims[1][0], lims[1][1], grid_size)
    ax.contour(grid2, grid1, array_2d / array_2d_max, levels=n_levels, colors=cmap2, linewidths=1.)
    ax.contourf(grid2, grid1, array_2d / array_2d_max, levels=n_levels, colors=cmap)

    # Create Y-marginal (right)
    axt.fill_between(grid2, dens_2, alpha=alpha * 0.25, color=clust_color)
    axt.plot(grid2, dens_2, color=clust_color, lw=1, alpha=alpha)
    # Create X-marginal (top)
    axr.fill_betweenx(grid1, dens_1, alpha=alpha * 0.25, color=clust_color)
    axr.plot(dens_1, grid1, color=clust_color, lw=1, alpha=alpha)

    axt.axis('off')
    axr.axis('off')

    # Bring the marginals closer to the scatter plot

    """ Annotations for number of mutations in the cluster - feel free to return if you're using this"""
    # ax.annotate(num_assigned[1+i],(ccf_dens[i][n[0]].argmax()/float(grid_size),ccf_dens[i][n[1]].argmax()/float(grid_size)),color=map(lambda x:max([x-0.2,0]),clust_color))
    """"""

    ax.set_xlabel(kwargs["sample1_name"] if "sample1_name" in kwargs else "sample1")
    ax.set_ylabel(kwargs["sample2_name"] if "sample2_name" in kwargs else "sample2")

    axr.set_ylim([0, 1])
    axt.set_xlim([0, 1])

    return [fig, ax, axt, axr]

    # axr.cla()
    # axt.cla()
    # ax.cla()


import glob
import matplotlib.gridspec as gs

colors = np.array([[0, 99, 230],
                   [169, 213, 58],
                   [255, 168, 54],
                   [239, 58, 176],
                   [0, 163, 99],
                   [158, 34, 31],
                   [1, 156, 157],
                   [255, 168, 213]]) / 255.
from scipy.spatial import Delaunay
import numpy as np


def alpha_shape(points, alpha, only_outer=True):
    """
    Compute the alpha shape (concave hull) of a set of points.
    :param points: np.array of shape (n,2) points.
    :param alpha: alpha value.
    :param only_outer: boolean value to specify if we keep only the outer border
    or also inner edges.
    :return: set of (i,j) pairs representing edges of the alpha-shape. (i,j) are
    the indices in the points array.
    """
    assert points.shape[0] > 3, "Need at least four points"

    def add_edge(edges, i, j):
        """
        Add an edge between the i-th and j-th points,
        if not in the list already
        """
        if (i, j) in edges or (j, i) in edges:
            # already added
            assert (j, i) in edges, "Can't go twice over same directed edge right?"
            if only_outer:
                # if both neighboring triangles are in shape, it's not a boundary edge
                edges.remove((j, i))
            return
        edges.add((i, j))

    tri = Delaunay(points)
    edges = set()
    # Loop over triangles:
    # ia, ib, ic = indices of corner points of the triangle
    for ia, ib, ic in tri.vertices:
        pa = points[ia]
        pb = points[ib]
        pc = points[ic]
        # Computing radius of triangle circumcircle
        # www.mathalino.com/reviewer/derivation-of-formulas/derivation-of-formula-for-radius-of-circumcircle
        a = np.sqrt((pa[0] - pb[0]) ** 2 + (pa[1] - pb[1]) ** 2)
        b = np.sqrt((pb[0] - pc[0]) ** 2 + (pb[1] - pc[1]) ** 2)
        c = np.sqrt((pc[0] - pa[0]) ** 2 + (pc[1] - pa[1]) ** 2)
        s = (a + b + c) / 2.0
        area = np.sqrt(s * (s - a) * (s - b) * (s - c))
        circum_r = a * b * c / (4.0 * area)
        if circum_r < alpha:
            add_edge(edges, ia, ib)
            add_edge(edges, ib, ic)
            add_edge(edges, ic, ia)
    return edges

def find_edges_with(i, edge_set):
    i_first = [j for (x, j) in edge_set if x == i]
    i_second = [j for (j, x) in edge_set if x == i]
    return i_first, i_second


def stitch_boundaries(edges):
    edge_set = edges.copy()
    boundary_lst = []
    while len(edge_set) > 0:
        boundary = []
        edge0 = edge_set.pop()
        boundary.append(edge0)
        last_edge = edge0
        while len(edge_set) > 0:
            i, j = last_edge
            j_first, j_second = find_edges_with(j, edge_set)
            if j_first:
                edge_set.remove((j, j_first[0]))
                edge_with_j = (j, j_first[0])
                boundary.append(edge_with_j)
                last_edge = edge_with_j
            elif j_second:
                edge_set.remove((j_second[0], j))
                edge_with_j = (j, j_second[0])  # flip edge rep
                boundary.append(edge_with_j)
                last_edge = edge_with_j

            if edge0[0] == last_edge[1]:
                break

        boundary_lst.append(boundary)
    return boundary_lst

def PolyArea(x, y):
    return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))


import h5py

def get_status(X):


    npoints=1001
    #ic_pos = X[0]
    tmax=10#10**X[0]

    p = 10 ** np.array(X[2:])
    par=np.zeros(8) #nc,ec,ka,kd1,kd2,kd3,[e,kp]
    par[0]=2
    par[1] = X[1]
    par[2:6] = p
    #par[3] = 0 #kd1 = 0
    par[6] = 0
    par[7] = 1


    ic = np.zeros(jmax + 1)
    ic[0] = 10**X[0]# 0 # p[1]
    #counts = scipy.stats.norm(loc=ic_pos, scale=2).pdf(np.linspace(1, jmax, jmax))
    #counts = counts / sum(counts * np.arange(1, jmax + 1))
    ic[1:]  = 0#counts * 100
    #atol = np.array([1e-10] * len(C0))
    #rtol = 1e-4
    #nsteps = 1000

    try:
        res = neq_coop_compiled.integrate(np.linspace(0, tmax, npoints), ic, params=par,
                                          integrator="cvode",atol=1e-5,rtol=1e-3)
        yo = res.yout
        #idx= np.arange(0,jmax + 1)
        #idx[0] = 1
        #PI = np.nansum(res.yout.T * idx[:, None] ** 2, axis=0) / np.nansum(res.yout.T, axis=0)
        #M = res.yout.T[0]

        #yo = np.vstack([M,PI]).T

    except:
        print("!")#, end="", flush=True)
        return 1

    try:
        if max(yo[500:, -4]) > 5e-8: return 0 #leak hack
        points = yo[500:, 0:2]
        edges = alpha_shape(points, 10.)
        Area = PolyArea(*np.array([points[x[0], :] for x in stitch_boundaries(edges)[0]]).T)
        print(".")#, end="", flush=True)

        return -Area

    except Exception as e:
        print("0")#, end="", flush=True)
        return 0


def get_integrated_res(X):


    npoints=1001
    #ic_pos = X[0]
    tmax=10#10**X[0]

    p = 10 ** np.array(X[3:])
    par=np.zeros(8) #nc,ec,ka,kd1,kd2,kd3,[e,kp]
    par[0]=2
    par[1] = X[2]
    par[2:6] = p
    #par[3] = 0 #kd1 = 0
    par[6] = 0
    par[7] = 1


    ic = np.zeros(jmax + 2)
    ic[0] = 10**X[0]# 0 # p[1]
    ic[1] = 10**X[1]# 0 # p[1]
    #ic_pos = 25
    #counts = scipy.stats.norm(loc=ic_pos, scale=2).pdf(np.linspace(1, jmax, jmax))
    #ic[2:]  = (10**X[1]) *counts / sum(counts * np.arange(1, jmax + 1))

    try:
        res = neq_coop_tr.integrate(np.linspace(0, tmax, npoints), ic, params=par,
                                          integrator="cvode",atol=1e-8,rtol=1e-8,nsteps=5000)
        yo = res.yout
        #idx= np.arange(0,jmax + 1)
        #idx[0] = 1
        #PI = np.nansum(res.yout.T * idx[:, None] ** 2, axis=0) / np.nansum(res.yout.T, axis=0)
        #M = res.yout.T[0]

        #yo = np.vstack([M,PI]).T

    except:
        yo = np.zeros((1001,jmax+1)) -1

    return yo

def get_integrated_res_adaptive(X):


    #ic_pos = X[0]
    tmax=10#10**X[0]


    p = 10 ** np.array(X[2:])
    #    :param p: nc, ec,ka,kd1,kd2,kd3,[e,kp] #scaled

    par=np.zeros(8) #nc,ec,ka,kd1,kd2,kd3,[e,kp]
    par[0]=2
    par[1] = X[1]
    par[2:6] = p
    #par[3] = 0 #kd1 = 0
    par[6] = 0
    par[7] = 1


    ic = np.zeros(jmax + 1)
    ic[0] = 10**X[0]# 0 # p[1]
    #ic[1] = 10**X[1]# 0 # p[1]
    #ic_pos = 25
    #counts = scipy.stats.norm(loc=ic_pos, scale=2).pdf(np.linspace(1, jmax, jmax))
    #ic[2:]  = (10**X[1]) *counts / sum(counts * np.arange(1, jmax + 1))

    try:
        res = neq_coop.integrate(np.linspace(0, tmax, npoints), ic, params=par,
                                          integrator="cvode",atol=1e-8,rtol=1e-8,nsteps=10000)
        yo = res.yout
        to = res.xout
        #idx= np.arange(0,jmax + 1)
        #idx[0] = 1
        #PI = np.nansum(res.yout.T * idx[:, None] ** 2, axis=0) / np.nansum(res.yout.T, axis=0)
        #M = res.yout.T[0]

        #yo = np.vstack([M,PI]).T

    except Exception as e:
        print(e)
        yo = np.zeros((2,jmax+2)) -1
        to = np.zeros(2) -1

    return to,yo



install()


#neq_coop_tr = base_integration_wrapper("neq_coop_tr")

neq_coop =  base_integration_wrapper("neq_coop")
"""
eq_isodesic_compiled, neq_isodesic_compiled, neq2_isodesic_compiled, eq_coop_compiled, neq_coop_compiled = load_systems()
get_integrated_res([0.,0.,0.,0,0,0.])
"""
pool=mp.Pool()

#design_ranges=np.array([[0,4],[20,32],
#                        [-2,0.5],[-32,0],
#                        [-32,5],[-1,3]])


design_ranges=np.array([[2,4],[20,32],#nc,ec,ka,kd1,kd2,kd3,[e,kp]
                        [-2,2],[-2,2],
                        [-1, 4],[-1,3]])

sampling = LHS(xlimits=design_ranges,criterion="c")
num=500000
npoints = 1001
nbatches=5000
exp_list=sampling(num)

import time
startt=time.perf_counter()
with h5py.File("noneq_coop_lhs_nc2_v2wm0_partialres_{}_5e6_zoom.hdf5".format(num),"w") as of:
    for batch in range(nbatches):
        batch_size=num//nbatches
        batchres=[]
        #for pt in exp_list[batch_size*batch:batch_size*(batch+1)]:
        #    batchres.append(get_integrated_res(pt))

        #pts = f['pts_{}'.format(batch)]

        batchres=list(map(get_integrated_res_adaptive, exp_list[batch_size*batch:batch_size*(batch+1)]))
        endc=np.array([x[1][-1,:] for x in batchres])


        all_batchres=np.zeros((batch_size,npoints,5))
        all_times=np.zeros((batch_size,npoints))
        idx_arr=np.arange(jmax)


        for bi,batchr in enumerate(batchres):

            all_batchres[bi,:len(batchr[0]),:]=batchr[1][:,:5]
            all_times[bi,:len(batchr[0])]=batchr[0]

        try:
            of.create_dataset(name='res_{}_end'.format(batch),
                              data=endc)
        except:
            print("fail storing!")
        try:
            of.create_dataset(name='res_{}_conc'.format(batch),
                          data=all_batchres,compression="lzf")
        except:
            print("fail storing!")
        try:
            of.create_dataset(name='res_{}_times'.format(batch),
                          data=all_times,compression="lzf")
        except:
            print("fail storing!")
        try:
            of.create_dataset(name='pts_{}'.format(batch),
                              data=exp_list[batch_size*batch:batch_size*(batch+1)])
        except:
            print("fail storing!")

        print(batch)#,end=".",flush=True)

print("##############")
print(time.perf_counter()-startt)
res=pool.map(get_status,exp_list)

"""
#np.savetxt("results/noneq_coop_lhs_nc2_v2wm0_fullmetric_{}_res.txt".format(num),res)
#np.savetxt("results/noneq_coop_lhs_nc2_v2wm0_fullmetric_{}_pts.txt".format(num),exp_list)
#print(res[np.argmin(res)],exp_list[np.argmin(res)])

#"""
"""
bb.search_min(get_status,[[0,6],[0,32],
                        [-32,32],[-32,32],
                        [-32,32],[-32,32]],10000,16,"results/noneq_coop_bb_nc2_v2wm0_10000.csv")
#"""
"""
design_ranges=np.array([[0,4],[0,32],
                        [-32,5],[-32,32],
                        [-32,32],[-32,32]])

def __optim_pso(opt_f, design_ranges):
    from pyswarm import pso
    bounds = np.array(design_ranges)

    lb = [x[0] for x in bounds]
    ub = [x[1] for x in bounds]
    xopt, fopt, all_pos, all_obj = pso(opt_f, lb, ub, maxiter=100,swarmsize=1000,
                                       minstep=np.nan, minfunc=np.nan, detail=True)
    return xopt, all_pos, all_obj

num=1000
xopt, all_pos, all_obj = __optim_pso(get_status,design_ranges)
print(xopt)
np.savetxt("results/noneq_coop_pso_nc2_v2wm0_fullmetric_long_{}_res.txt".format(num),all_obj)
np.savetxt("results/noneq_coop_pso_nc2_v2wm0_fullmetric_long_{}_pts.txt".format(num),np.vstack(all_pos))
#"""

