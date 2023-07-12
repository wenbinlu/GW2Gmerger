import numpy as np
from math import sqrt, sin, cos, asin, pi, acos, log10, floor, exp
import matplotlib as mpl
from scipy import interpolate
import matplotlib.pylab as pl
import cst


def mplchange():
    mpl.rcParams.update({'font.size': 30})
    mpl.rcParams['xtick.major.size'] = 7
    mpl.rcParams['xtick.major.width'] = 2
    mpl.rcParams['xtick.minor.size'] = 4
    mpl.rcParams['xtick.minor.visible'] = True
    mpl.rcParams['xtick.minor.width'] = 1
    mpl.rcParams['ytick.major.size'] = 7
    mpl.rcParams['ytick.major.width'] = 2
    mpl.rcParams['ytick.minor.size'] = 4
    mpl.rcParams['ytick.minor.width'] = 1
    mpl.rcParams['ytick.minor.visible'] = True
    mpl.rcParams['xtick.major.pad'] = 10
    mpl.rcParams['ytick.major.pad'] = 6
    mpl.rcParams['pdf.fonttype'] = 42
    mpl.rcParams['ps.fonttype'] = 42
    mpl.rcParams['xtick.direction'] = 'in'
    mpl.rcParams['ytick.direction'] = 'in'
    mpl.rcParams['xtick.top'] = True
    mpl.rcParams['ytick.right'] = True
    # mpl.rcParams['font.family'] = 'sans-serif'
    # mpl.rcParams["font.family"] = "DejaVu Sans"
    mpl.rcParams["mathtext.fontset"] = "cm"


def Gamma(b):
    return (np.sqrt(1-b**2))**-1


def Beta(g):
    return np.sqrt(1-1/g**2)


def read_1D_table(fdir, fname, nbegin):   # read x-y tabel
    f = open(fdir + fname + '.txt', 'r')
    f.seek(0)  # go to the beginning
    nbeg = int(max(nbegin, 0))
    data = f.read().split('\n')[nbeg:]  # all rows
    Nx = len(data)
    xarr = np.empty(Nx)
    yarr = np.empty(Nx)
    for i in range(Nx):
        row = data[i].split('\t')
        xarr[i] = row[0]
        yarr[i] = row[1]
    f.close()
    return xarr, yarr


def read_2D_table(fdir, fname, nbegin):   # read x-y tabel
    f = open(fdir + fname + '.txt', 'r')
    f.seek(0)  # go to the beginning
    nbeg = int(max(nbegin, 0))
    data = f.read().split('\n')[nbeg:]  # all rows
    Nx = len(data)
    xarr = np.empty(Nx)
    y1arr = np.empty(Nx)
    y2arr = np.empty(Nx)
    for i in range(Nx):
        row = data[i].split('\t')
        xarr[i] = row[0]
        y1arr[i] = row[1]
        y2arr[i] = row[2]
    f.close()
    return xarr, y1arr, y2arr


# luminosity distance (Gpc) at redshift z
def D_L(z):  # unit Gpc
    # Luminosity distance according to the concordance Lambda CDM model
    Omega_m = 0.308
    Omega_Lambda = 1. - Omega_m
    c = 3.e10
    H_0 = 6.78e6
    Nx = 100
    x = np.linspace(0, z, Nx)
    dx = x[2]-x[1]
    temp = 0
    for i in range(Nx):
        temp += dx/np.sqrt(Omega_m*(1+x[i])**3 + Omega_Lambda)
    return c*(1+z)/H_0 * temp/1e3


def dV_dz(z): # unit Gpc^3
    # comoving volume per redshift bin
    Omega_m = 0.308
    Omega_Lambda = 1. - Omega_m
    c = 3e10
    H_0 = 6.78e6
    Nx = 100
    x = np.linspace(0, z, Nx)
    dx = x[2]-x[1]
    temp = 0
    for i in range(Nx):
        temp += dx/sqrt(Omega_m*(1+x[i])**3 + Omega_Lambda)
    return 4*pi*c**3/H_0**3/1e9\
        * temp*temp/sqrt(Omega_m*(1+z)**3 + Omega_Lambda)


def pltimg(ax, xarr, yarr, zarr, min_val, max_val, extend, xlabel, ylabel, zlabel, cmap,
           CB_levels, CB_ticklabels, flag_contour):
    ax.set_xlabel(xlabel, labelpad=-2)
    ax.set_ylabel(ylabel)
    im = ax.imshow(zarr.transpose(),
                   interpolation='nearest', origin='lower',
                   cmap=cmap, aspect='equal', alpha=0.7,
                   extent=(min(xarr), max(xarr),
                           min(yarr), max(yarr)))
    im.set_clim(vmin=min_val, vmax=max_val)

    CB = pl.colorbar(im, ax=ax, ticks=CB_levels, extend=extend,
                     orientation='vertical')
    # CB.ax.set_xticklabels(CB_ticklabels)
    # CB.ax.set_xlabel(zlabel, labelpad=3)
    CB.ax.set_yticklabels(CB_ticklabels)
    CB.ax.set_ylabel(zlabel, labelpad=3)
    CB.ax.minorticks_off()

    if flag_contour:
        X, Y = np.meshgrid(xarr, yarr)
        CS = ax.contour(X, Y, zarr.transpose(),
                        CB_levels, linestyles='solid',
                        colors='k', linewidths=2, alpha=0.5)
        fmt = {}
        for l, s in zip(CS.levels, CB_ticklabels):
            fmt[l] = s
        pl.clabel(CS, CS.levels, inline=True, fmt=fmt,
                  fontsize=30, colors=None)


# solve for pericenter for given energy E <= 1 and angular momentum L
# for Schwarzschild metric
def rp_EL_Sch(E, L, tol):
    # goal is to solve rdot = 0 for the pericenter radius
    rmin_return = 1.
    if E == 1:  # parabolic case
        if L <= 4:  # plunging
            return rmin_return
        return L**2/4 * (1 + sqrt(1 - 16./L**2))
    if (1-E**2)*L**2 > 4./3:  # plunging
        return rmin_return
    rleft = (2 - sqrt(4 - 3*(1-E**2)*L**2))/(3*(1-E**2))
    rright = (2 + sqrt(4 - 3*(1-E**2)*L**2))/(3*(1-E**2))
    # solution is between rmin and rmax

    def y_func(r, E, L):
        return (1-E**2) * r**3 - 2*r**2 + L**2 * r - 2*L**2

    yleft = y_func(rleft, E, L)
    yright = y_func(rright, E, L)
    if yleft < 0 or yright > 0:
        return rmin_return
    # use bisection to find the solution y=0
    while (rright-rleft)/rleft > tol:
        rmid = 0.5 * (rleft + rright)
        ymid = y_func(rmid, E, L)
        if ymid * yleft < 0:
            rright = rmid
        else:
            rleft = rmid
            yleft = ymid
    return 0.5 * (rleft + rright)


def rp_aEL_Kerr_eq(a, E, L, tol):
    # for -1 < a < 1, E<=1, any L
    # goal is to solve rdot = 0 for the pericenter radius
    # for an equatorial Kerr orbit
    rmin_return = 1.
    K = (L - a*E)**2   # useful new variable
    if E == 1:  # parabolic case
        if L <= 2*(1 + sqrt(1-a)):  # plunging
            return rmin_return
        return L**2/4 * (1 + sqrt(1 - 16.*(L-a)**2/L**4))
    Delta_c = 4 - 3*(1-E**2) * (a**2 + K + 2*a*E*K**0.5)
    if Delta_c < 0:  # plunging
        return rmin_return
    rleft = (2 - sqrt(Delta_c))/(3*(1-E**2))
    rright = (2 + sqrt(Delta_c))/(3*(1-E**2))
    # solution is between rmin and rmax

    def y_func(r, a, E, K):
        return (1-E**2) * r**3 - 2*r**2 \
               + (a**2 + K + 2*a*E*K**0.5) * r - 2*K

    yleft = y_func(rleft, a, E, K)
    yright = y_func(rright, a, E, K)
    if yleft < 0 or yright > 0:
        return rmin_return
    # use bisection to find the solution y=0
    while (rright-rleft)/rleft > tol:
        rmid = 0.5 * (rleft + rright)
        ymid = y_func(rmid, a, E, K)
        if ymid * yleft < 0:
            rright = rmid
        else:
            rleft = rmid
            yleft = ymid
    return 0.5 * (rleft + rright)


def ra_aEL_Kerr_eq(a, E, L, tol):
    # for -1 < a < 1, 0 < E < 1, any L
    # goal is to solve rdot = 0 for the apocenter radius
    # for an equatorial Kerr orbit
    rmin_return = 1.
    K = (L - a*E)**2   # useful new variable
    rleft = 2./(3*(1-E**2))
    rright = 2/(1-E**2)
    # solution is between rmin and rmax

    def y_func(r, a, E, K):
        return (1-E**2) * r**3 - 2*r**2 \
               + (a**2 + K + 2*a*E*K**0.5) * r - 2*K

    yleft = y_func(rleft, a, E, K)
    yright = y_func(rright, a, E, K)
    # use bisection to find the solution y=0
    while (rright-rleft)/rleft > tol:
        rmid = 0.5 * (rleft + rright)
        ymid = y_func(rmid, a, E, K)
        if ymid * yleft < 0:
            rright = rmid
        else:
            rleft = rmid
            yleft = ymid
    return 0.5 * (rleft + rright)


def tgw_fit(m1, m2, a0, e0):
    # GW merger time [in yr] for given masses [in Msun]
    # e is eccentricity, a is the separation between the binary [in Rsun]
    beta = (64./5)*cst.G**3 * m1 * m2 * (m1+m2) * cst.Msun**3 / cst.c**5
    Tc = a0**4*cst.Rsun**4/(4*beta)
    T0 = Tc * (1-e0**2)**3.5
    p = 2.66   # a fitting parameter
    return T0 * (768./425 - p*(1-e0*e0)**0.5 + (p-343./425)*(1-e0*e0)**0.8) / cst.yr


def bisec(y, xleft, xright, tol, *args):
    # use bisection method to find xleft<x<xright such that y(x) = 0
    # function y(x) must be monotonic (increasing or decreasing)
    yleft = y(xleft, *args)
    yright = y(xright, *args)
    # print(xleft, xright, *args)
    # return 0

    if yleft * yright > 0:
        print('bisection fails for parameters', xleft, xright, tol, *args)
        return 0
    while (xright-xleft)/xleft > tol:
        xmid = 0.5*(xright+xleft)
        ymid = y(xmid, *args)
        if ymid*yleft > 0:
            xleft = xmid
            yleft = ymid
        else:
            xright = xmid
    return 0.5*(xright+xleft)


def round_sig(x, sig=2):
    # round a number to a given significant digit
    return round(x, sig - int(floor(log10(abs(x)))) - 1)

