import numpy as np
import cst
from math import pi, log10, sqrt, atan, tan, sin, cos, asin, acos, erf, floor
from multiprocessing import Pool
from dir_info import *
import multiprocessing as mpros

# Ncpu = mpros.cpu_count()
# print("Number of cpu : ", Ncpu)
Ncpu = 12  # manually set the number of logical cores


def cot(x):
    return 1./tan(x)


# ---- consider variations of the following parameters
case_paras = [1, 15, 50]    # M [Msun], m [Msun], vk_mean [km/s]
# ----

# units: mass in Msun, length in AU, time in yr/2pi, G=1, speed in AU*2*pi/yr

# specify the resolution
Nlga, Nlgrp = 30, 30  # low (30, 30); medium (100, 100) ; high resolution (200, 200)

# specify the parameter range
amin, amax = 0.5, 100
rpmin, rpmax = 0.3, amax        # below rp=0.1 AU, the triple may be unstable


lgamin, lgamax = log10(amin), log10(amax)
lga_arr = np.linspace(lgamin, lgamax, Nlga, endpoint=True)
lgrpmin, lgrpmax = log10(rpmin), log10(rpmax)
lgrp_arr = np.linspace(lgrpmin, lgrpmax, Nlgrp, endpoint=True)
eps_small = 1e-10   # very small floor number
lgf_floor = log10(eps_small)

M = case_paras[0]     # BH mass [Msun]
m = case_paras[1]     # total mass for NS+NS [Msun]

Nmu = 50    # using a linear grid in mu from -1 to mu_max, 50 should be sufficient
mu_min = -1.

# the phi grid concentrates near phi=0 and phi=pi
Nphi_half = 30  # 30 should be sufficient
dphi_half = pi/(Nphi_half*(Nphi_half+1))*(np.arange(Nphi_half)+1)
dphi_arr = np.concatenate((dphi_half, np.flip(dphi_half)))
phi_arr = np.cumsum(dphi_arr) - dphi_arr/2  # put at the middle of each bin
dParr_phi = dphi_arr/pi    # probability for each bin dP/dphi*dphi
Nphi = 2*Nphi_half

# the psi grid for orbital phase  (with slight concentration near psi=pi)
Npsi = 50  # 50 should be sufficient
dpsi_min = 2*pi/(5*Npsi)    # dpsi_min = dpsi_max/4
dpsi_arr = dpsi_min + 3*dpsi_min/(Npsi-1) * np.arange(Npsi)
psi_arr = np.cumsum(dpsi_arr) - dpsi_arr/2  # put at the middle of each bin
dParr_psi = np.empty(Npsi, dtype=float)

# kick velocity amplitude distribution (log-normal)
vunit2cgs = 2*pi*cst.AU/cst.yr
lgvk_mean = log10(case_paras[2]*1e5/vunit2cgs)  # amplitude of the kick in machine units
lgvk_sigma = 0.3
Nlgvk = 1000    # this high resolution does not add to computational burden
lgvk_min, lgvk_max = lgvk_mean - 5*lgvk_sigma, lgvk_mean + 5*lgvk_sigma
lgvk_arr = np.linspace(lgvk_min, lgvk_max, Nlgvk, endpoint=True)
dlgvk = lgvk_arr[1] - lgvk_arr[0]
CDF_lgvk_arr = np.array([0.5 + 0.5*erf((lgvk-lgvk_mean)/(sqrt(2)*lgvk_sigma))
                         for lgvk in lgvk_arr])
CDF_lgvk_diff = np.diff(CDF_lgvk_arr)


def calc_lgfmerg_for_each_lgrp(lgrp):
    rp = 10**lgrp
    lgfmerg_lgra_arr = np.empty(Nlga, dtype=float)
    for i_a in range(Nlga):
        a = 10**lga_arr[i_a]
        if a < rp:      # unphysical
            lgfmerg_lgra_arr[i_a] = lgf_floor
            continue
        e = max(1 - rp/a, 1e-5)     # 1e-5 is to avoid numerical problem for e=0 exactly
        # calculate dP/dpsi probability
        temp_CDF_left, temp_CDF_right = 0., 0.
        for i in range(Npsi):
            psi = psi_arr[i] + dpsi_arr[i]/2    # right boundary of each bin
            if i == Npsi - 1:   # the precise right boundary psi = pi (where tan(psi/2) diverges)
                temp_CDF_right = 1.
            else:
                temp_CDF_right = 1/pi * (2*atan(sqrt((1-e)/(1+e)) * tan(psi/2))
                                         - e*sqrt(1-e*e)*sin(psi)/(1+e*cos(psi)))
            dParr_psi[i] = temp_CDF_right - temp_CDF_left
            temp_CDF_left = temp_CDF_right
        # from merger time = Hubble time
        fc = 4.6e-2*(M*m*(M+m)/(20*2.7*22.7))**(2./7) / (a**(8./7)*(1-e*e))

        # then calculate merger fraction for each set of (a, e)
        fmerg = eps_small
        for i in range(Npsi):
            psi = psi_arr[i]
            dP_psi = dParr_psi[i]   # probability for this dpsi bin
            r = a*(1-e*e)/(1+e*cos(psi))
            v = sqrt((M+m)*(2./r - 1./a))
            alpha = asin((1+e*cos(psi))/sqrt(e*e + 2*e*cos(psi) + 1))  # (0, pi/2)
            temp_phi = 0.
            for j in range(Nphi):
                phi = phi_arr[j]
                dP_phi = dParr_phi[j]   # probability for this dphi bin
                xi = cot(alpha)*cos(phi) - sqrt(1./fc - 1)*sin(phi)/sin(alpha)
                mu_max = xi/sqrt(1 + xi*xi)
                dmu = (mu_max-mu_min)/Nmu
                mu = mu_min + dmu/2
                dP_mu = dmu/2.  # probability for each dmu bin
                temp_mu = 0.
                while mu < mu_max:
                    the = acos(mu)
                    ymax = sqrt(mu*mu + r/(2*a-r)) - mu     # avoid unbound orbit
                    eta = (sin(phi)/sin(alpha)/(cot(the) - cot(alpha)*cos(phi)))**2
                    s2 = cos(the) - cot(alpha)*sin(the)*cos(phi)
                    yplus = min((1 + sqrt((eta + 1)*fc - eta))/(-s2*(eta+1)), ymax)
                    yminus = (1 - sqrt((eta + 1)*fc - eta))/(-s2*(eta+1))
                    lgvk_plus = min(log10(yplus*v), lgvk_max)
                    lgvk_minus = max(log10(yminus*v), lgvk_min)
                    if lgvk_plus < lgvk_min or lgvk_minus > lgvk_max or lgvk_plus < lgvk_minus:
                        mu += dmu
                        continue    # no such y for merger to happen
                    i_floor = int(floor((lgvk_plus-lgvk_min)/dlgvk))
                    CDF_lgvk_plus = CDF_lgvk_arr[i_floor] \
                        + CDF_lgvk_diff[i_floor]/dlgvk*(lgvk_plus-lgvk_arr[i_floor])
                    i_floor = int(floor((lgvk_minus-lgvk_min)/dlgvk))
                    CDF_lgvk_minus = CDF_lgvk_arr[i_floor] \
                        + CDF_lgvk_diff[i_floor]/dlgvk*(lgvk_minus-lgvk_arr[i_floor])
                    temp_mu += dP_mu * (CDF_lgvk_plus - CDF_lgvk_minus)
                    mu += dmu
                temp_phi += temp_mu*dP_phi
            fmerg += temp_phi*dP_psi
        # print('for a = %.2e, rp = %.2e, fmerg = %.3e' % (a, rp, fmerg))
        lgfmerg_lgra_arr[i_a] = log10(fmerg)
    return lgfmerg_lgra_arr


savename = 'fmerger2D_M%d_m%.1f_vkmean%d' % (case_paras[0], case_paras[1], case_paras[2])

# parallel calculation starting from here
if __name__ == '__main__':
    with Pool(Ncpu) as po:
        lgfmerg = np.array(po.map(calc_lgfmerg_for_each_lgrp, lgrp_arr))

    # write the solution into a file
    print('writing:' + data_dir+savename+'.txt')
    with open(data_dir+savename+'.txt', 'w') as f:
        f.write('lgamin\tlgamax\tNlga\t%.5f\t%.5f\t%d\n' % (lgamin, lgamax, Nlga))
        f.write('lgrpmin\tlgrpmax\tNlgrp\t%.5f\t%.5f\t%d\n' % (lgrpmin, lgrpmax, Nlgrp))
        f.write('lgfmerg(Nlga, Ne)')
        for i_a in range(Nlga):
            f.write('\n')
            for i_rp in range(Nlgrp):
                if i_rp == 0:
                    f.write('%.5f' % lgfmerg[i_rp, i_a])
                else:
                    f.write('\t%.5f' % lgfmerg[i_rp, i_a])
