import numpy as np
import pylab as pl
from my_func import pltimg, round_sig
from math import pi, log10, sqrt, floor, ceil
from dir_info import *

fname = 'fmerger2D_M1_m15.0_vkmean50'
savedir = data_dir
savename = fname

lgfmerg_floor = -10
# read lgfmerger from a file
with open(data_dir+fname+'.txt', 'r') as f:
    row1 = f.readline().strip('\n').split('\t')
    row2 = f.readline().strip('\n').split('\t')
    lgamin, lgamax, Nlga = float(row1[3]), float(row1[4]), int(float(row1[5]))
    lgrpmin, lgrpmax, Nlgrp = float(row2[3]), float(row2[4]), int(float(row2[5]))
    lga_arr = np.linspace(lgamin, lgamax, Nlga)
    lgrp_arr = np.linspace(lgrpmin, lgrpmax, Nlgrp)
    lgfmerg = np.empty((Nlga, Nlgrp), dtype=float)
    row = f.readline()  # skip this line
    for i_a in range(Nlga):
        row = f.readline().strip('\n').split('\t')
        lgfmerg[i_a, :] = row
        # print(lgfmerg[i_a, :])

lgfmerg_real = np.ma.masked_where(lgfmerg < lgfmerg_floor+1, lgfmerg)

plt_real = lgfmerg_real
xarr = lga_arr
yarr = lgrp_arr

# --- the following is the masked array (not used)
# lgfmerg_masked = np.ma.masked_where(lgfmerg > lgfmerg_floor+1, lgfmerg)
# plt_masked = lgfmerg_masked

Ncont = 6   # number of contours
xlabel = r'$\log (a/\mathrm{AU})$'
ylabel = r'$\log (r_{\rm p}/\mathrm{AU})$'
pltlabel = r'$\log f_{\rm merger}$'

fig = pl.figure(figsize=(13, 10))
ax = fig.add_axes([0.05, 0.10, 0.92, 0.85])

# min_val, max_val = np.min(plt_real), np.max(plt_real)
min_val, max_val = np.min(plt_real), np.max(plt_real)
extend = 'neither'

step = round_sig((max_val-min_val)/Ncont, sig=2)  # round to significant figure = 2
res = step
potCB_levels = np.arange(ceil(min_val/res)*res, floor(max_val/res)*res + step, step)
potCB_ticklabels = [('%.1f' % num).replace('.0', '') for num in potCB_levels]

pltimg(ax, xarr, yarr, plt_real, min_val, max_val, extend, xlabel, ylabel, pltlabel, 'BrBG',
       potCB_levels, potCB_ticklabels, flag_contour=True)

# --- overplot the masked region (not used)
# im = ax.imshow(plt_masked.transpose(),
#                interpolation='nearest', origin='lower',
#                cmap='twilight', aspect='auto',
#                extent=(min(xarr), max(xarr),
#                        min(yarr), max(yarr)))

color = 'olive'
ax.plot(lga_arr, lga_arr, '-', lw=5, color=color)

ax.text(-0.3, -0.1, r'$r_{\rm p}=a$', rotation=42, color=color, fontsize=40)
ax.text(-0.1, 0.9, 'unphysical', fontsize=40, color='k')

pl.subplots_adjust(bottom=0.11, left=0.13, top=0.98, right=1.01)
fig.savefig(savedir + savename + '.pdf')
