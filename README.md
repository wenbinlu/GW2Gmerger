# GW2Gmerger
The codes calculate the probability of having a 2G GW merger between 'm' and 'M' after 'm' receives a kick. The context here is that 'm' is the result of the first generation (1G) merger and the linear momentum taken away by the GW gives rises to the kick, which is assumed to be isotropically distribution with a lognormal distribution in its magnitude.

Using the codes here, one should be able to reproduce the Figure 3 (the key results) of [this paper](https://ui.adsabs.harvard.edu/abs/2021MNRAS.500.1817L/abstract).

After downloading the codes, the first step is to change data_dir in dir_info.py to where you want to store the data and files to be created by the codes.

Then, you should change Ncpu to the number of cores in your computer and then run MP_fmerger_lgnorm.py. It may take at least a few minutes (depending on the number of cores amd the adopted resolution) to generate the results, which are written in a file in your data_dir.

Then, you can run plt_fmerger2D.py to view the results.
