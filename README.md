# GW2Gmerger
The codes calculate the probability of having a 2G GW merger between 'm' and 'M' after m receives a kick as a result of 1G merger (which created 'm').

After downloading the codes, the first step is to change data_dir in dir_info.py to where you want to store the data and files to be created by the codes.

Then, you should change Ncpu to the number of cores in your machine and then run MP_fmerger_lgnorm.py. It may take up to a few minutes (depending on the number of cores) to generate the results, which are written in a file in your data_dir.

Then, you can run plt_fmerger2D.py to view the results.
