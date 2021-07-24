#!/usr/bin/env python3

# generateMig.py 0.0001 0.01 1 5 100 mig_val.txt
import random
from scipy.stats import loguniform
import numpy as np
import sys

lower = float(sys.argv[1])
upper = float(sys.argv[2])
# 0 = symmetric, 1 = one different, 2 = all different
asym = int(sys.argv[3])
seed = int(sys.argv[4])
n = int(sys.argv[5])
outfile = str(sys.argv[6])

# upper = 0.01
# lower = 0.0001
# seed = 5
# n = 100

np.random.seed(seed=seed)
random.seed(seed)

file = open(outfile, "w")

if(asym == 0):
    for i in range(0,n): # all equal
        mig_sym = str(round(loguniform.rvs(lower, upper),6))
        out = "\t".join([mig_sym, mig_sym, mig_sym, mig_sym])
        file.write(out + "\n")
elif(asym == 1): # three equal, one different
    for i in range(0,n):
        mig_sym = str(round(loguniform.rvs(lower, upper),6))
        mig_asym = str(round(loguniform.rvs(lower, upper),6))
        out_list = [mig_sym, mig_sym, mig_sym, mig_asym]
#         random.shuffle(out_list)
        out = "\t".join(out_list)
        file.write(out + "\n")
elif(asym == 2):
	for i in range(0,n): # all different
		out_list = []
		for j in range(0,4):
			mig_asym = str(round(loguniform.rvs(lower, upper),6))
			out_list.append(mig_asym)
		out = "\t".join(out_list)
		file.write(out + "\n")

file.close()
