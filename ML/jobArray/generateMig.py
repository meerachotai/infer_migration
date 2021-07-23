#!/usr/bin/env python3

# generateMig.py 0.0001 0.01 1 5 100
import random
from scipy.stats import loguniform
import numpy as np
import sys

lower = float(sys.argv[1])
upper = float(sys.argv[2])
asym = int(sys.argv[3]) # operates like boolean 0/1
seed = int(sys.argv[4])
n = int(sys.argv[5])

# upper = 0.01
# lower = 0.0001
# asym = 0
# seed = 5
# n = 100

np.random.seed(seed=seed)
random.seed(seed)

file = open("mig_val.txt", "w")

if(asym != 0): # three equal, one different
    for i in range(0,n):
        mig_sym = str(round(loguniform.rvs(lower, upper),4))
        mig_asym = str(round(loguniform.rvs(lower, upper),4))
        out_list = [mig_sym, mig_sym, mig_sym, mig_asym]
        random.shuffle(out_list)
        out = "\t".join(out_list)
        file.write(out + "\n")
else:
    for i in range(0,2): # all equal
        mig_sym = str(round(loguniform.rvs(lower, upper),4))
        out = "\t".join([mig_sym, mig_sym, mig_sym, mig_sym])
        file.write(out + "\n")

file.close()
