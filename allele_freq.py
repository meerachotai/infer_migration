# python3 allele_freq.py sub_test.vcf plot.png 5 5 10 60000
import pandas as pd
import io
import math
import matplotlib.pyplot as plt
import sys

input_vcf = str(sys.argv[1])
new_file = str(sys.argv[2])
numPopEW = int(sys.argv[3]) 
numPopNS = int(sys.argv[4]) 
sampleSize = int(sys.argv[5])
ylim = int(sys.argv[6])

populations = numPopEW * numPopNS

line = ""
index = 0

empty_list = []
freq = [[] for _ in range(populations)] # pre-assign size of matrix

first_line = True
with open(input_vcf, 'r') as f:
    for line in f:
        if line.startswith("##"):
            continue
        else:
            if(first_line):
                line = line.strip().split("\t")
                for num,string in enumerate(line):
                    if(any(char.isdigit() for char in string)):
                        index = num
                        break
                first_line = False
                continue
            else:
                line = line.strip().split("\t")
                # 0 = REF, 1 = ALT 
                line = "".join(line[index:]).replace("|", "") # separating the diploid counts
                top = 0
                for i in range(populations):
                    bottom = top + (sampleSize *2)  # for diploid, x2
                    alt = line[top:bottom].count("1")
                    if(alt != 0): # exclude only ref
                    	freq[i].append(alt)
                    top = bottom
                
                  
# print(freq)

fig, axs = plt.subplots(numPopEW,numPopNS, figsize=(15, 10)) #, facecolor='w', edgecolor='k')
axs = axs.ravel()
for i in range(populations):
    axs[i].hist(freq[i], bins = 20) #, align = "left", rwidth = 1)
    axs[i].set_ylim(0,ylim) # set upper limit, for comparison
fig.tight_layout()

plt.savefig(new_file)
