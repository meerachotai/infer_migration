#!/usr/bin/env python3

# allele_freq.py test_20.vcf plot.png table.txt 5 5 10 1000 0.005
import io
import math
import matplotlib.pyplot as plt
import collections
import sys

input_vcf = str(sys.argv[1])
new_file = str(sys.argv[2])
table = str(sys.argv[3])
numPopEW = int(sys.argv[4]) 
numPopNS = int(sys.argv[5]) 
sampleSize = int(sys.argv[6])
ne = int(sys.argv[7])
mig = str(sys.argv[8])

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
                
                  
title = "Allele Frequency Spectrum: Stepping-Stone Matrix" + "\n$N_e$: " + str(ne) + ", Sample size: " + str(sampleSize) + ", Homogenous Migration Rate: " + str(mig)

write_tab = open(table, "w")

ylim = 0

freq_data = [[0 for _ in range(2 * sampleSize)] for _ in range(populations)] # pre-assign size of matrix

for i in range(len(freq)):
    table = collections.Counter(freq[i])
    for key in table:
        if(table[key] > ylim):
            ylim = table[key]
        freq_data[i][key-1] = table[key] # key-1 because we're ignoring 0 counts
    x = math.floor(i/numPopEW) # goes row-by-row
    y = i % numPopEW
    out = [str(x), str(y), str(i)]
    write_tab.write("\t".join(out) + "\t")
    out3 = []
    for element in freq_data[i]:
        out3.append(str(element))
    write_tab.write("\t".join(out3) + "\n")
           
write_tab.close()

fig, axs = plt.subplots(numPopEW,numPopNS, figsize=(15, 10)) #, facecolor='w', edgecolor='k')
axs = axs.ravel()
for i in range(populations):
    axs[i].hist(freq[i], bins = 20) #, align = "left", rwidth = 1)
    axs[i].set_ylim(0,ylim) # set upper limit, for comparison
fig.suptitle(title)
fig.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig(new_file)
