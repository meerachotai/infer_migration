#!/usr/bin/env python3

# allele_freq_norm.py test_20.vcf plot.png table.txt 5 5 10 1000 0.005
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
mig = float(sys.argv[8])

populations = numPopEW * numPopNS

line = ""
index = 0

empty_list = []
freq = [[] for _ in range(populations)] # pre-assign size of matrix

first_line = True
total_snps = 0

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
                total_snps += 1
                for i in range(populations):
                    bottom = top + (sampleSize *2)  # for diploid, x2
                    alt = line[top:bottom].count("1")
                    if(alt != 0): # exclude only ref
                        freq[i].append(alt)
                    top = bottom

                  
title = "Allele Frequency Spectrum: Stepping-Stone Matrix" + "\n$N_e$: " + str(ne) + ", Sample size: " + str(sampleSize) + ", Homogenous Migration Rate: " + str(mig)

write_tabn = open(table, "w")
ylim = 0

freqn_data = [[0 for _ in range(2 * sampleSize)] for _ in range(populations)]  # normalised

for i in range(len(freq)):
    table = collections.Counter(freq[i]) # does histogram-like frequency calculations and inputs into dict
    for key in table:
        freqn_data[i][key-1] = table[key] / total_snps # normalised
    x = math.floor(i/numPopEW)
    y = i % numPopEW
    out = [str(x), str(y), str(i)] # x-y coordinates + population index
    write_tabn.write("\t".join(out) + "\t")
    out_freqn = []
    for j in range(len(freqn_data[i])):
        out_freqn.append(str(freqn_data[i][j]))
    write_tabn.write("\t".join(out_freqn) + "\n")
    
write_tabn.close()